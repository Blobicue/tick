//! BitLinear — the tq1_0 × q8_K matvec, mirroring the reference kernel
//! exactly (ggml-cpu `ggml_vec_dot_tq1_0_q8_K_generic`, arm/quants.c
//! NEON path): per-256-block INTEGER accumulation (order-free, exact),
//! then `sumf += (float)sum * (d_x * d_y)` accumulated sequentially in
//! block order. The NEON kernel's lane reordering changes nothing
//! because integer addition is exact — this scalar mirror is
//! bit-identical to what the frozen binary computes.

pub const QK_K: usize = 256;

/// A dequantized tq1_0 tensor: per-block f16 scale + ternary rows.
/// One block covers QK_K consecutive elements of a row-major
/// \[rows, cols\] matrix (gguf ne\[0\] = cols contiguous).
pub struct TqTensor {
    pub rows: usize,
    pub cols: usize,
    /// blocks along ne\[0\]; row-major: row r, block b at r * n_blocks + b
    pub n_blocks: usize,
    /// per block: 256 trits as 2-bit (trit+1) packed 4-per-byte,
    /// straight element order — 64 bytes per block (the unpacked i8
    /// form streamed 4x the DRAM traffic and was the token bottleneck)
    pub packed: Vec<u8>,
    /// per block: the f16 block scale (as f32)
    pub d: Vec<f32>,
}

/// unpack element j of a packed block (2 bits, trit+1)
#[inline]
pub fn trit_at(packed: &[u8], j: usize) -> i32 {
    ((packed[j >> 2] >> (2 * (j & 3))) & 3) as i32 - 1
}

/// pack trits (i8 in {-1,0,1}) into 2-bit little-element order
pub fn pack_trits(trits: &[i8]) -> Vec<u8> {
    assert_eq!(trits.len() % 4, 0);
    let mut out = vec![0u8; trits.len() / 4];
    for (j, &t) in trits.iter().enumerate() {
        out[j >> 2] |= ((t + 1) as u8) << (2 * (j & 3));
    }
    out
}

/// ggml nearest_int — the magic-number round-to-nearest-EVEN
/// (add 1.5·2^23 and read the mantissa). Half-away rounding is NOT
/// equivalent here: the q8 scale makes half-integers common, and this
/// difference alone cost ~1% per matvec (found via tensorprobe diff).
#[inline]
// unused helper kept verbatim from the reference quantizer for provenance
#[allow(dead_code)]
fn nearest_int(f: f32) -> i32 {
    let val = f + 12582912.0f32;
    ((val.to_bits() as i32) & 0x007f_ffff) - 0x0040_0000
}

/// The CONTRACTED form actually running in the compiled ggml: the
/// optimizer fuses `nearest_int(iscale * x)` into
/// `fma(iscale, x, 12582912)` — one rounding of the EXACT product,
/// which shifts exact-tie outcomes vs rounding the product first.
/// Verified empirically against libggml-cpu's quantize_row_q8_K.
#[inline]
fn nearest_int_scaled(scale: f32, x: f32) -> i32 {
    let val = x.mul_add(scale, 12582912.0f32);
    ((val.to_bits() as i32) & 0x007f_ffff) - 0x0040_0000
}

/// One quantized activation block: q8 values + scale d (may be
/// negative; sign cancels by construction — see the reference).
pub struct Q8Block {
    pub q: [i8; QK_K],
    /// slice-major copy of q for the packed kernel: element 64*k + j'
    /// holds q[4*j' + k] — one shuffle pass at quantize time replaces
    /// a per-row 64-byte table gather
    pub qp: [i8; QK_K],
    pub d: f32,
}

/// quantize_row_q8_K_ref verbatim (bsums omitted — unused by the
/// tq1_0 dot): amax picks the SIGNED max, iscale = -127/max.
/// bench seam: quantize through the public API
pub fn quantize_q8_k_pub(x: &[f32]) -> Vec<Q8Block> {
    quantize_q8_k(x)
}

pub fn quantize_q8_k(x: &[f32]) -> Vec<Q8Block> {
    assert!(x.len().is_multiple_of(QK_K));
    let nb = x.len() / QK_K;
    let mut out = Vec::with_capacity(nb);
    for b in 0..nb {
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            out.push(Q8Block {
                q: [0; QK_K],
                qp: [0; QK_K],
                d: 0.0,
            });
            continue;
        }
        let iscale = -127.0f32 / max;
        let mut q = [0i8; QK_K];
        for j in 0..QK_K {
            let v = nearest_int_scaled(iscale, xb[j]);
            q[j] = v.min(127) as i8;
        }
        let mut qp = [0i8; QK_K];
        for k in 0..4 {
            for jp in 0..64 {
                qp[k * 64 + jp] = q[4 * jp + k];
            }
        }
        out.push(Q8Block {
            q,
            qp,
            d: 1.0 / iscale,
        });
    }
    out
}

/// The reference vec_dot for one output row of a tq1_0 matrix against
/// a quantized activation. `w` is the whole tensor, `r` the output row.
/// SDOT inner kernel: the trit×q8 block sum is EXACT integer
/// arithmetic, so any lane structure yielding the same i32 sum is
/// bit-identical — 16 i8 products per SDOT into i32 lanes.
/// NEON inner kernel: widen i8→i16, MLA into two i16x8 accumulators
/// (each lane sums <=32 products of |trit*q| <= 128, so max 4096 —
/// no overflow), then pairwise-widen to i32. Exact integer math, so
/// the result equals the scalar loop bit-for-bit.
#[cfg(target_arch = "aarch64")]
fn dot_tq_neon(w: &TqTensor, r: usize, x: &[Q8Block]) -> f32 {
    use std::arch::aarch64::*;
    let mut sumf = 0.0f32;
    for b in 0..w.n_blocks {
        let pbase = r * w.n_blocks * 64 + b * 64;
        let packed = &w.packed[pbase..pbase + 64];
        let _q = &x[b].q;
        unsafe {
            let mut acc_lo = vdupq_n_s16(0);
            let mut acc_hi = vdupq_n_s16(0);
            // process 64 elements = 16 packed bytes per iteration;
            // qp holds q slice-major (qp[k*64 + j'] = q[4j' + k]) so
            // slice k pairs with a plain 16-byte load at offset 16k
            let qp = &x[b].qp;
            let mut p = 0; // packed byte base = 16*j' offset within each slice
            while p < 64 {
                let pv = vld1q_u8(packed.as_ptr().add(p));
                let qk0 = vld1q_u8(qp.as_ptr().add(p) as *const u8);
                let qk1 = vld1q_u8(qp.as_ptr().add(64 + p) as *const u8);
                let qk2 = vld1q_u8(qp.as_ptr().add(128 + p) as *const u8);
                let qk3 = vld1q_u8(qp.as_ptr().add(192 + p) as *const u8);
                let t0 = vandq_u8(pv, vdupq_n_u8(3));
                let t1 = vandq_u8(vshrq_n_u8(pv, 2), vdupq_n_u8(3));
                let t2 = vandq_u8(vshrq_n_u8(pv, 4), vdupq_n_u8(3));
                let t3 = vandq_u8(vshrq_n_u8(pv, 6), vdupq_n_u8(3));
                // widen pairs to i16 and fma: (t-1) * q
                macro_rules! slice_mla {
                    ($t:expr, $qk:expr, $lo:expr, $hi:expr) => {{
                        let tw0 = vsubq_s16(
                            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8($t))),
                            vdupq_n_s16(1),
                        );
                        let qw0 = vmovl_s8(vget_low_s8(vreinterpretq_s8_u8($qk)));
                        $lo = vmlaq_s16($lo, tw0, qw0);
                        let tw1 = vsubq_s16(
                            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8($t))),
                            vdupq_n_s16(1),
                        );
                        let qw1 = vmovl_s8(vget_high_s8(vreinterpretq_s8_u8($qk)));
                        $hi = vmlaq_s16($hi, tw1, qw1);
                    }};
                }
                slice_mla!(t0, qk0, acc_lo, acc_hi);
                slice_mla!(t1, qk1, acc_lo, acc_hi);
                slice_mla!(t2, qk2, acc_lo, acc_hi);
                slice_mla!(t3, qk3, acc_lo, acc_hi);
                p += 16;
            }
            let wide = vpadalq_s16(vpadalq_s16(vdupq_n_s32(0), acc_lo), acc_hi);
            let sum = vaddvq_s32(wide);
            let d = w.d[r * w.n_blocks + b] * x[b].d;
            sumf = d.mul_add(sum as f32, sumf);
        }
    }
    sumf
}

pub fn dot_tq1_0_q8_k(w: &TqTensor, r: usize, x: &[Q8Block]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return dot_tq_neon(w, r, x);
    }
    let mut sumf = 0.0f32;
    for b in 0..w.n_blocks {
        let packed = &w.packed[r * w.n_blocks * 64..r * w.n_blocks * 64 + 64];
        let xb = &x[b];
        let mut sum = 0i32;
        for j in 0..QK_K {
            sum += trit_at(packed, j) * xb.q[j] as i32;
        }
        // per-block scale is (row, block)-indexed — row 0's scales are
        // NOT everyone's (FFN tensors have all-zero blocks). The C
        // kernel's `sumf += d * (float)sum` compiles to an FMA (single
        // rounding) under default -ffp-contract; mirror with mul_add.
        let d = w.d[r * w.n_blocks + b] * xb.d;
        sumf = d.mul_add(sum as f32, sumf);
    }
    sumf
}

/// BitLinear: quantize the (already RMSNorm'd) activation row and
/// project through the ternary weights, kernel-exact.
/// Two/three matvecs sharing one join via a flat buffer with raw
/// disjoint writes (the per-row results are join-structure independent).
pub fn bitlinear2(a: &TqTensor, b: &TqTensor, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), a.cols);
    assert_eq!(x.len(), b.cols);
    let xq = quantize_q8_k(x);
    let n1 = a.rows;
    let mut flat = vec![0.0f32; a.rows + b.rows];
    let total = flat.len();
    let sp = crate::par::SendPtr(flat.as_mut_ptr());
    let base = &sp;
    crate::par::for_chunks(total, 256, |s, e| unsafe {
        let base = base.0;
        for i in s..e {
            *base.add(i) = if i < n1 {
                dot_tq1_0_q8_k(a, i, &xq)
            } else {
                dot_tq1_0_q8_k(b, i - n1, &xq)
            };
        }
    });
    let mut oa = flat;
    let ob = oa.split_off(n1);
    (oa, ob)
}

pub fn bitlinear3(
    a: &TqTensor,
    b: &TqTensor,
    c: &TqTensor,
    x: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), a.cols);
    assert_eq!(x.len(), b.cols);
    assert_eq!(x.len(), c.cols);
    let xq = quantize_q8_k(x);
    let n1 = a.rows;
    let n2 = n1 + b.rows;
    let mut flat = vec![0.0f32; n2 + c.rows];
    let total = flat.len();
    let sp = crate::par::SendPtr(flat.as_mut_ptr());
    let base = &sp;
    crate::par::for_chunks(total, 256, |s, e| unsafe {
        let base = base.0;
        for i in s..e {
            *base.add(i) = if i < n1 {
                dot_tq1_0_q8_k(a, i, &xq)
            } else if i < n2 {
                dot_tq1_0_q8_k(b, i - n1, &xq)
            } else {
                dot_tq1_0_q8_k(c, i - n2, &xq)
            };
        }
    });
    let mut oa = flat;
    let oc = oa.split_off(n2);
    let ob = oa.split_off(n1);
    (oa, ob, oc)
}

pub fn bitlinear(w: &TqTensor, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), w.cols, "bitlinear width mismatch");
    let xq = quantize_q8_k(x);
    let mut out = vec![0.0f32; w.rows];
    crate::par::par_rows(&mut out, |r| dot_tq1_0_q8_k(w, r, &xq));
    out
}

/// Decode raw tq1_0 bytes into a TqTensor. Raw layout per block:
/// qs[48] (5 trits/byte, ceil-div base-3), qh[4] (4 trits/byte), d(f16).
/// TQ1_0 byte per base-3 value: v = Σ t_l·3^l (t = trit+1,
/// little-endian digits) → the byte whose `wrapping_mul(POW3[l])*3>>8`
/// extracts digit l. Built by running the decode over all 256 bytes —
/// exact inverse by construction (the closed form is ggml's ceil-chain
/// recursion; this table sidesteps it).
const TQ_ENCODE: [u8; 243] = [
    0, 86, 171, 29, 114, 200, 57, 143, 228, 10, 95, 181, 38, 124, 209, 67, 152, 238, 19, 105, 190,
    48, 133, 219, 76, 162, 247, 4, 89, 174, 32, 117, 203, 61, 146, 231, 13, 98, 184, 42, 127, 212,
    70, 155, 241, 23, 108, 193, 51, 136, 222, 80, 165, 250, 7, 92, 177, 35, 121, 206, 64, 149, 234,
    16, 102, 187, 45, 130, 215, 73, 159, 244, 26, 111, 196, 54, 140, 225, 83, 168, 253, 2, 87, 172,
    30, 115, 201, 58, 144, 229, 11, 96, 182, 39, 125, 210, 68, 153, 239, 21, 106, 191, 49, 134,
    220, 77, 163, 248, 5, 90, 175, 33, 118, 204, 62, 147, 232, 14, 100, 185, 43, 128, 213, 71, 156,
    242, 24, 109, 194, 52, 137, 223, 81, 166, 251, 8, 93, 179, 36, 122, 207, 65, 150, 235, 17, 103,
    188, 46, 131, 216, 74, 160, 245, 27, 112, 198, 55, 141, 226, 84, 169, 254, 3, 88, 173, 31, 116,
    202, 59, 145, 230, 12, 97, 183, 41, 126, 211, 69, 154, 240, 22, 107, 192, 50, 135, 221, 78,
    164, 249, 6, 91, 176, 34, 120, 205, 63, 148, 233, 15, 101, 186, 44, 129, 214, 72, 157, 243, 25,
    110, 195, 53, 139, 224, 82, 167, 252, 9, 94, 180, 37, 123, 208, 66, 151, 236, 18, 104, 189, 47,
    132, 218, 75, 161, 246, 28, 113, 199, 56, 142, 227, 85, 170, 255,
];

pub fn decode_tq1_0(raw: &[u8], rows: usize, cols: usize) -> Result<TqTensor, String> {
    if !cols.is_multiple_of(QK_K) {
        return Err(format!("cols {cols} not a multiple of {QK_K}"));
    }
    let n_blocks = cols / QK_K;
    let want = rows * n_blocks * 54;
    if raw.len() != want {
        return Err(format!("tq1_0 raw {} bytes, want {want}", raw.len()));
    }
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let mut trits = vec![0i8; rows * n_blocks * QK_K];
    let mut d = vec![0f32; rows * n_blocks];
    // (trits is a load-time temporary; steady state keeps the packed form)
    for rb in 0..(rows * n_blocks) {
        let base = rb * 54;
        let qs = &raw[base..base + 48];
        let qh = &raw[base + 48..base + 52];
        let dh = u16::from_le_bytes([raw[base + 52], raw[base + 53]]);
        d[rb] = half_to_f32(dh);
        let out = &mut trits[rb * QK_K..(rb + 1) * QK_K];
        // group 1: bytes 0..32 — elements l*32+m, l=0..5
        for j in 0..2usize {
            // two 16-byte halves to avoid the odd 32-byte stride split
            for l in 0..5usize {
                for m in 0..16usize {
                    let byte = qs[j * 16 + m];
                    let t = ((byte.wrapping_mul(POW3[l]) as u16).wrapping_mul(3)) >> 8;
                    out[l * 32 + j * 16 + m] = t as i8 - 1;
                }
            }
        }
        // group 2: bytes 32..48 — elements 160 + l*16+m
        for l in 0..5usize {
            for m in 0..16usize {
                let byte = qs[32 + m];
                let t = ((byte.wrapping_mul(POW3[l]) as u16).wrapping_mul(3)) >> 8;
                out[160 + l * 16 + m] = t as i8 - 1;
            }
        }
        // qh: 4 trits/byte — elements 240 + l*4+j
        for l in 0..4usize {
            for j in 0..4usize {
                let byte = qh[j];
                let t = ((byte.wrapping_mul(POW3[l]) as u16).wrapping_mul(3)) >> 8;
                out[240 + l * 4 + j] = t as i8 - 1;
            }
        }
    }
    let packed = pack_trits(&trits);
    Ok(TqTensor {
        rows,
        cols,
        n_blocks,
        packed,
        d,
    })
}

/// Encode a TqTensor back to raw tq1_0 blocks (54 bytes/block) — the
/// trainer's export seam (M4 emits tickfile directly). Round-trips
/// decode_tq1_0 exactly (fuzz-tested over random trits + scales).
pub fn encode_tq1_0(t: &TqTensor) -> Vec<u8> {
    let nb = t.n_blocks;
    let mut out = vec![0u8; t.rows * nb * 54];
    for rb in 0..(t.rows * nb) {
        let base = rb * 54;
        let pbase = rb * 64;
        let mut qs = [0u8; 48];
        let mut qh = [0u8; 4];
        let tri = |j: usize| -> u8 { (t.packed[pbase + (j >> 2)] >> (2 * (j & 3))) & 3 };
        // group 1: bytes 0..32 hold elements {l*32 + j*16 + m} for the
        // five l at each (j, m); group 2: bytes 32..48 hold elements
        // {160 + l*16 + m}
        for j in 0..2usize {
            for m in 0..16usize {
                let mut v = 0usize;
                for l in 0..5usize {
                    v += tri(l * 32 + j * 16 + m) as usize * POW3_TABLE[l];
                }
                qs[j * 16 + m] = TQ_ENCODE[v];
            }
        }
        for m in 0..16usize {
            let mut v = 0usize;
            for l in 0..5usize {
                v += tri(160 + l * 16 + m) as usize * POW3_TABLE[l];
            }
            qs[32 + m] = TQ_ENCODE[v];
        }
        for j in 0..4usize {
            let mut v = 0usize;
            for l in 0..4usize {
                v += tri(240 + l * 4 + j) as usize * POW3_TABLE[l];
            }
            qh[j] = TQ_ENCODE[v];
        }
        out[base..base + 48].copy_from_slice(&qs);
        out[base + 48..base + 52].copy_from_slice(&qh);
        let dh = f32_to_f16(t.d[rb]);
        out[base + 52..base + 54].copy_from_slice(&dh.to_le_bytes());
    }
    out
}

const POW3_TABLE: [usize; 5] = [1, 3, 9, 27, 81];

/// IEEE 754 binary16 → f32 (ggml's FP16_TO_FP32 on little-endian).
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if man == 0 {
                sign
            } else {
                // subnormal: normalize. value = man × 2^-24 = (1.frac)
                // × 2^(-14-s) after s left-shifts, so the f32 exponent
                // field is 113 - s. Starting e at -1 DOUBLED every
                // subnormal — caught via ggml mul_mat on real head data
                // (0x805f → -5.66244e-6, not -1.13249e-5).
                let mut e = 0i32;
                let mut m = man;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e += 1;
                }
                sign | (((113 - e) as u32) << 23) | ((m & 0x3ff) << 13)
            }
        }
        0x1f => sign | (0xff << 23) | (man << 13),
        e => sign | ((e + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// f32 → IEEE binary16 bits, round-to-nearest-even (ggml FP32_TO_FP16).
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v;
    let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
    let mut mant = x.to_bits() & 0x007f_ffff;
    let exp = ((x.to_bits() >> 23) & 0xff) as i32;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        mant |= 0x0080_0000;
        let shift = 14 - e;
        let m = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let mut m = m;
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    let mut m = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            return sign | (((e + 1) as u16) << 10);
        }
    }
    sign | ((e as u16) << 10) | m as u16
}

/// f64 → IEEE binary16 bits, round-to-nearest-even — needed for the
/// exact f16 fused multiply-add (the exact product+acc sum can need
/// more precision than f32 carries).
pub fn f64_to_f16(v: f64) -> u16 {
    let x = v;
    let sign = ((x.to_bits() >> 48) & 0x8000) as u16;
    let mut mant = x.to_bits() & 0x000f_ffff_ffff_ffff;
    let exp = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if exp == 0x7ff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 1023 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if x == 0.0 {
        return sign;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        mant |= 0x0010_0000_0000_0000;
        // f64 carries 29 more mantissa bits than f32: the shift that
        // lands the 53-bit significand in 10-bit subnormal range is
        // 43 - e (the f32 version's 14 - e was wrong here)
        let shift = 43 - e;
        let m = mant >> shift;
        let rem = mant & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        let mut m = m;
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    let mut m = mant >> 42;
    let rem = mant & 0x3ff_ffff_ffff; // 42 bits
    let half = 1u64 << 41;
    if rem > half || (rem == half && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            return sign | (((e + 1) as u16) << 10);
        }
    }
    sign | ((e as u16) << 10) | m as u16
}

/// ggml_vec_scale_f16 as COMPILED (fitted bit-exact vs ggml's own
/// flash_attn_ext on real model operands): the scalar stays RAW f32,
/// the multiply runs in f32, the result rounds ONCE to f16.
pub fn vec_scale_f16(y: &mut [u16], v: f32) {
    for b in y.iter_mut() {
        *b = f32_to_f16(half_to_f32(*b) * v);
    }
}

/// ggml_vec_mad_f16 as COMPILED (bit-exact vs ggml flash_attn_ext):
/// y = f16(fma_f32(x, RAW f32 v, y_widened)) — the scalar is NOT
/// f16-quantized; the f32 FMA rounds once to f32 and the store rounds
/// once to f16 (double rounding is the actual hardware behavior).
pub fn vec_mad_f16(y: &mut [u16], x: &[u16], v: f32) {
    for i in 0..y.len() {
        y[i] = f32_to_f16(half_to_f32(x[i]).mul_add(v, half_to_f32(y[i])));
    }
}

/// ggml_vec_dot_f16 on this machine (SVE, svcnth=8): 8-chunk main
/// loop, chunks assigned to 4 widened (lo/hi) f32 lane accumulators
/// with FUSED multiply-add, pairwise reduction, horizontal add.
/// Mirrors vec.cpp's __ARM_FEATURE_SVE path; verified against the
/// actual libggml-cpu function on adversarial inputs.
/// NEON form of dot_f16_sve — the SAME arithmetic order (4 contiguous
/// groups per 16-step, f32x4 fma chains, (s0+s2)+(s1+s3) lane-wise,
/// faddp pairwise reduce, f64 leftovers), vectorized. Bit-exact vs the
/// scalar form by construction; verified on all 128256 head rows.
/// Widen 4 f16 bits to f32 with the EXACT semantics of half_to_f32
/// (unit-tested incl. subnormals, -0, inf):
///   em   = (h & 0x7FFF) << 13   — exp16 at bits 23-27, man at 13-22
///   norm = em + 0x3800_0000     — exponent rebias +112 (no carry:
///                                  man bits 13-22 don't reach bit 23)
///   exp16==0: em as an f32 subnormal = m·2^-136; ×2^112 (exact
///   power of two) = m·2^-24; ±0 stays ±0. f16 inf/NaN rebias to the
///   f32 encodings (0x7C00 -> 0x7F80_0000).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn widen_f16x4(h: std::arch::aarch64::uint16x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let em = vshll_n_u16(vand_u16(h, vdup_n_u16(0x7FFF)), 13);
    let sign = vshlq_n_u32(vshll_n_u16(vshr_n_u16(h, 15), 16), 15);
    let norm = vaddq_u32(em, vdupq_n_u32(0x3800_0000));
    let subv = vmulq_f32(vreinterpretq_f32_u32(em), vdupq_n_f32(2.0f32.powi(112)));
    let m = vceqzq_u32(vandq_u32(em, vdupq_n_u32(0x0F80_0000)));
    let val = vbslq_f32(m, subv, vreinterpretq_f32_u32(norm));
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(val), sign))
}

/// Head dot straight from f16 rows (half the streaming of pre-widened
/// f32) against the pre-widened x — SAME order as dot_f32x16_neon /
/// dot_f16_sve: 4 contiguous groups per 16-step, fma chains,
/// (s0+s2)+(s1+s3) lane-wise, faddp-style pairwise reduce, f64
/// leftovers. Bit-exact vs dot_f16_sve(row, x16).
#[cfg(target_arch = "aarch64")]
pub fn dot_f16row_f32x_neon(row: &[u16], xw: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = row.len();
    debug_assert_eq!(n, xw.len());
    let np = n & !15;
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i < np {
            let h0 = vld1_u16(row.as_ptr().add(i));
            let h1 = vld1_u16(row.as_ptr().add(i + 4));
            let h2 = vld1_u16(row.as_ptr().add(i + 8));
            let h3 = vld1_u16(row.as_ptr().add(i + 12));
            let a0 = widen_f16x4(h0);
            let b0 = vld1q_f32(xw.as_ptr().add(i));
            s0 = vfmaq_f32(s0, a0, b0);
            let a1 = widen_f16x4(h1);
            let b1 = vld1q_f32(xw.as_ptr().add(i + 4));
            s1 = vfmaq_f32(s1, a1, b1);
            let a2 = widen_f16x4(h2);
            let b2 = vld1q_f32(xw.as_ptr().add(i + 8));
            s2 = vfmaq_f32(s2, a2, b2);
            let a3 = widen_f16x4(h3);
            let b3 = vld1q_f32(xw.as_ptr().add(i + 12));
            s3 = vfmaq_f32(s3, a3, b3);
            i += 16;
        }
        let c = vaddq_f32(vaddq_f32(s0, s2), vaddq_f32(s1, s3));
        let mut tot = vaddvq_f32(c) as f64;
        for k in np..n {
            tot += (half_to_f32(row[k]) * xw[k]) as f64;
        }
        tot as f32
    }
}

/// NEON f32×f32 dot in the ggml_vec_dot_f16 order (4 contiguous groups
/// per 16-step, fma chains, (s0+s2)+(s1+s3) lane-wise, faddp-style
/// pairwise reduce). For the head: operands are PRE-WIDENED f32 (the
/// widening equals the hardware f16->f32 convert incl. subnormals), so
/// this is bit-exact vs dot_f16_sve on the widened pair.
#[cfg(target_arch = "aarch64")]
pub fn dot_f32x16_neon(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len();
    debug_assert_eq!(n, y.len());
    let np = n & !15;
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i < np {
            let a0 = vld1q_f32(x.as_ptr().add(i));
            let b0 = vld1q_f32(y.as_ptr().add(i));
            s0 = vfmaq_f32(s0, a0, b0);
            let a1 = vld1q_f32(x.as_ptr().add(i + 4));
            let b1 = vld1q_f32(y.as_ptr().add(i + 4));
            s1 = vfmaq_f32(s1, a1, b1);
            let a2 = vld1q_f32(x.as_ptr().add(i + 8));
            let b2 = vld1q_f32(y.as_ptr().add(i + 8));
            s2 = vfmaq_f32(s2, a2, b2);
            let a3 = vld1q_f32(x.as_ptr().add(i + 12));
            let b3 = vld1q_f32(y.as_ptr().add(i + 12));
            s3 = vfmaq_f32(s3, a3, b3);
            i += 16;
        }
        let c = vaddq_f32(vaddq_f32(s0, s2), vaddq_f32(s1, s3));
        let mut tot = vaddvq_f32(c) as f64;
        for k in np..n {
            tot += (x[k] * y[k]) as f64;
        }
        tot as f32
    }
}

pub fn dot_f16_sve(x: &[u16], y: &[u16]) -> f32 {
    // NEON form: widen BOTH operands (exact incl. subnormals — the
    // verified widen_f16x4) and fma in the identical order: 4
    // contiguous groups per 16-step, (s0+s2)+(s1+s3) lane-wise,
    // faddp-style pairwise reduce, f64 leftovers. f16×f16 products
    // are exact in f32, so this is bit-identical to the scalar form.
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = x.len();
        debug_assert_eq!(n, y.len());
        let np = n & !15;
        unsafe {
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = vdupq_n_f32(0.0);
            let mut s2 = vdupq_n_f32(0.0);
            let mut s3 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i < np {
                let a0 = widen_f16x4(vld1_u16(x.as_ptr().add(i)));
                let b0 = widen_f16x4(vld1_u16(y.as_ptr().add(i)));
                s0 = vfmaq_f32(s0, a0, b0);
                let a1 = widen_f16x4(vld1_u16(x.as_ptr().add(i + 4)));
                let b1 = widen_f16x4(vld1_u16(y.as_ptr().add(i + 4)));
                s1 = vfmaq_f32(s1, a1, b1);
                let a2 = widen_f16x4(vld1_u16(x.as_ptr().add(i + 8)));
                let b2 = widen_f16x4(vld1_u16(y.as_ptr().add(i + 8)));
                s2 = vfmaq_f32(s2, a2, b2);
                let a3 = widen_f16x4(vld1_u16(x.as_ptr().add(i + 12)));
                let b3 = widen_f16x4(vld1_u16(y.as_ptr().add(i + 12)));
                s3 = vfmaq_f32(s3, a3, b3);
                i += 16;
            }
            let c = vaddq_f32(vaddq_f32(s0, s2), vaddq_f32(s1, s3));
            let mut tot = vaddvq_f32(c) as f64;
            for k in np..n {
                tot += (half_to_f32(x[k]) * half_to_f32(y[k])) as f64;
            }
            return tot as f32;
        }
    }
    // scalar form (the ACTUAL compiled ggml_vec_dot_f16 order)
    // the ACTUAL compiled ggml_vec_dot_f16 on this machine: plain NEON
    // (STEP=16, EPR=4), NOT the SVE branch — four f32x4 accumulators
    // take contiguous 4-element groups per 16-step; reduce is
    // (s0+s2)+(s1+s3) lane-wise then vaddvq pairwise; leftovers in
    // f64. Fitted bit-exact against dlsym'd libggml-cpu (verified on
    // real kq values, j=0 and j=1 samples).
    let n = x.len();
    debug_assert_eq!(n, y.len());
    let np = n & !15;
    let mut s = [[0.0f32; 4]; 4];
    let mut i = 0;
    while i < np {
        for j in 0..4 {
            for lane in 0..4 {
                let idx = i + 4 * j + lane;
                let a = half_to_f32(x[idx]);
                let b = half_to_f32(y[idx]);
                s[j][lane] = a.mul_add(b, s[j][lane]);
            }
        }
        i += 16;
    }
    let mut c = [0.0f32; 4];
    for l in 0..4 {
        c[l] = (s[0][l] + s[2][l]) + (s[1][l] + s[3][l]);
    }
    let r = (c[0] + c[1]) + (c[2] + c[3]);
    let mut tot = r as f64;
    for k in np..n {
        tot += (half_to_f32(x[k]) * half_to_f32(y[k])) as f64;
    }
    tot as f32
}

/// REJECTED historical fit (kept only as a comparison variant in
/// examples/headtest): an even/odd SVE lane mapping was fitted on
/// crafted inputs but the compiled ggml_vec_dot_f16 on this machine
/// is the plain-NEON kernel (see dot_f16_sve) — llamafile is compiled
/// out and no SVE vec_dot variant runs for these shapes. Do not use.
pub fn vec_dot_f16(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    let mut s = [[[0.0f32; 4]; 2]; 4];
    let np = n - n % 64;
    let mut i0 = 0;
    while i0 < np {
        for c in 0..8 {
            let acc = c % 4;
            let base = i0 + c * 8;
            for j in 0..8 {
                let (lh, lane) = if j < 4 { (0, j) } else { (1, j - 4) };
                let cell = &mut s[acc][lh][lane];
                *cell = x[base + j].mul_add(y[base + j], *cell);
            }
        }
        i0 += 64;
    }
    // np..np2: 8-wide chunks into sum1
    let np2 = n - n % 8;
    while i0 < np2 {
        for j in 0..8 {
            let (lh, lane) = if j < 4 { (0, j) } else { (1, j - 4) };
            let cell = &mut s[0][lh][lane];
            *cell = x[i0 + j].mul_add(y[i0 + j], *cell);
        }
        i0 += 8;
    }
    // masked tail: elements 0..rem into sum1 lanes (0..3 lo, 4..7 hi)
    let rem = n - np2;
    for j in 0..rem {
        let (lh, lane) = if j < 4 { (0, j) } else { (1, j - 4) };
        let cell = &mut s[0][lh][lane];
        *cell = x[i0 + j].mul_add(y[i0 + j], *cell);
    }
    let mut lo = [0.0f32; 4];
    let mut hi = [0.0f32; 4];
    for j in 0..4 {
        lo[j] = (s[0][0][j] + s[1][0][j]) + (s[2][0][j] + s[3][0][j]);
        hi[j] = (s[0][1][j] + s[1][1][j]) + (s[2][1][j] + s[3][1][j]);
    }
    let red = |v: &[f32; 4]| (v[0] + v[1]) + (v[2] + v[3]);
    ((red(&lo) as f64) + (red(&hi) as f64)) as f32
}

/// ggml_vec_dot_f32 on this machine (SVE codepath, runtime VL=128-bit
/// -> epr=4, step=32): eight 4-lane accumulators taking contiguous
/// 4-element chunks per 32-step; reduce (s1+s2)(s3+s4)(s5+s6)(s7+s8)
/// -> (s1+s3), (s5+s7) -> s1 += s5, all lane-wise FMA; final
/// svaddv horizontal into f64 then one narrowing to f32.
/// Used for the KQ scores: ggml_mul_mat_set_prec(kq, GGML_PREC_F32)
/// forces the f32 dot — Q stays full f32, K widened from f16.
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    const EPR: usize = 4;
    const STEP: usize = 32;
    let mut s = [[0.0f32; 4]; 8];
    let np = n - n % STEP;
    let mut i = 0;
    while i < np {
        for c in 0..8 {
            let base = i + c * EPR;
            for lane in 0..EPR {
                let cell = &mut s[c][lane];
                *cell = x[base + lane].mul_add(y[base + lane], *cell);
            }
        }
        i += STEP;
    }
    let np2 = n - n % EPR;
    while i < np2 {
        for lane in 0..EPR {
            let cell = &mut s[0][lane];
            *cell = x[i + lane].mul_add(y[i + lane], *cell);
        }
        i += EPR;
    }
    let rem = n - np2;
    for j in 0..rem {
        let cell = &mut s[0][j];
        *cell = x[i + j].mul_add(y[i + j], *cell);
    }
    // lane-wise tree
    let mut t = [0.0f32; 4];
    for l in 0..4 {
        t[l] = ((s[0][l] + s[1][l]) + (s[2][l] + s[3][l]))
            + ((s[4][l] + s[5][l]) + (s[6][l] + s[7][l]));
    }
    // svaddv horizontal (order fitted empirically: pairwise)

    (t[0] + t[1]) + (t[2] + t[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_roundtrip_known_values() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert_eq!(half_to_f32(0x4000), 2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x8000), -0.0);
        assert_eq!(half_to_f32(0x3555), 0.33325195f32); // ~1/3 in half
    }

    #[test]
    fn tq1_0_encode_roundtrip() {
        // random trits + scales -> encode -> decode -> identical packed form
        let mut st = 0x1234_5678_9abc_def1u64;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for trial in 0..50 {
            let rows = 1 + (rnd() % 7) as usize;
            let n_blocks = 1 + (rnd() % 4) as usize;
            let cols = n_blocks * QK_K;
            let n = rows * n_blocks * QK_K;
            let trits: Vec<i8> = (0..n).map(|_| (rnd() % 3) as i8 - 1).collect();
            // the format stores f16 scales: pre-round like a trainer
            // exporting after quantization would
            let d: Vec<f32> = (0..rows * n_blocks)
                .map(|i| {
                    let v = (((rnd() % 2000) as f32) + 1.0)
                        * if i % 2 == 0 { 1.0 } else { -1.0 }
                        * 1e-3;
                    half_to_f32(f32_to_f16(v))
                })
                .collect();
            let t = TqTensor {
                rows,
                cols,
                n_blocks,
                packed: pack_trits(&trits),
                d,
            };
            let raw = encode_tq1_0(&t);
            let back = decode_tq1_0(&raw, rows, cols).unwrap();
            assert_eq!(back.packed, t.packed, "trial {trial}: packed mismatch");
            assert_eq!(back.d, t.d, "trial {trial}: scales mismatch");
        }
    }

    #[test]
    fn trit_decode_symmetric() {
        // all-zero bytes decode to -1? no: byte 0 -> trit 0 -> -1.
        // encode check: trit t stored via ceil-div base3; byte=0 => t=0
        let raw = vec![0u8; 54];
        let t = decode_tq1_0(&raw, 1, 256).unwrap();
        assert!((0..256).all(|j| trit_at(&t.packed, j) == -1));
        // byte 128: t = (128*3)>>8 = 1 -> trit 0
        let raw = vec![128u8; 54];
        let t = decode_tq1_0(&raw, 1, 256).unwrap();
        assert!(
            (0..256).all(|j| trit_at(&t.packed, j) == 0),
            "expected all-zero trits"
        );
        // byte 255: t = (255*3)>>8 = 2 -> trit +1
        let raw = vec![255u8; 54];
        let t = decode_tq1_0(&raw, 1, 256).unwrap();
        assert!(
            (0..256).all(|j| trit_at(&t.packed, j) == 1),
            "expected all-ones trits"
        );
    }
}
