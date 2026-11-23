//! tick-kernels — the ternary compute core: `Tri`/`TernaryMat` (the
//! multiply-free 2-bit vocabulary), `bitlinear` (tq1_0 packing, q8_K
//! activation quantization, NEON/SVE dot kernels, f16 helpers), and
//! `par` (lock-free fork-join pool).

// The kernels and conversions mirror a frozen, sha-pinned reference
// implementation bit-for-bit: loop shapes, constants, and intermediate
// writes stay close to that reference on purpose, so idiom lints that
// would push rewrites are disabled crate-wide. In particular, scalar
// fallbacks sit behind arch-cfg! early returns and read as unreachable
// on architectures where the vector path is compiled.
#![allow(unreachable_code, unused_assignments)]
#![allow(clippy::needless_range_loop, clippy::should_implement_trait)]

pub mod bitlinear;
pub mod par;

/// One ternary weight: −1, 0, or +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tri {
    Neg,
    Zero,
    Pos,
}

impl Tri {
    /// Decode a 2-bit code (weight layouts pack two bits per weight).
    pub fn from_bits2(b: u8) -> Tri {
        match b & 0b11 {
            0 => Tri::Zero,
            1 => Tri::Pos,
            2 => Tri::Neg,
            _ => Tri::Zero, // 0b11 unused; treated as zero by the readers
        }
    }
    /// The accumulator update for one activation: add, subtract, skip.
    /// This IS the engine's multiply-free inner loop.
    #[inline]
    pub fn apply(self, x: i8, acc: i32) -> i32 {
        match self {
            Tri::Neg => acc - x as i32,
            Tri::Zero => acc,
            Tri::Pos => acc + x as i32,
        }
    }
}

/// Packed ternary matrix: `rows × cols` weights at two bits each,
/// row-major nibble order (little-endian pairs). Allocation happens at
/// load; steady state never touches the heap.
pub struct TernaryMat {
    rows: usize,
    cols: usize,
    packed: Vec<u8>, // ceil(cols/4) bytes per row... see row_bytes
}

impl TernaryMat {
    pub fn zeros(rows: usize, cols: usize) -> TernaryMat {
        TernaryMat {
            rows,
            cols,
            packed: vec![0u8; rows * Self::row_bytes(cols)],
        }
    }

    /// 2 bits per weight, 4 weights per byte.
    pub fn row_bytes(cols: usize) -> usize {
        cols.div_ceil(4)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> Tri {
        let byte = self.packed[row * Self::row_bytes(self.cols) + col / 4];
        Tri::from_bits2(byte >> ((col % 4) * 2))
    }

    pub fn set(&mut self, row: usize, col: usize, w: Tri) {
        let code = match w {
            Tri::Zero => 0u8,
            Tri::Pos => 1,
            Tri::Neg => 2,
        };
        let rb = Self::row_bytes(self.cols);
        let byte = &mut self.packed[row * rb + col / 4];
        let shift = (col % 4) * 2;
        *byte = (*byte & !(0b11 << shift)) | (code << shift);
    }

    /// The tick engine's one matvec: for each row, add/subtract/skip
    /// per weight — no multiplies. Accumulation order is row-major,
    /// left to right, fixed: it is part of the parity contract.
    pub fn matvec_i8(&self, x: &[i8]) -> Vec<i32> {
        assert_eq!(x.len(), self.cols, "activation vector width mismatch");
        let rb = Self::row_bytes(self.cols);
        let mut out = vec![0i32; self.rows];
        for (r, o) in out.iter_mut().enumerate() {
            let row = &self.packed[r * rb..(r + 1) * rb];
            let mut acc = 0i32;
            let mut c = 0usize;
            for &byte in row {
                for shift in [0, 2, 4, 6] {
                    if c >= self.cols {
                        break;
                    }
                    acc = Tri::from_bits2(byte >> shift).apply(x[c], acc);
                    c += 1;
                }
            }
            *o = acc;
        }
        out
    }
}

/// The tick counter — per-tick telemetry, the spark-engine discipline
/// at tick granularity. `tick` is the only mutation point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickTelemetry {
    pub tick: u64,
    pub layer: u32,
    pub cycles_hint: u32,
}

impl TickTelemetry {
    pub fn new() -> TickTelemetry {
        TickTelemetry {
            tick: 0,
            layer: 0,
            cycles_hint: 0,
        }
    }

    /// Advance one layer tick; returns the new state.
    pub fn tick(&mut self, layer: u32) -> TickTelemetry {
        self.tick += 1;
        self.layer = layer;
        *self
    }
}

impl Default for TickTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

pub use bitlinear::TqTensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_matvec_matches_scalar_reference() {
        // deterministic pseudo-random weights + activations; the
        // packed kernel must equal the naive Tri::apply fold
        let (rows, cols) = (37usize, 101usize);
        let mut m = TernaryMat::zeros(rows, cols);
        let mut x = vec![0i8; cols];
        let mut seed = 0x12345678u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };
        for r in 0..rows {
            for c in 0..cols {
                m.set(
                    r,
                    c,
                    match rnd() % 3 {
                        0 => Tri::Neg,
                        1 => Tri::Zero,
                        _ => Tri::Pos,
                    },
                );
            }
        }
        for v in &mut x {
            *v = ((rnd() % 251) as i32 - 125) as i8;
        }
        let packed = m.matvec_i8(&x);
        for r in 0..rows {
            let naive = (0..cols).fold(0i32, |acc, c| m.get(r, c).apply(x[c], acc));
            assert_eq!(packed[r], naive, "row {r}");
        }
    }

    #[test]
    fn roundtrip_packing() {
        let mut m = TernaryMat::zeros(3, 9); // odd width exercises tail bytes
        let ws = [
            Tri::Pos,
            Tri::Neg,
            Tri::Zero,
            Tri::Neg,
            Tri::Pos,
            Tri::Pos,
            Tri::Pos,
            Tri::Zero,
            Tri::Neg,
        ];
        for (c, w) in ws.iter().enumerate() {
            m.set(1, c, *w);
        }
        for (c, w) in ws.iter().enumerate() {
            assert_eq!(m.get(1, c), *w, "col {c}");
        }
    }

    #[test]
    fn telemetry_ticks_monotonically() {
        let mut t = TickTelemetry::new();
        assert_eq!(t.tick, 0);
        t.tick(3);
        t.tick(4);
        assert_eq!((t.tick, t.layer), (2, 4));
    }
}
