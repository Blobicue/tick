# tick-kernels

The ternary compute core: `Tri`/`TernaryMat` (the multiply-free 2-bit
vocabulary), tq1_0 packing with the decode-built encode table, q8_K
activation quantization, NEON/SVE dot kernels, exact f16↔f32 helpers,
and a lock-free fork-join pool.

No dependencies beyond libc. Kernels carry documented lint allowances
where idiom rewrites would churn bit-exactness-verified code.
