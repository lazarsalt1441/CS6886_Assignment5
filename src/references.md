# Architecture references and assumptions

1. Apple, MacBook Air (M1, 2020) specifications:
   https://support.apple.com/en-us/111883
2. Dougall Johnson's primary instruction microbenchmarks:
   https://dougallj.github.io/applecpu/firestorm-simd.html
   https://dougallj.github.io/applecpu/icestorm-simd.html
   FMLA vector 4S reciprocal throughput is 0.25 cycles on P and 0.5 cycles on E.
   This is reverse-engineering evidence, not an Apple architectural guarantee.
3. Asahi/Linux T8103 device tree:
   https://github.com/torvalds/linux/blob/master/arch/arm64/boot/dts/apple/t8103.dtsi
   Reference maximum operating points: P turbo 3.204 GHz and E 2.064 GHz.
   The P turbo entry's Linux enablement does not establish macOS scheduling or
   sustained clock. The benchmark did not record actual clocks under load.
4. iFixit original M1 chip identification:
   https://www.ifixit.com/News/46884/m1-macbook-teardowns-something-old-something-new
   Identifies M1 LPDDR4X-4266 packages. For the reference bandwidth calculation,
   the report explicitly assumes a 128-bit aggregate interface. No physical
   subchannel count or exact memory-vendor identity is claimed from sysctl.
5. Apple's asymmetric-core scheduling guidance:
   https://developer.apple.com/documentation/apple-silicon/tuning-your-code-s-performance-for-apple-silicon/
6. Starter:
   https://github.com/NMJ0/Assignment-5-Systems-for-Deep-learning

All platform references were checked on 6 November 2026. Cache/core/RAM capacity
facts and all timing/correctness results come from the included native Mac run.
