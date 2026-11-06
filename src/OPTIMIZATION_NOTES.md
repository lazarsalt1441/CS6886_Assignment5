# Online attention update

The baseline, all starter drivers, and the four official inputs are unchanged.
`report.pdf` is the PDF uploaded by the owner. The existing main CSV, plots and
native evidence describe the previous kernel; they are retained without relabelling.
New M1 performance numbers require a fresh native run.

## Implementation

The fused kernel works on 48 query rows and 128 keys. Its score scratch is 24 KiB
per worker, independent of sequence length. Keys are packed by key block and then
feature: the inner reduction stride is 128 rather than the full sequence length.
The extra O(Nd) key buffer and its allocation/packing are included in timed calls.
Key and score buffers are allocated with 128-byte alignment.
The whole working set also includes Q, K, V and output, so fitting the scores in
L1 alone does not establish that this is the fastest tile.

Each row keeps maximum m, denominator l and an unnormalised output numerator u.
For each score block s:

    m_new = max(m, max(s))
    alpha = exp(m - m_new)  (zero for the first block)
    p = exp(s - m_new)
    l = alpha*l + sum(p)
    u = alpha*u + p*V_block

Return u/l after the last block. Rescaling only the denominator would be wrong;
the previous numerator is rescaled inside the PV microkernel too. An unchanged
maximum uses alpha=1 without calling scalar expf.

Four-row register kernels reuse each key/value vector across queries. M1 NEON
has four FP32 lanes: two independent vector exponentials process eight scores in
one loop iteration. This is not a claim of an eight-lane M1 instruction. The
degree-seven range-reduced approximation is unchanged from the validated prior
NEON kernel; it is used only for nonpositive softmax arguments. Scalar tails and
maximum-change corrections use expf. No fast-math, BLAS or vector-math library is used.

Best/omp can also use six register rows, retaining 24 vector accumulators instead
of 16. More register reuse can compete with register pressure and cache costs;
the native tuner measures it instead of assuming it wins. Packing and query work
share one OpenMP parallel region and a barrier; query tiles are dynamically
scheduled with private scratch and disjoint output rows.

## Measure on the Mac

Run from a clean checkout of the current repository:

```sh
bash src/run_macos.sh --tune --compare-legacy --profile
```

The optional tuner tries ten query/key/register configurations on all four input
shapes, reverses candidate order in its second round, and selects one global
configuration by geometric mean of per-input medians. The single global choice
may trade performance between individual shapes. Candidates are checked against
tiled outputs; the final selection is checked against naive and FP64
references. Compiler flags and every sample are retained. This is a finite tile
search, not proof of global optimality.

`legacy_comparison.json` records fresh paired old/new measurements, with order
reversed on alternate trials. `phase_profiles.json` records separate instrumented
single-thread runs, discards the warm-up, and uses the median-total-time sample
for phase shares. Its matrix throughput is useful 2*N*N*d FLOPs per product.
Instrumentation has overhead; these values do not replace normal timing rows.

`Mac_Results.zip` contains the measurements and source/input hashes. The normal
40 timing rows remain compatible with the report/plot tools. The updated FP64
suite has 168 checks, including score-tile tails, later/earlier maximum changes,
constant values and zero output. Numerical validation remains at 1e-4.

Interpret speedup against a named baseline. A 50x improvement over naive is not
a 50x improvement over the previous optimised kernel. If matrix products really
occupy 84% of a run, removing all of a 13% softmax phase can improve total time by
at most about 1.15x. Key packing, register reuse and cache tuning therefore matter.
Recalculate peak percentages using the same machine/core/thread model; do not
transfer an x86 peak or development timing into the M1 report.
