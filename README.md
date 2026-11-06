# CS6886 Assignment 5: native Apple M1 submission

The plots, CSV and native evidence now use the verified M1 run of the
current online-attention code: 40 timing configurations and 168 FP64 checks.
At N=4096, d=64, best takes 58.175 ms (29.82x naive); eight threads take
13.578 ms (127.75x naive). The largest measured speedups are 35.22x serial
and 149.50x with eight threads, at N=4096, d=128.
The measured best/omp setting is 48 query rows, 64 keys and four register rows;
the Makefile pins these flags for reproduction. Fused uses 48x128 tiles.
All original drivers and official inputs are unchanged.

## Deliverables

- `report.pdf`: Q1-Q5, calculations, measurements, explanations and references.
- `machine_spec.txt`: observed machine properties and labelled reference models.
- `results.csv`: all 40 input/configuration rows, useful GFLOP/s, errors, speedups,
  parallel efficiency and reference peak percentages.
- `plots/roofline.png`, `plots/time_N<N>_d<d>.png`: required plots.
- `src/`, `input/`, `Makefile`, `assignment_handout.pdf`: implementation and inputs.
- `src/evidence/mac_run/`: native raw timings, compiler commands, checks, hardware
  observations, bandwidth/FMA tests and input/source hashes.

## Run and validate on the M1

```sh
brew install llvm libomp python
bash src/run_macos.sh --tune --compare-legacy --profile
```

This creates `Mac_Results.zip`: all standard variant/thread measurements,
168 independent FP64 checks, ten candidate tile configurations, paired previous
versus new measurements, and separate instrumented phase profiles. The submitted
results are already from this run; a new run may vary with scheduling and temperature. See `src/OPTIMIZATION_NOTES.md` for the algorithm and tradeoffs.

For the simpler build/check workflow:

```sh
make all
make check
make edge-check
```

The full measurement controller is `bash src/run_macos.sh`. It runs three
independent driver invocations per configuration, each with one warm-up and the
median of five timed calls, and creates `Mac_Results.zip`. See
`src/START_HERE_MAC.md` for run details. Standard-library Python validation uses
no external packages, and the C kernels link only the standard math/OpenMP
runtimes, with no BLAS or external vector-math library.

## Rebuild the supplied report and plots

```sh
python3 -m pip install matplotlib numpy
make report
```

This requires pdflatex. Report regeneration checks all source/input hashes,
raw timing medians and GFLOP/s calculations before rebuilding. For another native
run, use `python3 src/finalize_mac.py --data PATH_TO_RESULT_FOLDER`.

Q1 uses observed hardware/cache details plus clearly stated reference P/E clock
and instruction-throughput values for theoretical peak. Actual run-time CPU
clocks/core placement are unmeasured; no exact sustained frequency is invented.
Arithmetic intensity is an analytical DRAM traffic model, not hardware-counter
measurement. The independent FMA test is achieved throughput, not a hard ceiling.
