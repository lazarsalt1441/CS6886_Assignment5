# CS6886 Assignment 5: native Apple M1 submission

The report, CSV and plots use the native MacBook Air run collected on 6 November
2026. All four official inputs and the exact C/header sources match the returned
SHA-256 hashes. Every original starter driver and naive source remain unchanged.

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

This also requires pdflatex. Report regeneration checks all source/input hashes,
raw timing medians and GFLOP/s calculations before rebuilding. For another native
run, use `python3 src/finalize_mac.py --data PATH_TO_RESULT_FOLDER`.

Q1 uses observed hardware/cache details plus clearly stated reference P/E clock
and instruction-throughput values for theoretical peak. Actual run-time CPU
clocks/core placement are unmeasured; no exact sustained frequency is invented.
Arithmetic intensity is an analytical DRAM traffic model, not hardware-counter
measurement. The independent FMA test is achieved throughput, not a hard ceiling.

