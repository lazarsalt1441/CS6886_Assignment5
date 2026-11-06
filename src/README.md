# CS6886 Assignment 5: native Apple M1 submission

The current code adds blocked online attention and optional native tile tuning.
The archived CSV/plots and `src/evidence/mac_run/` describe the earlier M1 kernel;
they are not measurements of this revision. `report.pdf` is the replacement PDF
uploaded by the repository owner. Run the command below to measure the new code.
The four official inputs, original naive implementation and all starter drivers
are unchanged.

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
versus new measurements, and separate instrumented phase profiles. Upload the ZIP
to update the report with measured M1 results. No 50x or peak-utilisation claim is
assumed in advance. See `src/OPTIMIZATION_NOTES.md` for the algorithm and tradeoffs.

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
python3 src/finalize_mac.py --data output/mac_run_YYYYMMDD_HHMMSS
```

Replace the timestamp with the fresh run directory; this also requires pdflatex.
Report regeneration checks all source/input hashes,
raw timing medians and GFLOP/s calculations before rebuilding. For another native
run, use `python3 src/finalize_mac.py --data PATH_TO_RESULT_FOLDER`.

Q1 uses observed hardware/cache details plus clearly stated reference P/E clock
and instruction-throughput values for theoretical peak. Actual run-time CPU
clocks/core placement are unmeasured; no exact sustained frequency is invented.
Arithmetic intensity is an analytical DRAM traffic model, not hardware-counter
measurement. The independent FMA test is achieved throughput, not a hard ceiling.

Local assignment-root paths in the archived logs are normalised to
`<assignment_root>` for portability. Original measurements/hashes are retained as
historical evidence. They intentionally fail the source-hash guard for the new
kernel; use a fresh native run for report regeneration. Linux development data,
when supplied separately, must not be presented as M1 measurements.

Submission repository: https://github.com/lazarsalt1441/CS6886_Assignment5

The contents of this repository follow the assignment-root structure required
by the handout.
