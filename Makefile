# Apple M1 / macOS edition. Drivers and official inputs are unchanged.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LLVM_PREFIX := $(shell brew --prefix llvm)
OMP_PREFIX := $(shell brew --prefix libomp)
CC = $(LLVM_PREFIX)/bin/clang
CFLAGS = -O3 -mcpu=apple-m1 -fopenmp-simd -Wall -Wextra -Wno-unknown-pragmas
CPPFLAGS = -I$(OMP_PREFIX)/include
LDFLAGS = -L$(OMP_PREFIX)/lib -Wl,-rpath,$(OMP_PREFIX)/lib
LDLIBS = -lomp -lm
else
CC = gcc
CFLAGS = -O3 -march=native -fopenmp-simd -Wall -Wextra -Wno-unknown-pragmas
CPPFLAGS =
LDFLAGS =
LDLIBS = -lm
endif
OMP = -fopenmp
INPUTS = $(sort $(wildcard input/*.bin))
VARIANTS = naive loops tiled fused best omp
OMP_NUM_THREADS ?= 8
export OMP_NUM_THREADS

all: $(VARIANTS)
build output:
	mkdir -p $@
build/omp: src/attn_omp.c src/best_kernel.h src/neon_kernel.h | build output
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMP) $< -o $@ $(LDFLAGS) $(LDLIBS)
build/%: src/attn_%.c src/best_kernel.h src/neon_kernel.h | build output
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
naive loops tiled fused best omp: %: build/%
	@set -e; for f in $(INPUTS); do ./build/$@ "$$f" "output/$@_$$(basename "$$f")"; done
compile: $(addprefix build/,$(VARIANTS))
check:
	python3 src/check_outputs.py
edge-check: compile
	python3 src/edge_check.py
macos-run:
	bash src/run_macos.sh
clean:
	rm -rf build
.PHONY: all naive loops tiled fused best omp compile check edge-check macos-run clean

report:
	python3 src/finalize_mac.py
plots:
	python3 src/finalize_mac.py --plots-only
.PHONY: report plots
