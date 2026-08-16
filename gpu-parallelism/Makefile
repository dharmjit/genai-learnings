CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= 120

NVCCFLAGS := -O3 -std=c++17 -arch=sm_$(ARCH) -lineinfo -Icommon \
             --expt-relaxed-constexpr -Xcompiler -fopenmp,-march=native
LDLIBS    := -lcublas

SRCS  := $(wildcard [0-9]*/*.cu)
NAMES := $(basename $(notdir $(SRCS)))
BINS  := $(addprefix bin/,$(NAMES))

vpath %.cu $(sort $(dir $(SRCS)))

.PHONY: all clean run
all: $(BINS)

bin:
	@mkdir -p bin

bin/%: %.cu common/helpers.cuh common/collectives.cuh | bin
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -rf bin

# Ordered tour. Each stage assumes you read the output of the previous one.
run: all
	@for b in device_probe transpose matmul p2p_bandwidth overlap_streams \
	          dp_matmul tp_matmul pp_matmul alltoall_transpose mlp_2gpu; do \
	  echo ""; echo "=============== $$b ==============="; \
	  ./bin/$$b || echo "($$b failed)"; \
	done
