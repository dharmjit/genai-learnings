#!/usr/bin/env bash
# Make a runtime-only NeevCloud GPU pod able to compile CUDA.
#
# The obvious command does NOT work on these images:
#   apt-get install cuda-toolkit-12-9
#     -> cuda-libraries-12-9 : Depends: libcublas-12-9 (>= 12.9.2.10)
#        but 12.9.1.4-1 is to be installed
#        E: Unable to correct problems, you have held broken packages.
#
# The image `apt-mark hold`s libcublas-12-9 and libnccl2, and the repo's
# current -dev packages demand newer runtimes. So install the compiler pieces
# directly and pin each -dev package to the held runtime version.
set -euo pipefail

CUDA_MAJOR_MINOR=12-9
CUDA_DIR=/usr/local/cuda-12.9

pinned() { dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true; }

CUBLAS_V="$(pinned libcublas-${CUDA_MAJOR_MINOR})"
NCCL_V="$(pinned libnccl2)"

echo "held libcublas: ${CUBLAS_V:-<none>}"
echo "held libnccl2 : ${NCCL_V:-<none>}"

apt-get update -qq

PKGS=(cuda-nvcc-${CUDA_MAJOR_MINOR} cuda-cudart-dev-${CUDA_MAJOR_MINOR}
      cuda-nvtx-${CUDA_MAJOR_MINOR})
[ -n "$CUBLAS_V" ] && PKGS+=("libcublas-dev-${CUDA_MAJOR_MINOR}=${CUBLAS_V}") \
                   || PKGS+=("libcublas-dev-${CUDA_MAJOR_MINOR}")
[ -n "$NCCL_V" ]   && PKGS+=("libnccl-dev=${NCCL_V}")

DEBIAN_FRONTEND=noninteractive apt-get install -y "${PKGS[@]}"

# CUDA is not on PATH by default on these images.
cat > /etc/profile.d/cuda.sh <<EOF
export PATH=${CUDA_DIR}/bin:\$PATH
export LD_LIBRARY_PATH=${CUDA_DIR}/lib64:\${LD_LIBRARY_PATH:-}
EOF
chmod +x /etc/profile.d/cuda.sh

export PATH=${CUDA_DIR}/bin:$PATH
nvcc --version | tail -2
echo
echo "OK. New shells get nvcc via /etc/profile.d/cuda.sh;"
echo "in THIS shell run: export PATH=${CUDA_DIR}/bin:\$PATH"

# Warn about the other thing that bites people on these pods.
SHM=$(df -m /dev/shm | awk 'NR==2{print $2}')
if [ "$SHM" -lt 1024 ]; then
  echo
  echo "WARNING: /dev/shm is only ${SHM} MiB (Docker default)."
  echo "  PyTorch DataLoader(num_workers>0) and NCCL's shm transport will"
  echo "  fail or degrade. Needs a larger --shm-size at pod creation time;"
  echo "  it cannot be fixed from inside the container."
fi
