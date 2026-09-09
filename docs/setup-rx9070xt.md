# RX 9070 XT Setup

The RX 9070 XT is an RDNA 4 device with LLVM target `gfx1201`. Native Linux is
the primary route; AMD also lists the card for the Windows HIP SDK and for ROCm
under WSL. Always recheck the matrix because the supported driver, ROCm, and
distribution versions move together.

`gfx1201` requires ROCm 6.4 or newer. Anything older will not produce a usable
code object for this card.

Authoritative links:

- [ROCm installation selector](https://rocm.docs.amd.com/en/latest/install/rocm.html)
- [Windows system requirements](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/shared/hipsdk/reference/system-requirements.html)
- [ROCm on WSL installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)
- [ROCprofiler SDK quick reference](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/quick_guide.html)

## Arch Linux (including Omarchy)

Arch ships ROCm in the `extra` repository, so no AMD installer or external
repository is needed. Confirm the kernel already enumerates the card first:

```bash
cat /sys/class/kfd/kfd/topology/nodes/*/properties | grep gfx_target_version
```

`120001` is `gfx1201`. If the file is missing, the `amdgpu` kernel driver is not
loaded and no userspace install will help.

Install the SDK, the query tools, and the profiler:

```bash
sudo pacman -Syu rocm-hip-sdk rocminfo rocm-smi-lib rocprofiler
```

Do not use `-Sy` without `-u`. Partial upgrades are unsupported on Arch and are a
common source of mismatched ROCm and LLVM libraries.

Arch installs into `/opt/rocm`, but `hipcc` is not on `PATH` by default:

```bash
export PATH=/opt/rocm/bin:$PATH
```

`/dev/kfd` is typically world-readable on Arch. If it is not, add yourself to the
`render` and `video` groups and start a new session:

```bash
ls -l /dev/kfd                       # expect crw-rw-rw- or a render-group mode
sudo usermod -aG render,video "$USER"
```

Export `ROCM_PATH` and `HIP_PATH` as well:

```bash
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
```

These are not optional on Arch. The package installs LLVM under
`/opt/rocm/lib/llvm`, which is not the layout the compiler driver probes for by
default, so without them `hipcc` fails with `fatal error: 'hip/hip_runtime.h'
file not found` even though the header is present at `/opt/rocm/include/hip/`.
`hipconfig -C` will show a telltale bare `-I/include` when they are unset. CMake
fails in a quieter way: it reports `The HIP compiler identification is unknown`,
configures anyway, and then breaks at the first HIP translation unit. With the
variables set it reports a real version instead.

## Ubuntu under WSL

Use Ubuntu 24.04 or 22.04 with the driver/ROCm pair named by AMD's current
support matrix, then follow AMD's installation guide. Native Windows remains
useful for HIP SDK samples and Radeon GPU Profiler, but its component set differs
from Linux. Install ordinary build tools from the selected release:

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build python3
```

## Verifying the environment

On either route, check the toolchain before building:

```bash
rocminfo | grep -A 2 gfx1201
hipconfig --full
cmake --version
```

Expected essentials:

- The GPU appears as `gfx1201`.
- HIP reports the AMD platform and a valid ROCm path.
- CMake is version 3.25 or newer.

Then use the HIP build command from the main README.

If the distribution's CMake is older than 3.25, or you would rather not touch the
system packages, a user-level CMake and Ninja are enough to configure and build
this project:

```bash
python3 -m venv .venv
.venv/bin/pip install cmake ninja
export PATH="$PWD/.venv/bin:$PATH"
```

## Memory planning for a 32 GB host

The GPU memory requirement matters more than system RAM for the kernels. The
materialized score buffer alone uses:

```text
batch * heads * sequence^2 * 4 bytes
```

For batch 1 and 8 heads, it is about 32 MiB at sequence 1024, 128 MiB at 2048,
and 512 MiB at 4096. Q, K, V, and output plus runtime allocations are additional.
Begin with sequence 256, verify correctness, and scale up while watching GPU
memory. The fused path does not allocate the score buffer.

## Profiling split

- Use HIP events in the executable for repeatable kernel latency.
- Use `rocprofv3 --runtime-trace` for dispatch and memory-operation traces where supported.
- Use `rocprofv3-avail` before selecting hardware counters.
- Use Radeon GPU Profiler on native Windows when low-level wavefront analysis is needed.

