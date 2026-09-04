# RX 9070 XT Setup

The RX 9070 XT is an RDNA 4 device with LLVM target `gfx1201`. AMD's current
compatibility pages list it for the Windows HIP SDK and for ROCm under WSL.
Always recheck the matrix because the supported driver, ROCm, and Ubuntu versions
move together.

Authoritative links:

- [ROCm installation selector](https://rocm.docs.amd.com/en/latest/install/rocm.html)
- [Windows system requirements](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/shared/hipsdk/reference/system-requirements.html)
- [ROCm on WSL installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)
- [ROCprofiler SDK quick reference](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/quick_guide.html)

## Recommended development route

Use Ubuntu 24.04 or 22.04 under WSL with the driver/ROCm pair named by AMD's
current support matrix. This provides the Linux CMake HIP language and ROCm CLI
workflow used by the project. Native Windows remains useful for HIP SDK samples
and Radeon GPU Profiler, but its component set differs from Linux.

After following AMD's installation guide, verify the environment before building:

```bash
rocminfo | grep -A 2 gfx1201
hipconfig --full
cmake --version
```

Expected essentials:

- The GPU appears as `gfx1201`.
- HIP reports the AMD platform and a valid ROCm path.
- CMake is version 3.25 or newer.

Install ordinary build tools using the packages for the selected Ubuntu release:

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build python3
```

Then use the HIP build command from the main README.

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

