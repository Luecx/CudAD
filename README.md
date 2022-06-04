# CudAD

A cuda enabled nueral network trainer for the [Koivisto Chess Engine](https://github.com/Luecx/Koivisto).

## Requirements

- [CMake](https://cmake.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Visual Studio](https://visualstudio.microsoft.com/downloads/)

## Installation

*`cl` (from Visual Studio) must be included in your path!*

### Clone

```bash
git clone https://github.com/Luecx/CudAD
cd CudAD
```

### Build

```bash
cmake -B cmake-build-release -S .
cmake --build cmake-build-release --config Release --target CudAD -j4
```
### Run

```bash
./cmake-build-release/Release/CudAD
```

## Maintaining

### Formatting

This project uses `clang-format` to keep it's code formatted. The style can be found in [.clang-format](.clang-format).

To format a file, with `clang-format` in your path.
```bash
clang-format -i <file_path>
```

Below is a helpful script to format the codebase in one go.
```bash
find src/ -iname *.h -o -iname *.cu | xargs clang-format -i
```
