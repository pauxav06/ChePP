# ChePP

## Build Instructions

ChePP requires a fairly modern compiler with **C++20 support**.

Tested with:

- **GCC** 11–16
- **Clang** 17–18
- **Apple Clang** 17
- **MSVC** 17 (Visual Studio 2022) / 18 (Visual Studio 2026)

Compatibility options are provided where possible.

---

## Native build
```bash
cmake -S . -B build [-G <generator>] [-A <platform> (MSVC)] [CMAKE_OPTIONS...]
cmake --build build [--config <config>] [--target <target>]
```
### Example - Linux/MacOS

Compile and run tests using **Posix Makefiles**:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DARCH=native
cmake --build build
ctest --test-dir build
```
### Example - Windows
Compile and run tests using **MSVC**:
```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DARCH=AVX2 # no native on MSVC
cmake --build build --config Release
ctest --test-dir build -C Release
```

## Cross-arch build
### Example - Windows x64 portable build
Compile using **MSVC** for a **SSE2** baseline on a Windows x64 computer.\
The Core engine will run slightly slower than compiling for a more specific target.\
However the NNUE kernels are **dynamically dispatched** and will choose the best target at runtime.\
Running the engine compiled this way on a target using AVX512 will use AVX512 for the NNUE.

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DARCH=SSE2
cmake --build build --config Release
```
## Cross-platform build

Cross-platform compilation requires a **two-phase build**:

1. Build **host tools**
2. Build the **target binary**

```bash
cmake -S . -B <build-host-dir> [-G <generator>]
[-A <native-platform> (MSVC)] [CMAKE_OPTIONS...]

cmake --install <build-host-dir> --target DumpBin Bin2Cpp --config Release

cmake -S . -B <build-dir> [-G <generator>] \
-DCMAKE_TOOLCHAIN_FILE=<toolchain> \
-DCMAKE_PREFIX_PATH=<build-host-dir> \
[-A <target-platform> (MSVC)] [CMAKE_OPTIONS...]

cmake --build <build-dir> --target ChePP [--config <config>]
```

### Example - Windows x64 Haswell build from Linux using `mingw-w64`

```bash
cmake -S . -B build-host -G "Ninja Multi-Config"

cmake --build build-host --target dump_magics bin2cpp --config Release

cmake -S . -B build -G "Ninja Multi-Config" \
-DCMAKE_TOOLCHAIN_FILE=mingw-w64-toolchain.cmake \
-DCMAKE_PREFIX_PATH=$(pwd)/build-host \
-DARCH=haswell \
-DSTATIC=True

cmake --build build --target ChePP --config Release
```


---
## CMake Options

- `-DNO_LTO=TRUE`Disable **Link Time Optimization (LTO)**.
- `-DSANITIZE=TRUE` Enable **sanitizers** in **Debug builds**.
- `-DARCH=<arch>` Set the **target architecture** used by the compiler.\
  Enables optimizations such as `POPCOUNT`, `PEXT`, and improved vectorization.\
  NNUE kernels are **dynamically dispatched** and are **not affected by this flag**.

  | Compiler                                  | Architectures                                 |
  |-------------------------------------------|-----------------------------------------------|
  | **MSVC / Clang-cl**                       | `SSE2`, `AVX2`, `AVX512`, etc                 |
  | **GCC / Clang / Apple Clang / Mingw-GCC** | `native`, `x86-64`, `skylake`, `armv8-a`, etc |

- **`-DSTATIC=TRUE`** Enable **static linking** of libc and libc++ (useful for cross-compilation).
---

# Usage

ChePP supports all major **UCI commands** and can be used with a **UCI-compatible chess interface** (GUI, Python-Lichess, etc.).

**_Example_ session**:
```bash
position startpos moves e2e4
go movetime 1000
setoption name Tune
go depth 20
stop

# non UCI commands:
print
print-nnue
eval

quit
```

---

# Overview

**ChePP** is a **portable UCI-compliant chess engine** written in **C++**.

The engine combines a **Negamax search** with an **Efficiently Updatable Neural Network (NNUE)** for position evaluation.

Current estimated strength on Lichess-Blitz ladder: **~2650**.  
For comparison, **Stockfish** on similar hardware https://lichess.org/@/ProteusSF-lite achieves around **3000**.

A precise **CCRL Elo estimate** is being measured, but takes a lot of resources :(

---

# Features

- **Bitboard-based move generation**
    - Magic bitboards / `PEXT`

- **NNUE evaluation**
    - Trained on ~300 GB of chess positions
    - SIMD kernels
    - Runtime kernel selection via the `Tune` option

- **Negamax search**
    - Iterative deepening
    - Move ordering (history, killer, continuation...)
    - Quiescence search
    - Aspiration windows
    - Transposition tables
    - Null-move pruning
    - Late move reductions
    - Additional standard search optimizations

---

# Future Improvements

- Implement missing UCI commands (e.g. `ponder`)
- Improve time management and responsiveness of `go`
- Add NNUE accumulator caches
- Implement sparse affine evaluation
- More pruning == more elo :)
---

# Acknowledgements

- **Google Highway** - portable SIMD abstraction
- **Fathom** - tablebase probing (TBProbe)
- **Grapheus** - C++ CUDA NNUE trainer by Luecx
- **Stockfish** - reference implementation and amazing documentation of the NNUE 
- **Rice** - for the clean and understandable code