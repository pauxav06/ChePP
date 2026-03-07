# ChePP

## Build Instructions

#### /!\ Only support native builds /!\

<details>
<summary>Linux/MacOS</summary>

Tested with GCC 13-14-15-16, Clang 17-18, Apple-clang 17
```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build
```

</details> 

<details> 
<summary>Windows</summary>

Tested with MSVC 17 2022 / 18 2026
```bash
cmake -S . -B build -G "Visual Studio 1x 20xx" -A x64
cmake --build build --config Release
ctest --test-dir build
```
</details>

### CMake Options

- `-DNO_LTO=TRUE`  
  Disables **Link Time Optimization (LTO)** during compilation.

- `-DSANITIZE=TRUE`  
  Enables **sanitizers** when building in **Debug mode**.

- `-DTARGET_ARCH=XXX`  
  Sets the **Windows CPU instruction set target**. Possible values:
    - `SSE2`
    - `AVX`
    - `AVX2`
    - `AVX512`

## Usage


Support all major **UCI** commands. Can be use with a UCI chess app (UI, python lichess...)

```bash
  position startpos moves e2e4
  go movetime 1000
  setoption name Tune // find best nnue config
  go depth 20
  quit
```

## Overview


**ChePP** is a performant, **UCI-compliant chess engine** written in C++.  
It combines a **negamax search** with an **Efficiently Updatable Neural Network (NNUE)** for evaluation.\
Lichess rating: **2650**. Comparativly, Stockfish running on similar hardware achieves a rating of 3000...\
I do not currently have resources to estimate ccrl elo :(

### Features

- **Bitboard-based move generation** using magics and `PEXT`.
- **NNUE evaluation** trained on ~300GB of chess positions. Kernels use SIMD and are selected at runtime via the `Tune` option.
- **Negamax search** uses:
  - Iterative deepening
  - Quiescence search (qsearch)
  - Aspiration windows
  - Transposition tables
  - Null-move pruning
  - Late move reductions
  - And more...
  
## Future Improvements

- Cross-compilation support via a two-phase superbuild (also enabling better native feature detection).
- Add missing UCI commands like `ponder`.
- Enhance time management and responsiveness of the `go` command.
- Implement NNUE accumulator caches and sparse affine evaluation.
- and more pruning = more elo :)
## Acknowledgements

- **Google Highway** – for portable **SIMD** code.
- **Fathom** – for **TBProbe** integration.
- **Grapheus** – for the **NNUE trainer**.
- **Stockfish** - for the well written code, great article on NNUEs and datasets.



