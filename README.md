# ChePP

---

## Build Instructions

<details>
<summary>Linux/MacOS</summary>

Tested with GCC 13-14-15-16, Clang 17-18, Apple-clang 17
```bash
cmake -S . -B build -G Ninja
cmake --build build --target ChePP --config Release
```

</details> 

<details> 
<summary>Windows</summary>

Tested with MSVC 18 2026 \
Explicit target (SSE2, AVX, AVX2, AVX512):
```bash
cmake -S . -B build -G "Visual Studio 1x 20xx" -A x64 -DTARGET_ARCH=AVX2
cmake --build build --target ChePP --config Release
```
</details>

## Usage

Support all major UCI commands. Can be use with a UCI chess app (UI, python lichess...)

Example usage:

```bash
  position startpos moves e2e4
  go movetime 1000
  setoption name Tune // find best nnue config
  go depth 20
  quit
```




