# Build
#### Only native builds are currently supported
## Linux

```bash
cmake -S . -B build
cmake --build build --target ChePP --config Release
```

## Windows
#### Explicit target on windows (SSE2, AVX, AVX2, AVX512)
```bash
cmake -S . -B build -G "Visual Studio 17 20xx" -A x64 -DTARGET_ARCH=AVX2
cmake --build build --target ChePP --config Release
```
