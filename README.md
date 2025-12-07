# zmai

The Zig Mini AI (ZMAI) Library is a miniature machine learning library intented for training tiny models on the CPU (memory bandwidth is likely to become a bottleneck when using the GPU). This library is currently being developed solely to support [Budget Tetris](https://github.com/Zemogus/Budget-Tetris), but may be used as a standalone library as well.

Run the examples (inside /src/examples) with:

```sh
zig build run -Dexample=<example_name> -Doptions=ReleaseFast
```
