# AlexZero Connect Four

An AlphaZero-style agent that learned Connect Four purely by self-play, playable in the browser with no server.

**Play:** https://alexbaker0.github.io/connect4/

## How it runs in the browser

The agent was trained in PyTorch (residual conv net, 64 channels × 4 blocks, policy + value heads). For the web:

- `export_web.py` (in the training project) folds BatchNorm into the convolutions and writes `weights.bin` + `weights.json`.
- `src/net.c` is the forward pass in C, compiled to WebAssembly SIMD (`net.wasm`), about 1 ms per evaluation.
- `engine.js` has the game rules, a plain-JS fallback network, and the same PUCT Monte Carlo Tree Search as the Python. It was checked against PyTorch: outputs agree to ~1e-5 and MCTS visit counts match exactly.
- `worker.js` runs the search in a Web Worker so the page stays responsive.

Difficulty = MCTS simulations per move: Easy 100, Medium 300, Hard 700.

## Rebuilding the wasm

```
clang --target=wasm32 -O3 -msimd128 -mbulk-memory -nostdlib -Wl,--no-entry -o net.wasm src/net.c
```
