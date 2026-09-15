# Optimization table

One row per change, added at the time of measurement. Configuration for every
row unless stated otherwise: K = 8192, T = 50, dynamic model, FP32.

| # | Change | Why a gain was expected | Time / iter (RTX 3060) | Time / iter (Orin Nano) | Cumulative speedup |
|---|---|---|---|---|---|
| 0 | Naive port, one thread per trajectory | baseline | | | 1.00x |
| 1 | Fused rollout + cost, state in registers | | | | |
| 2 | cuRAND in device, one state per thread | | | | |
| 3 | Warp-level reduction | | | | |
| 4 | Cost map in texture memory | | | | |
| 5 | Structure-of-arrays layout | | | | |
| 6 | Fast math intrinsics | | | | |

Changes that produced no gain belong in `docs/journal.md`, with what was
expected and what was measured.
