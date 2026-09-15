# Work journal

One entry per session. Record what was measured, not what was expected.
Optimizations that gave nothing go in here too: that is the record showing
measurement rather than belief.

Format:

## YYYY-MM-DD - step N - short title

**Goal.**

**Done.**

**Measured.** Numbers, with the config they came from (K, T, model, hardware,
power mode).

**No effect / reverted.** What was tried, why a gain was expected, what
actually happened.

**Next.**

---

## 2026-09-09 - step 0 - repo foundations

**Goal.** Repo structure, CMake, pixi environment, shared YAML config.

**Done.** Directory layout, `pixi.toml` with linux-64 and linux-aarch64,
CMake with CUDA as a first-class language targeting sm_86 and sm_87,
`config/mppi.yaml` as the single shared config.

**Measured.**

**No effect / reverted.**

**Next.** Step 1, kinematic MPPI in NumPy.
