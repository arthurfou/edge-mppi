"""Non-regression gate: Python costs vs CUDA costs on a fixed noise tensor.

Generates the (K, T, control_dim) noise in NumPy from the config seed, saves it,
feeds the same tensor to both implementations, compares the K per-trajectory
costs one by one. They must agree to about 1e-4 in FP32. A single mismatch
means a dynamics bug, and pinpoints it far faster than comparing final
trajectories.

Also holds the kinematic/dynamic cross-check from step 2: at low speed and low
steering the two models must produce near-identical trajectories.
"""
