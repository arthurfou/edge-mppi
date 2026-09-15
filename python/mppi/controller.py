"""Vectorized NumPy MPPI, the reference implementation.

Per control step:
  1. sample noise of shape (K, T, control_dim)
  2. roll out the dynamics
  3. accumulate a cost per trajectory
  4. w = exp(-(S - S_min) / lambda), normalized
  5. weighted mean over the noise, applied to the nominal sequence
  6. shift the nominal sequence by one step

This is the ground truth the CUDA kernel is checked against.
"""
