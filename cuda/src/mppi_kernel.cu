// Step 3: one thread per trajectory, horizon loop inside the kernel, cost
// accumulated on the fly, one float written per thread. Correctness first.
// Step 4 fuses rollout and cost and keeps the state in registers.
