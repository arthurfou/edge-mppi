"""Latency sweeps: K from 256 to 32768, T, model, compute budget (power limit, SM share).

Reports p50 and p99, not just the mean. Jitter matters as much as the average
under a real-time budget.
"""
