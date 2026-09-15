"""Per-trajectory cost.

Terms: lateral offset to the centerline, curvilinear progress, off-track
penalty, control regularization.

Each term is normalized to the same order of magnitude. Without that, tuning
lambda against the cost scale is guesswork.
"""
