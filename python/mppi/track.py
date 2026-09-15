"""Track generation and cost map baking.

Control points -> periodic spline (scipy.interpolate.splprep) -> resampling at
constant arc length -> 2D grid filled by KD-tree lookup.

Each cell stores the signed lateral offset to the centerline and the
curvilinear progress. Exported as .npy for Python and as a raw binary for C++:
header with the grid dimensions and the domain bounds, then row-major float32.
"""
