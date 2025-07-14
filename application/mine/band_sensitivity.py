import numpy as np
from pprint import pprint

def make_band_edges(r_max: float, k: int) -> np.ndarray:
    """
    Equal-width concentric-band edges for a given outer radius and band count.

    Parameters
    ----------
    r_max : float
        Outer radius (µm, px, or any consistent distance unit).
    k : int
        Number of bands.

    Returns
    -------
    np.ndarray
        Array of length k+1 with edges `[0, Δr, 2Δr, …, r_max]`,
        where Δr = r_max / k.
    """
    return np.linspace(0, r_max, k + 1)


# --- example grid -----------------------------------------------------------
outer_radii = [300, 500, 600]   # µm
band_counts = [1, 3, 5]

idx = 0

K = band_counts[idx]
R_max = band_counts[idx]

radii = make_band_edges(R_max, K)

print()
# band_edges = {
#     (R, K): make_band_edges(R, K)           # dict keyed by (Rmax, K)
#     for R in outer_radii
#     for K in band_counts
# }

# # pretty-print to verify
# for (R, K), edges in band_edges.items():
#     width = edges[1] - edges[0] if len(edges) > 1 else R   # band width
#     print(f"R_max={R:>3} µm | K={K} → Δr={width:>6.1f} µm | edges={edges}")
