import numpy as np
import pandas as pd

### HELPER FUNCTIONS FOR HEX GEOMETRY###
def pq_to_xy(p, q, flip_p=True, hexsize: float = 1.0, rotate: float | None = -np.pi / 6):
    """Convert Zhao-style hex coordinates to 2D xy coordinates."""
    a = q
    if flip_p:
        b = -p
    else:
        b = p
    x = hexsize * np.sqrt(3) * (a + b / 2)
    y = hexsize * 1.5 * b

    if rotate is not None:
        c, s = np.cos(rotate), np.sin(rotate)
        xr = c * x - s * y
        yr = s * x + c * y
        return xr, yr

    return x, y

def hex_distance(dp, dq):
    return np.maximum.reduce([
        np.abs(dp),
        np.abs(dq),
        np.abs(dq - dp)
    ])


def hex_offsets(radius):
    rows = []

    for dp in range(-radius, radius + 1):
        for dq in range(-radius, radius + 1):
            if hex_distance(dp, dq) <= radius:
                rows.append((dp, dq))

    return pd.DataFrame(rows, columns=["dp", "dq"])

