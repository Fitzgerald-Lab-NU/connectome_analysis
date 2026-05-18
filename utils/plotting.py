import numpy as np
def pq_to_xy(p, q, hexsize=1.0, rotate=-np.pi/6):
    """
    Convert Zhao-style hex coordinates to 2D xy for plotting.
    """
    a = q
    b = -p

    x = hexsize * np.sqrt(3) * (a + b / 2)
    y = hexsize * 1.5 * b

    if rotate is not None:
        c, s = np.cos(rotate), np.sin(rotate)
        xr = c * x - s * y
        yr = s * x + c * y
        return xr, yr

    return x, y