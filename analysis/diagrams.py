import numpy as np


def calculate_reactions(P, L, a):
    """
    Menghitung reaksi tumpuan balok sederhana
    dengan beban titik P pada jarak a dari tumpuan A.
    """

    b = L - a

    RA = P * b / L
    RB = P * a / L

    return RA, RB


def calculate_diagrams(P, L, a, E, I, n=500):
    """
    Menghasilkan data SFD, BMD, dan deflection
    untuk balok sederhana dengan satu beban titik.
    """

    x = np.linspace(0, L, n)

    RA, RB = calculate_reactions(P, L, a)

    shear = np.zeros_like(x)
    moment = np.zeros_like(x)
    deflection = np.zeros_like(x)

    b = L - a

    for i, xi in enumerate(x):

        # ==========================
        # SHEAR FORCE
        # ==========================

        if xi < a:
            shear[i] = RA
        else:
            shear[i] = RA - P

        # ==========================
        # BENDING MOMENT
        # ==========================

        if xi <= a:
            moment[i] = RA * xi
        else:
            moment[i] = RA * xi - P * (xi - a)

        # ==========================
        # DEFLECTION
        # ==========================

        if xi <= a:

            deflection[i] = (
                P * b * xi
                / (6 * L * E * I)
                * (
                    L**2
                    - b**2
                    - xi**2
                )
            )

        else:

            deflection[i] = (
                P * a * (L - xi)
                / (6 * L * E * I)
                * (
                    L**2
                    - a**2
                    - (L - xi)**2
                )
            )

    # Deflection ke bawah dibuat negatif
    deflection = -deflection

    return {
        "x": x,
        "shear": shear,
        "moment": moment,
        "deflection": deflection,
        "RA": RA,
        "RB": RB
    }