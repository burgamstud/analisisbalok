import math
import numpy as np


# ============================================================
# BURGAM.STUD
# CLASSICAL BEAM ANALYSIS + RC BEAM DESIGN
#
# Classical Beam Analysis
# -----------------------
# Geometry : m
# E        : MPa
# Loads    : kN / kN.m
# Deflection returned in m
#
# RC Design
# ---------
# Geometry : mm
# fc, fy   : MPa
# Mu       : kN.m
# Vu       : kN
#
# Reference concept:
# SNI 2847:2019
# ============================================================


# ============================================================
# BASIC UTILITIES
# ============================================================

def _safe_float(value, default=0.0):
    """
    Convert value to float safely.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _validate_positive(value, name):
    """
    Validate positive numerical value.
    """
    value = float(value)

    if value <= 0:
        raise ValueError(
            f"{name} harus lebih besar dari nol."
        )

    return value


# ============================================================
# SECTION PROPERTIES
# ============================================================

def rectangular_section(b, h):
    """
    Rectangular beam section.

    Parameters
    ----------
    b : float
        Width, m
    h : float
        Height, m

    Returns
    -------
    A : float
        Area, m2
    I : float
        Moment of inertia, m4
    """

    b = _validate_positive(b, "Lebar b")
    h = _validate_positive(h, "Tinggi h")

    A = b * h
    I = b * h**3 / 12.0

    return A, I


# ============================================================
# LOAD HELPERS
# ============================================================

def _load_magnitude(load):
    """
    Return load magnitude.

    App convention:
        positive vertical load = downward
    """

    return _safe_float(
        load.get("magnitude", 0.0)
    )


def _load_position(load):
    """
    Return point load / moment position.
    """

    return _safe_float(
        load.get("position", 0.0)
    )


# ============================================================
# CLASSICAL REACTION CALCULATION
# ============================================================

def _calculate_reactions(
    L,
    loads,
):
    """
    Calculate reactions for a simply supported beam.

    Assumption:
        - Support A at x = 0
        - Support B at x = L
        - Vertical loads only
        - Point load
        - UDL
        - Applied moment

    Sign convention:
        Downward load = positive magnitude
        Clockwise moment = positive

    Returns
    -------
    RA, RB
    """

    L = _validate_positive(
        L,
        "Panjang L"
    )

    total_vertical = 0.0

    moment_about_A = 0.0

    for load in loads:

        load_type = load.get("type")

        # ----------------------------------------------------
        # POINT LOAD
        # ----------------------------------------------------

        if load_type == "Point Load":

            P = _load_magnitude(load)

            x = _load_position(load)

            total_vertical += P

            moment_about_A += P * x

        # ----------------------------------------------------
        # UDL
        # ----------------------------------------------------

        elif load_type == "UDL":

            w = _load_magnitude(load)

            x1 = _safe_float(
                load.get("start", 0.0)
            )

            x2 = _safe_float(
                load.get("end", L)
            )

            if x2 <= x1:
                continue

            load_total = w * (x2 - x1)

            centroid = (
                x1 + x2
            ) / 2.0

            total_vertical += load_total

            moment_about_A += (
                load_total * centroid
            )

        # ----------------------------------------------------
        # APPLIED MOMENT
        # ----------------------------------------------------

        elif load_type == "Moment":

            M = _load_magnitude(load)

            # Clockwise positive load convention.
            #
            # For equilibrium:
            # RA*L + M - Σ(P*x) = 0
            #
            # Therefore:
            # RB = (Σ(P*x) - M) / L
            moment_about_A -= M

    # --------------------------------------------------------
    # EQUILIBRIUM
    # --------------------------------------------------------

    RB = (
        moment_about_A / L
    )

    RA = (
        total_vertical - RB
    )

    return RA, RB


# ============================================================
# CLASSICAL SHEAR FUNCTION
# ============================================================

def _shear_at_x(
    x,
    L,
    loads,
    RA,
):
    """
    Calculate shear force V(x).

    Convention:
        Positive shear = upward reaction side.

    Downward point load causes negative jump.
    """

    V = float(RA)

    for load in loads:

        load_type = load.get("type")

        # ----------------------------------------------------
        # POINT LOAD
        # ----------------------------------------------------

        if load_type == "Point Load":

            P = _load_magnitude(load)

            xp = _load_position(load)

            if x >= xp:
                V -= P

        # ----------------------------------------------------
        # UDL
        # ----------------------------------------------------

        elif load_type == "UDL":

            w = _load_magnitude(load)

            x1 = _safe_float(
                load.get("start", 0.0)
            )

            x2 = _safe_float(
                load.get("end", L)
            )

            if x <= x1:
                continue

            loaded_length = min(
                x,
                x2
            ) - x1

            if loaded_length > 0:
                V -= (
                    w * loaded_length
                )

    return V


# ============================================================
# CLASSICAL MOMENT FUNCTION
# ============================================================

def _moment_at_x(
    x,
    L,
    loads,
    RA,
):
    """
    Calculate bending moment M(x).

    Convention:
        Sagging positive.
    """

    M = RA * x

    for load in loads:

        load_type = load.get("type")

        # ----------------------------------------------------
        # POINT LOAD
        # ----------------------------------------------------

        if load_type == "Point Load":

            P = _load_magnitude(load)

            xp = _load_position(load)

            if x >= xp:
                M -= P * (
                    x - xp
                )

        # ----------------------------------------------------
        # UDL
        # ----------------------------------------------------

        elif load_type == "UDL":

            w = _load_magnitude(load)

            x1 = _safe_float(
                load.get("start", 0.0)
            )

            x2 = _safe_float(
                load.get("end", L)
            )

            if x <= x1:
                continue

            a = x1

            b = min(
                x,
                x2
            )

            if b > a:

                loaded_length = b - a

                resultant = (
                    w * loaded_length
                )

                centroid = (
                    a + b
                ) / 2.0

                M -= (
                    resultant
                    * (x - centroid)
                )

        # ----------------------------------------------------
        # APPLIED MOMENT
        # ----------------------------------------------------

        elif load_type == "Moment":

            Mload = _load_magnitude(
                load
            )

            xm = _load_position(
                load
            )

            if x >= xm:

                M += Mload

    return M


# ============================================================
# CLASSICAL DEFLECTION
# ============================================================

def _calculate_deflection(
    x_values,
    M_values,
    E,
    I,
):
    """
    Numerical integration of curvature.

    Euler-Bernoulli:
        EI d²v/dx² = M

    Two integration constants are determined
    using simple-support conditions:

        v(0) = 0
        v(L) = 0
    """

    x = np.asarray(
        x_values,
        dtype=float
    )

    M = np.asarray(
        M_values,
        dtype=float
    )

    if len(x) < 2:
        return np.zeros_like(x)

    EI = (
        float(E)
        * float(I)
    )

    if EI <= 0:
        raise ValueError(
            "EI harus lebih besar dari nol."
        )

    curvature = M / EI

    # First integration:
    # theta(x) = integral curvature dx + C1
    theta0 = np.zeros_like(x)

    for i in range(1, len(x)):

        dx = (
            x[i]
            - x[i - 1]
        )

        theta0[i] = (
            theta0[i - 1]
            + 0.5
            * (
                curvature[i]
                + curvature[i - 1]
            )
            * dx
        )

    # Second integration:
    # v0(x) = integral theta0 dx
    deflection0 = np.zeros_like(x)

    for i in range(1, len(x)):

        dx = (
            x[i]
            - x[i - 1]
        )

        deflection0[i] = (
            deflection0[i - 1]
            + 0.5
            * (
                theta0[i]
                + theta0[i - 1]
            )
            * dx
        )

    # Boundary condition v(L)=0
    L = x[-1]

    if L <= 0:
        return np.zeros_like(x)

    C1 = (
        -deflection0[-1]
        / L
    )

    deflection = (
        deflection0
        + C1 * x
    )

    return deflection


# ============================================================
# CLASSICAL BEAM SOLVER
# ============================================================

def solve_beam(
    L,
    b,
    h,
    E_MPa,
    loads,
    n_points=1001,
):
    """
    Classical simply supported beam analysis.

    Supports:
        A = pin at x = 0
        B = roller at x = L

    Loads:
        - Point Load
        - UDL
        - Moment

    Parameters
    ----------
    L : float
        Beam span, m

    b : float
        Width, m

    h : float
        Height, m

    E_MPa : float
        Elastic modulus, MPa

    loads : list
        Load definitions from app.py.

    n_points : int
        Number of points for diagrams.

    Returns
    -------
    dict
        Compatible with app.py.
    """

    L = _validate_positive(
        L,
        "Panjang L"
    )

    b = _validate_positive(
        b,
        "Lebar b"
    )

    h = _validate_positive(
        h,
        "Tinggi h"
    )

    E_MPa = _validate_positive(
        E_MPa,
        "Modulus elastisitas E"
    )

    if not isinstance(
        loads,
        (list, tuple)
    ):
        loads = []

    # --------------------------------------------------------
    # SECTION
    # --------------------------------------------------------

    A, I = rectangular_section(
        b,
        h
    )

    # --------------------------------------------------------
    # E
    #
    # MPa = N/mm2
    # For geometry m and force kN:
    #
    # 1 MPa = 1000 kN/m2
    # --------------------------------------------------------

    E = (
        E_MPa
        * 1000.0
    )

    # --------------------------------------------------------
    # REACTIONS
    # --------------------------------------------------------

    RA, RB = _calculate_reactions(
        L=L,
        loads=loads,
    )

    # --------------------------------------------------------
    # X GRID
    # --------------------------------------------------------

    n_points = max(
        101,
        int(n_points)
    )

    x = np.linspace(
        0.0,
        L,
        n_points
    )

    # Add important load locations
    # to improve diagram resolution.

    important_x = [
        0.0,
        L,
    ]

    for load in loads:

        load_type = load.get(
            "type"
        )

        if load_type in [
            "Point Load",
            "Moment",
        ]:

            xp = _load_position(
                load
            )

            if 0.0 <= xp <= L:
                important_x.append(
                    xp
                )

        elif load_type == "UDL":

            x1 = _safe_float(
                load.get(
                    "start",
                    0.0
                )
            )

            x2 = _safe_float(
                load.get(
                    "end",
                    L
                )
            )

            if 0.0 <= x1 <= L:
                important_x.append(
                    x1
                )

            if 0.0 <= x2 <= L:
                important_x.append(
                    x2
                )

    x = np.unique(
        np.concatenate(
            [
                x,
                np.asarray(
                    important_x
                ),
            ]
        )
    )

    x.sort()

    # --------------------------------------------------------
    # SHEAR
    # --------------------------------------------------------

    V = np.array(
        [
            _shear_at_x(
                xi,
                L,
                loads,
                RA,
            )
            for xi in x
        ]
    )

    # --------------------------------------------------------
    # MOMENT
    # --------------------------------------------------------

    M = np.array(
        [
            _moment_at_x(
                xi,
                L,
                loads,
                RA,
            )
            for xi in x
        ]
    )

    # --------------------------------------------------------
    # DEFLECTION
    # --------------------------------------------------------

    deflection = (
        _calculate_deflection(
            x_values=x,
            M_values=M,
            E=E,
            I=I,
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Classical diagrams are based on the load functions.
    # Deflection sign follows positive sagging curvature.
    #
    # For downward loading, resulting deflection is
    # generally negative depending on sign convention.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # EXTREME VALUES
    # --------------------------------------------------------

    max_shear = float(
        np.max(
            np.abs(V)
        )
    )

    max_moment = float(
        np.max(
            np.abs(M)
        )
    )

    shear_index = int(
        np.argmax(
            np.abs(V)
        )
    )

    moment_index = int(
        np.argmax(
            np.abs(M)
        )
    )

    deflection_index = int(
        np.argmax(
            np.abs(deflection)
        )
    )

    max_moment_x = float(
        x[moment_index]
    )

    max_deflection = float(
        deflection[
            deflection_index
        ]
    )

    max_deflection_x = float(
        x[
            deflection_index
        ]
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {
        # Geometry
        "L": L,
        "b": b,
        "h": h,
        "A": A,
        "I": I,
        "E_MPa": E_MPa,

        # Reactions
        "RA": float(RA),
        "RB": float(RB),

        # Diagram
        "x": x,
        "V": V,
        "M": M,
        "deflection": deflection,

        # Maximum values
        "max_shear": max_shear,
        "max_moment": max_moment,
        "max_moment_x": max_moment_x,
        "max_deflection": max_deflection,
        "max_deflection_x": max_deflection_x,

        # Additional data
        "loads": list(loads),
    }


# ============================================================
# RC DESIGN — SNI 2847:2019
# ============================================================

def _beta1_sni(fc):
    """
    β1 according to concrete compressive strength.

    For fc' <= 28 MPa:
        β1 = 0.85

    For fc' > 28 MPa:
        β1 decreases 0.05 every 7 MPa,
        but not less than 0.65.
    """

    fc = float(fc)

    if fc <= 28.0:
        return 0.85

    beta1 = (
        0.85
        - 0.05
        * (
            (fc - 28.0)
            / 7.0
        )
    )

    return max(
        0.65,
        beta1
    )


def _phi_flexure(eps_t):
    """
    Flexural strength reduction factor.

    Simplified strain-based implementation.

    For tension-controlled:
        phi = 0.90

    For compression-controlled:
        phi = 0.65

    Transition region:
        linear interpolation.
    """

    eps_t = float(eps_t)

    eps_y = 0.002

    eps_tc = 0.005

    if eps_t <= eps_y:

        return 0.65

    if eps_t >= eps_tc:

        return 0.90

    return (
        0.65
        + (
            eps_t - eps_y
        )
        / (
            eps_tc - eps_y
        )
        * (
            0.90 - 0.65
        )
    )


def _solve_flexure(
    b,
    h,
    fc,
    fy,
    cover,
    stirrup_dia,
    bar_dia,
    n_bottom,
    Mu,
):
    """
    Calculate singly reinforced rectangular section.

    Returns flexural design parameters.
    """

    # --------------------------------------------------------
    # EFFECTIVE DEPTH
    # --------------------------------------------------------

    d = (
        h
        - cover
        - stirrup_dia
        - bar_dia / 2.0
    )

    if d <= 0:
        raise ValueError(
            "Effective depth d tidak valid. "
            "Periksa h, cover, diameter sengkang "
            "dan diameter tulangan."
        )

    # --------------------------------------------------------
    # MINIMUM REINFORCEMENT
    #
    # Basic SNI 2847 concept:
    #
    # As,min = max(
    #     0.25 sqrt(fc') / fy * bw*d,
    #     1.4/fy * bw*d
    # )
    # --------------------------------------------------------

    As_min_1 = (
        0.25
        * math.sqrt(fc)
        / fy
        * b
        * d
    )

    As_min_2 = (
        1.4
        / fy
        * b
        * d
    )

    As_min = max(
        As_min_1,
        As_min_2
    )

    # --------------------------------------------------------
    # AVAILABLE STEEL
    # --------------------------------------------------------

    As_bar = (
        math.pi
        * bar_dia**2
        / 4.0
    )

    As_provided = (
        n_bottom
        * As_bar
    )

    # --------------------------------------------------------
    # REQUIRED STEEL FROM Mu
    #
    # Solve:
    #
    # Mn = As fy (d - a/2)
    #
    # a = As fy / (0.85 fc b)
    #
    # φMn >= Mu
    #
    # For preliminary design use phi = 0.90.
    # --------------------------------------------------------

    phi_initial = 0.90

    Mu_Nmm = (
        abs(Mu)
        * 1e6
    )

    Mn_required = (
        Mu_Nmm
        / phi_initial
    )

    A_quad = (
        fy**2
        / (
            2.0
            * 0.85
            * fc
            * b
        )
    )

    B_quad = (
        -fy
        * d
    )

    C_quad = (
        Mn_required
    )

    # Equation:
    #
    # A As² + B As + C = 0
    #
    # Select smaller positive root.

    discriminant = (
        B_quad**2
        - 4.0
        * A_quad
        * C_quad
    )

    if discriminant < 0:

        As_required = np.nan

    else:

        sqrt_disc = math.sqrt(
            discriminant
        )

        root1 = (
            -B_quad
            - sqrt_disc
        ) / (
            2.0 * A_quad
        )

        root2 = (
            -B_quad
            + sqrt_disc
        ) / (
            2.0 * A_quad
        )

        positive_roots = [
            r
            for r in [
                root1,
                root2,
            ]
            if r > 0
        ]

        if positive_roots:

            As_required = min(
                positive_roots
            )

        else:

            As_required = np.nan

    # --------------------------------------------------------
    # ENSURE MINIMUM STEEL
    # --------------------------------------------------------

    if np.isfinite(
        As_required
    ):

        As_design = max(
            As_required,
            As_min
        )

    else:

        As_design = np.nan

    # --------------------------------------------------------
    # ACTUAL PROVIDED STEEL
    # --------------------------------------------------------

    if As_provided > 0:

        a = (
            As_provided
            * fy
            / (
                0.85
                * fc
                * b
            )
        )

    else:

        a = 0.0

    # --------------------------------------------------------
    # NEUTRAL AXIS
    # --------------------------------------------------------

    beta1 = _beta1_sni(
        fc
    )

    c = (
        a / beta1
        if beta1 > 0
        else np.nan
    )

    # --------------------------------------------------------
    # TENSION STRAIN
    # --------------------------------------------------------

    eps_cu = 0.003

    if c > 0:

        eps_t = (
            eps_cu
            * (
                d - c
            )
            / c
        )

    else:

        eps_t = np.inf

    # --------------------------------------------------------
    # PHI
    # --------------------------------------------------------

    phi = _phi_flexure(
        eps_t
    )

    # --------------------------------------------------------
    # NOMINAL MOMENT
    # --------------------------------------------------------

    Mn = (
        As_provided
        * fy
        * (
            d - a / 2.0
        )
    )

    phi_Mn = (
        phi
        * Mn
        / 1e6
    )

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    flexure_ok = (
        np.isfinite(
            As_required
        )
        and As_provided >= As_min
        and phi_Mn >= abs(Mu)
    )

    return {
        "d": d,
        "As_min": As_min,
        "As_required": As_design,
        "As_provided": As_provided,
        "beta1": beta1,
        "a": a,
        "c": c,
        "eps_t": eps_t,
        "phi_flex": phi,
        "Mn": Mn / 1e6,
        "phi_Mn": phi_Mn,
        "flexure_ok": bool(
            flexure_ok
        ),
    }


# ============================================================
# RC SHEAR DESIGN
# ============================================================

def _solve_shear(
    b,
    d,
    fc,
    fyv,
    Vu,
    stirrup_dia,
    stirrup_legs,
    stirrup_spacing,
):
    """
    Basic shear design for reinforced concrete beam.

    Uses:
        Vc = 0.17 sqrt(fc') bw d

    For vertical stirrups:
        Vs = Av fyv d / s

    Strength:
        phi Vn = phi (Vc + Vs)

    This is a preliminary non-seismic implementation.
    """

    # --------------------------------------------------------
    # SHEAR STRENGTH REDUCTION
    # --------------------------------------------------------

    phi_v = 0.75

    # --------------------------------------------------------
    # CONCRETE SHEAR CAPACITY
    # --------------------------------------------------------

    Vc_N = (
        0.17
        * math.sqrt(fc)
        * b
        * d
    )

    Vc = (
        Vc_N
        / 1000.0
    )

    # --------------------------------------------------------
    # STIRRUP AREA
    # --------------------------------------------------------

    area_one_leg = (
        math.pi
        * stirrup_dia**2
        / 4.0
    )

    Av = (
        stirrup_legs
        * area_one_leg
    )

    # --------------------------------------------------------
    # STIRRUP MAX SPACING
    #
    # Basic limitation:
    # s <= d/2
    #
    # Additional limit based on d:
    # s <= 600 mm
    # --------------------------------------------------------

    s_max = min(
        d / 2.0,
        600.0
    )

    # --------------------------------------------------------
    # MINIMUM SHEAR REINFORCEMENT
    #
    # Basic approximation:
    #
    # Av,min / s =
    # max(
    #     0.062 sqrt(fc) bw / fyv,
    #     0.35 bw / fyv
    # )
    # --------------------------------------------------------

    Av_s_min_1 = (
        0.062
        * math.sqrt(fc)
        * b
        / fyv
    )

    Av_s_min_2 = (
        0.35
        * b
        / fyv
    )

    Av_s_min_per_s = max(
        Av_s_min_1,
        Av_s_min_2
    )

    Av_min = (
        Av_s_min_per_s
        * stirrup_spacing
    )

    # --------------------------------------------------------
    # Vs
    # --------------------------------------------------------

    Vs_N = (
        Av
        * fyv
        * d
        / stirrup_spacing
    )

    Vs = (
        Vs_N
        / 1000.0
    )

    # --------------------------------------------------------
    # LIMIT VS
    #
    # Basic conservative implementation:
    #
    # Vs,max = 0.66 sqrt(fc) bw d
    # --------------------------------------------------------

    Vs_max_N = (
        0.66
        * math.sqrt(fc)
        * b
        * d
    )

    Vs_max = (
        Vs_max_N
        / 1000.0
    )

    # --------------------------------------------------------
    # NOMINAL + DESIGN STRENGTH
    # --------------------------------------------------------

    Vn = (
        Vc
        + Vs
    )

    phi_Vn = (
        phi_v
        * Vn
    )

    # --------------------------------------------------------
    # CHECKS
    # --------------------------------------------------------

    shear_strength_ok = (
        phi_Vn >= abs(Vu)
    )

    shear_reinf_ok = (
        Av >= Av_min
    )

    shear_vs_limit_ok = (
        Vs <= Vs_max
    )

    spacing_ok = (
        stirrup_spacing <= s_max
    )

    # --------------------------------------------------------
    # LONGITUDINAL BAR CLEAR SPACING
    #
    # Approximate check based on two or more bars.
    # --------------------------------------------------------

    if stirrup_dia > 0:

        clear_width = (
            b
            - 2.0
            * (
                40.0
                + stirrup_dia
            )
        )

    else:

        clear_width = b

    # The app does not pass bar diameter here.
    #
    # This flag will be overridden by the main design
    # function where bar diameter and longitudinal
    # reinforcement are available.

    spacing_long_ok = (
        clear_width > 0
    )

    shear_ok = (
        shear_strength_ok
        and shear_reinf_ok
        and shear_vs_limit_ok
        and spacing_ok
        and spacing_long_ok
    )

    return {
        "Vc": Vc,
        "Vs": Vs,
        "Vn": Vn,
        "phi_Vn": phi_Vn,
        "Av": Av,
        "Av_min": Av_min,
        "Vs_max": Vs_max,
        "s_max": s_max,
        "phi_shear": phi_v,
        "shear_strength_ok": bool(
            shear_strength_ok
        ),
        "shear_reinf_ok": bool(
            shear_reinf_ok
        ),
        "shear_vs_limit_ok": bool(
            shear_vs_limit_ok
        ),
        "spacing_ok": bool(
            spacing_ok
        ),
        "spacing_long_ok": bool(
            spacing_long_ok
        ),
        "shear_ok": bool(
            shear_ok
        ),
    }


# ============================================================
# RC BEAM DESIGN
# ============================================================

def design_rc_beam_sni(
    b,
    h,
    fc,
    fy,
    fyv,
    cover,
    Mu,
    Vu,
    bar_dia,
    n_bottom,
    n_top,
    stirrup_dia,
    stirrup_legs,
    stirrup_spacing,
):
    """
    Reinforced concrete rectangular beam design.

    Compatible with app.py.

    Parameters
    ----------
    b, h : mm
    fc : MPa
    fy : MPa
    fyv : MPa
    cover : mm
    Mu : kN.m
    Vu : kN
    bar_dia : mm
    n_bottom : number of bottom bars
    n_top : number of top bars
    stirrup_dia : mm
    stirrup_legs : number of stirrup legs
    stirrup_spacing : mm

    Returns
    -------
    dict
        Result dictionary used by app.py.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    b = _validate_positive(
        b,
        "Lebar b"
    )

    h = _validate_positive(
        h,
        "Tinggi h"
    )

    fc = _validate_positive(
        fc,
        "fc'"
    )

    fy = _validate_positive(
        fy,
        "fy longitudinal"
    )

    fyv = _validate_positive(
        fyv,
        "fy sengkang"
    )

    cover = _validate_positive(
        cover,
        "Selimut beton"
    )

    bar_dia = _validate_positive(
        bar_dia,
        "Diameter tulangan longitudinal"
    )

    stirrup_dia = _validate_positive(
        stirrup_dia,
        "Diameter sengkang"
    )

    stirrup_spacing = _validate_positive(
        stirrup_spacing,
        "Spacing sengkang"
    )

    n_bottom = int(
        n_bottom
    )

    n_top = int(
        n_top
    )

    stirrup_legs = int(
        stirrup_legs
    )

    if n_bottom < 1:
        raise ValueError(
            "Jumlah tulangan bawah minimal 1."
        )

    if stirrup_legs < 2:
        raise ValueError(
            "Jumlah kaki sengkang minimal 2."
        )

    # ========================================================
    # EFFECTIVE DEPTH
    # ========================================================

    d = (
        h
        - cover
        - stirrup_dia
        - bar_dia / 2.0
    )

    if d <= 0:

        raise ValueError(
            "Effective depth d <= 0. "
            "Periksa dimensi balok, cover, "
            "diameter sengkang, dan tulangan."
        )

    # ========================================================
    # FLEXURE
    # ========================================================

    flexure = _solve_flexure(
        b=b,
        h=h,
        fc=fc,
        fy=fy,
        cover=cover,
        stirrup_dia=stirrup_dia,
        bar_dia=bar_dia,
        n_bottom=n_bottom,
        Mu=Mu,
    )

    # ========================================================
    # SHEAR
    # ========================================================

    shear = _solve_shear(
        b=b,
        d=d,
        fc=fc,
        fyv=fyv,
        Vu=Vu,
        stirrup_dia=stirrup_dia,
        stirrup_legs=stirrup_legs,
        stirrup_spacing=stirrup_spacing,
    )

    # ========================================================
    # LONGITUDINAL CLEAR SPACING
    # ========================================================
    #
    # Clear distance between bottom bars.
    #
    # Available width inside stirrup:
    #
    # b_inside =
    # b - 2(cover + stirrup_dia)
    #
    # Bar center spacing:
    #
    # available / (n-1)
    #
    # Clear spacing =
    # center spacing - bar diameter
    #
    # Basic minimum:
    # max(
    #     25 mm,
    #     db
    # )
    #
    # This is a basic detailing check.
    # ========================================================

    if n_bottom <= 1:

        clear_spacing = np.inf
        spacing_long_ok = True

    else:

        inside_width = (
            b
            - 2.0
            * (
                cover
                + stirrup_dia
            )
        )

        if inside_width <= 0:

            clear_spacing = -np.inf
            spacing_long_ok = False

        else:

            center_spacing = (
                inside_width
                / (
                    n_bottom - 1
                )
            )

            clear_spacing = (
                center_spacing
                - bar_dia
            )

            minimum_clear_spacing = max(
                25.0,
                bar_dia
            )

            spacing_long_ok = (
                clear_spacing
                >= minimum_clear_spacing
            )

    # ========================================================
    # UPDATE SHEAR LONGITUDINAL CHECK
    # ========================================================

    shear["spacing_long_ok"] = bool(
        spacing_long_ok
    )

    shear["shear_ok"] = bool(
        shear["shear_strength_ok"]
        and shear["shear_reinf_ok"]
        and shear["shear_vs_limit_ok"]
        and shear["spacing_ok"]
        and spacing_long_ok
    )

    # ========================================================
    # RETURN
    # ========================================================

    result = {}

    # --------------------------------------------------------
    # FLEXURE
    # --------------------------------------------------------

    result.update(
        {
            "d": flexure["d"],
            "As_min": flexure["As_min"],
            "As_required": flexure[
                "As_required"
            ],
            "As_provided": flexure[
                "As_provided"
            ],
            "beta1": flexure[
                "beta1"
            ],
            "a": flexure["a"],
            "c": flexure["c"],
            "eps_t": flexure[
                "eps_t"
            ],
            "phi_flex": flexure[
                "phi_flex"
            ],
            "Mn": flexure["Mn"],
            "phi_Mn": flexure[
                "phi_Mn"
            ],
            "flexure_ok": flexure[
                "flexure_ok"
            ],
        }
    )

    # --------------------------------------------------------
    # SHEAR
    # --------------------------------------------------------

    result.update(
        {
            "Vc": shear["Vc"],
            "Vs": shear["Vs"],
            "Vn": shear["Vn"],
            "phi_Vn": shear[
                "phi_Vn"
            ],
            "Av": shear["Av"],
            "Av_min": shear[
                "Av_min"
            ],
            "Vs_max": shear[
                "Vs_max"
            ],
            "s_max": shear[
                "s_max"
            ],
            "phi_shear": shear[
                "phi_shear"
            ],
            "shear_strength_ok": shear[
                "shear_strength_ok"
            ],
            "shear_reinf_ok": shear[
                "shear_reinf_ok"
            ],
            "shear_vs_limit_ok": shear[
                "shear_vs_limit_ok"
            ],
            "spacing_ok": shear[
                "spacing_ok"
            ],
            "spacing_long_ok": shear[
                "spacing_long_ok"
            ],
            "shear_ok": shear[
                "shear_ok"
            ],
        }
    )

    # --------------------------------------------------------
    # DETAILING INFORMATION
    # --------------------------------------------------------

    result.update(
        {
            "b": b,
            "h": h,
            "fc": fc,
            "fy": fy,
            "fyv": fyv,
            "cover": cover,
            "bar_dia": bar_dia,
            "n_bottom": n_bottom,
            "n_top": n_top,
            "stirrup_dia": stirrup_dia,
            "stirrup_legs": stirrup_legs,
            "stirrup_spacing": stirrup_spacing,
            "Mu": Mu,
            "Vu": Vu,
            "clear_spacing_longitudinal": (
                clear_spacing
            ),
        }
    )

    return result


# ============================================================
# BACKWARD-COMPATIBILITY HELPERS
# ============================================================

def mp_a_to_kn_m2(E_MPa):
    """
    Backward-compatible helper.

    MPa -> kN/m2
    """
    return float(E_MPa) * 1000.0


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Simple test
    # --------------------------------------------------------

    test_loads = [
        {
            "name": "P1",
            "type": "Point Load",
            "magnitude": 20.0,
            "position": 5.0,
        }
    ]

    result = solve_beam(
        L=10.0,
        b=0.25,
        h=0.40,
        E_MPa=23500.0,
        loads=test_loads,
    )

    print(
        "======================================"
    )
    print(
        "BURGAM.STUD BEAM SOLVER TEST"
    )
    print(
        "======================================"
    )

    print(
        f"RA = {result['RA']:.3f} kN"
    )

    print(
        f"RB = {result['RB']:.3f} kN"
    )

    print(
        f"Mmax = "
        f"{result['max_moment']:.3f} kNm"
    )

    print(
        f"x Mmax = "
        f"{result['max_moment_x']:.3f} m"
    )

    print(
        f"delta max = "
        f"{result['max_deflection'] * 1000:.3f} mm"
    )

    print(
        f"x delta max = "
        f"{result['max_deflection_x']:.3f} m"
    )