import numpy as np


DOF_PER_NODE = 3


# ============================================================
# BASIC
# ============================================================

def dof_indices(node_id):
    """Return [UX, UY, RZ] global DOF for a node."""
    i = int(node_id) * DOF_PER_NODE
    return [i, i + 1, i + 2]


def rectangular_section(b, h):
    """
    Rectangular section.

    b, h : m

    Returns:
        A : m2
        I : m4
    """
    b = float(b)
    h = float(h)

    A = b * h
    I = b * h**3 / 12.0

    return A, I


def E_mpa_to_kn_m2(E_MPa):
    """MPa -> kN/m2"""
    return float(E_MPa) * 1000.0


# ============================================================
# ELEMENT STIFFNESS
# ============================================================

def beam_element_stiffness(E, A, I, L):
    """
    2D Euler-Bernoulli beam element.

    DOF:
    [u1, v1, theta1, u2, v2, theta2]

    E : kN/m2
    A : m2
    I : m4
    L : m
    """

    if L <= 0:
        raise ValueError("Element length must be > 0.")

    EA_L = E * A / L
    EI = E * I

    return np.array([
        [
            EA_L, 0, 0,
            -EA_L, 0, 0
        ],
        [
            0,
            12 * EI / L**3,
            6 * EI / L**2,
            0,
            -12 * EI / L**3,
            6 * EI / L**2
        ],
        [
            0,
            6 * EI / L**2,
            4 * EI / L,
            0,
            -6 * EI / L**2,
            2 * EI / L
        ],
        [
            -EA_L, 0, 0,
            EA_L, 0, 0
        ],
        [
            0,
            -12 * EI / L**3,
            -6 * EI / L**2,
            0,
            12 * EI / L**3,
            -6 * EI / L**2
        ],
        [
            0,
            6 * EI / L**2,
            2 * EI / L,
            0,
            -6 * EI / L**2,
            4 * EI / L
        ]
    ], dtype=float)


# ============================================================
# GLOBAL MATRIX
# ============================================================

def assemble_global_stiffness(
    nodes,
    elements,
    E_MPa,
    A,
    I
):

    n_nodes = len(nodes)
    n_dof = n_nodes * DOF_PER_NODE

    K = np.zeros((n_dof, n_dof))

    E = E_mpa_to_kn_m2(E_MPa)

    x = {
        int(node["id"]): float(node["x"])
        for node in nodes
    }

    element_data = []

    for element in elements:

        eid = int(element["id"])
        ni = int(element["node_i"])
        nj = int(element["node_j"])

        xi = x[ni]
        xj = x[nj]

        L = abs(xj - xi)

        ke = beam_element_stiffness(
            E,
            A,
            I,
            L
        )

        dofs = (
            dof_indices(ni)
            + dof_indices(nj)
        )

        for i in range(6):
            for j in range(6):
                K[dofs[i], dofs[j]] += ke[i, j]

        element_data.append({
            "id": eid,
            "node_i": ni,
            "node_j": nj,
            "L": L,
            "dofs": dofs,
            "k_local": ke
        })

    return K, element_data


# ============================================================
# NODAL LOADS
# ============================================================

def create_load_vector(nodes, loads):

    n_dof = len(nodes) * DOF_PER_NODE

    F = np.zeros(n_dof)

    for load in loads:

        node = int(load["node"])

        dofs = dof_indices(node)

        F[dofs[0]] += float(
            load.get("fx", 0.0)
        )

        F[dofs[1]] += float(
            load.get("fy", 0.0)
        )

        F[dofs[2]] += float(
            load.get("mz", 0.0)
        )

    return F


# ============================================================
# SUPPORTS
# ============================================================

def get_restrained_dofs(supports):

    restrained = []

    for support in supports:

        node = int(support["node"])

        ux, uy, rz = dof_indices(node)

        if support.get("ux", False):
            restrained.append(ux)

        if support.get("uy", False):
            restrained.append(uy)

        if support.get("rz", False):
            restrained.append(rz)

    return sorted(set(restrained))


# ============================================================
# SOLVER
# ============================================================

def solve_system(K, F, restrained_dofs):

    n = len(F)

    all_dofs = np.arange(n)

    restrained = np.array(
        sorted(set(restrained_dofs)),
        dtype=int
    )

    free = np.array(
        [
            i for i in all_dofs
            if i not in restrained
        ],
        dtype=int
    )

    if len(free) == 0:
        raise ValueError(
            "Tidak ada DOF bebas."
        )

    Kff = K[
        np.ix_(free, free)
    ]

    Ff = F[free]

    condition_number = np.linalg.cond(Kff)

    if not np.isfinite(condition_number):
        raise ValueError(
            "Global stiffness matrix singular."
        )

    if condition_number > 1e14:
        raise ValueError(
            "Struktur tidak stabil atau "
            "tumpuan belum mencukupi."
        )

    Uf = np.linalg.solve(
        Kff,
        Ff
    )

    U = np.zeros(n)

    U[free] = Uf

    R = K @ U - F

    return (
        U,
        R,
        free,
        restrained,
        condition_number
    )


# ============================================================
# ELEMENT FORCE
# ============================================================

def calculate_element_forces(
    U,
    element_data
):

    results = []

    for element in element_data:

        dofs = element["dofs"]

        ke = element["k_local"]

        ue = U[dofs]

        fe = ke @ ue

        results.append({
            "id": element["id"],
            "node_i": element["node_i"],
            "node_j": element["node_j"],
            "L": element["L"],
            "displacements": ue,
            "forces": fe
        })

    return results


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze(
    nodes,
    elements,
    supports,
    loads,
    E_MPa,
    A,
    I
):

    K, element_data = (
        assemble_global_stiffness(
            nodes,
            elements,
            E_MPa,
            A,
            I
        )
    )

    F = create_load_vector(
        nodes,
        loads
    )

    restrained = (
        get_restrained_dofs(
            supports
        )
    )

    (
        U,
        R,
        free,
        restrained,
        condition
    ) = solve_system(
        K,
        F,
        restrained
    )

    element_forces = (
        calculate_element_forces(
            U,
            element_data
        )
    )

    return {
        "K": K,
        "F": F,
        "U": U,
        "R": R,
        "free_dofs": free,
        "restrained_dofs": restrained,
        "condition_number": condition,
        "element_data": element_data,
        "element_forces": element_forces
    }


# ============================================================
# RESULT HELPERS
# ============================================================

def get_node_displacements(nodes, U):

    result = []

    for node in nodes:

        nid = int(node["id"])

        dofs = dof_indices(nid)

        result.append({
            "Node": nid,
            "x": node["x"],
            "UX": U[dofs[0]],
            "UY": U[dofs[1]],
            "RZ": U[dofs[2]]
        })

    return result


def get_reactions(nodes, R):

    result = []

    for node in nodes:

        nid = int(node["id"])

        dofs = dof_indices(nid)

        result.append({
            "Node": nid,
            "Rx": R[dofs[0]],
            "Ry": R[dofs[1]],
            "Mz": R[dofs[2]]
        })

    return result