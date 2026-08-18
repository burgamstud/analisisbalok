import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.beam import solve_beam, design_rc_beam_sni

from analysis.matrix_beam import (
    analyze as matrix_analyze,
    get_reactions,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BURGAM.STUD",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .status-lock {
        padding: 10px 14px;
        border-radius: 9px;
        background: #e8f5e9;
        border: 1px solid #b7dfb9;
        font-weight: 700;
    }

    .status-unlock {
        padding: 10px 14px;
        border-radius: 9px;
        background: #fff8e1;
        border: 1px solid #f0d98c;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "locked": False,
    "results": None,
    "matrix_results": {},
    "rc_design": None,

    "cases": {
        "DL": []
    },

    "case_order": [
        "DL"
    ],

    "active_case": "DL",

    "combinations": [
        {
            "name": "1.4DL",
            "factors": {
                "DL": 1.4
            }
        },
        {
            "name": "1.2DL 1.6LL",
            "factors": {
                "DL": 1.2
            }
        }
    ],

    "supports": [
        {
            "ID": "A",
            "Position": 0.0,
            "Type": "Pin",
        },
        {
            "ID": "B",
            "Position": 10.0,
            "Type": "Roll",
        },
    ],
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def rerun():
    st.rerun()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def combo_text(combo):
    """Generate readable load combination text."""

    parts = []

    for case_name, factor in combo["factors"].items():

        factor = float(factor)

        if abs(factor) > 1e-12:
            parts.append(
                f"{factor:g}{case_name}"
            )

    return " + ".join(parts) if parts else "0"


def build_combination_loads(combo):
    """
    Build loads from several load cases according
    to the selected load combination.
    """

    loads = []

    for case_name, factor in combo["factors"].items():

        factor = float(factor)

        for load in st.session_state.cases.get(case_name, []):

            item = dict(load)

            item["magnitude"] = (
                float(item["magnitude"]) * factor
            )

            item["name"] = (
                f"{case_name} | {load['name']}"
            )

            loads.append(item)

    return loads


# ============================================================
# MATRIX MODEL
# ============================================================

def build_matrix_model(L, supports, loads):
    """
    Create nodes and beam elements.

    Nodes are created at:
    - beam ends
    - supports
    - point loads
    - moments
    - UDL start/end
    """

    positions = {
        0.0,
        float(L),
    }

    # --------------------------------------------------------
    # Supports
    # --------------------------------------------------------

    for support in supports:

        x = float(
            support["Position"]
        )

        if 0.0 <= x <= float(L):
            positions.add(x)

    # --------------------------------------------------------
    # Loads
    # --------------------------------------------------------

    for load in loads:

        load_type = load["type"]

        if load_type in [
            "Point Load",
            "Moment",
        ]:

            x = float(
                load["position"]
            )

            if 0.0 <= x <= float(L):
                positions.add(x)

        elif load_type == "UDL":

            x1 = float(
                load["start"]
            )

            x2 = float(
                load["end"]
            )

            if 0.0 <= x1 <= float(L):
                positions.add(x1)

            if 0.0 <= x2 <= float(L):
                positions.add(x2)

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    positions = sorted(
        positions
    )

    nodes = [
        {
            "id": i,
            "x": x,
        }
        for i, x in enumerate(positions)
    ]

    # --------------------------------------------------------
    # Elements
    # --------------------------------------------------------

    elements = []

    for i in range(
        len(nodes) - 1
    ):

        elements.append(
            {
                "id": i,
                "node_i": i,
                "node_j": i + 1,
            }
        )

    return nodes, elements


def find_closest_node(nodes, x):
    """Find node closest to coordinate x."""

    return min(
        nodes,
        key=lambda node:
        abs(float(node["x"]) - float(x))
    )


def build_matrix_supports(nodes, supports):

    result = []

    support_dof = {
        "Pin": {
            "ux": True,
            "uy": True,
            "rz": False,
        },

        "Roll": {
            "ux": False,
            "uy": True,
            "rz": False,
        },

        "Jepit": {
            "ux": True,
            "uy": True,
            "rz": True,
        },
    }

    for support in supports:

        x = float(
            support["Position"]
        )

        closest_node = min(
            nodes,
            key=lambda n: abs(
                float(n["x"]) - x
            )
        )

        support_type = support.get(
            "Type",
            "Pin",
        )

        dof = support_dof.get(
            support_type,
            support_dof["Pin"],
        )

        result.append({

            "node": closest_node["id"],

            "ux": dof["ux"],
            "uy": dof["uy"],
            "rz": dof["rz"],

        })

    return result


def build_matrix_loads(nodes, loads):
    """
    Convert external loads to nodal loads.

    Point Load:
        directly assigned to node.

    Moment:
        directly assigned to node.

    UDL:
        converted into equivalent nodal forces
        using simple beam-element equivalent loading.
    """

    result = []

    # --------------------------------------------------------
    # Point Loads and Moments
    # --------------------------------------------------------

    for load in loads:

        load_type = load["type"]

        # ----------------------------------------------------
        # Point Load
        # ----------------------------------------------------

        if load_type == "Point Load":

            x = float(
                load["position"]
            )

            node = find_closest_node(
                nodes,
                x
            )

            result.append(
                {
                    "node": node["id"],
                    "fx": 0.0,
                    "fy": -abs(
                        float(load["magnitude"])
                    ),
                    "mz": 0.0,
                }
            )

        # ----------------------------------------------------
        # Moment
        # ----------------------------------------------------

        elif load_type == "Moment":

            x = float(
                load["position"]
            )

            node = find_closest_node(
                nodes,
                x
            )

            result.append(
                {
                    "node": node["id"],
                    "fx": 0.0,
                    "fy": 0.0,
                    "mz": float(
                        load["magnitude"]
                    ),
                }
            )

        # ----------------------------------------------------
        # UDL
        # ----------------------------------------------------

        elif load_type == "UDL":

            x1 = float(
                load["start"]
            )

            x2 = float(
                load["end"]
            )

            w = float(
                load["magnitude"]
            )

            if x2 <= x1:
                continue

            node_i = find_closest_node(
                nodes,
                x1
            )

            node_j = find_closest_node(
                nodes,
                x2
            )

            Li = float(node_j["x"]) - float(node_i["x"])

            if Li <= 0:
                continue

            # ------------------------------------------------
            # Equivalent nodal load for uniform load.
            #
            # Local convention:
            # downward load = negative Fy
            # ------------------------------------------------

            total_load = w * Li

            nodal_force = total_load / 2.0

            nodal_moment = (
                w * Li**2 / 12.0
            )

            result.append(
                {
                    "node": node_i["id"],
                    "fx": 0.0,
                    "fy": -nodal_force,
                    "mz": -nodal_moment,
                }
            )

            result.append(
                {
                    "node": node_j["id"],
                    "fx": 0.0,
                    "fy": -nodal_force,
                    "mz": nodal_moment,
                }
            )

    return result


def run_matrix_analysis(
    L,
    supports,
    loads,
    E_MPa,
    A,
    I,
):
    """
    Run Matrix Stiffness Analysis.
    """

    nodes, elements = build_matrix_model(
        L,
        supports,
        loads,
    )

    matrix_supports = build_matrix_supports(
        nodes,
        supports,
    )

    matrix_loads = build_matrix_loads(
        nodes,
        loads,
    )

    result = matrix_analyze(
        nodes=nodes,
        elements=elements,
        supports=matrix_supports,
        loads=matrix_loads,
        E_MPa=E_MPa,
        A=A,
        I=I,
    )

    result["nodes"] = nodes
    result["elements"] = elements

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_model(
    L,
    supports,
    cases,
    case_order,
):
    """Validate geometry, supports and loads."""

    errors = []

    # --------------------------------------------------------
    # Supports
    # --------------------------------------------------------

    if not supports:
        errors.append(
            "Minimal harus terdapat satu tumpuan."
        )

    for support in supports:

        x = float(
            support["Position"]
        )

        if not 0 <= x <= L:

            errors.append(
                f"Tumpuan {support['ID']} "
                f"berada di luar balok."
            )

    # --------------------------------------------------------
    # Load cases
    # --------------------------------------------------------

    for case_name in case_order:

        for load in cases.get(
            case_name,
            []
        ):

            load_type = load["type"]

            if load_type in [
                "Point Load",
                "Moment",
            ]:

                x = float(
                    load["position"]
                )

                if not 0 <= x <= L:

                    errors.append(
                        f"{case_name}/{load['name']}: "
                        f"posisi berada di luar balok."
                    )

            elif load_type == "UDL":

                x1 = float(
                    load["start"]
                )

                x2 = float(
                    load["end"]
                )

                if x2 <= x1:

                    errors.append(
                        f"{case_name}/{load['name']}: "
                        f"x₂ harus lebih besar dari x₁."
                    )

                if x1 < 0 or x2 > L:

                    errors.append(
                        f"{case_name}/{load['name']}: "
                        f"UDL berada di luar balok."
                    )

    return errors


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.title(
        "BALOK SEDERHANA"
    )

    st.caption(
        "Analisis Struktur Balok & "
        "Desain Tulangan"
    )



st.divider()


# ============================================================
# ANALYSIS ENGINE
# ============================================================

engine_col1, engine_col2 = st.columns(
    [1, 3]
)

with engine_col1:

    analysis_method = st.radio(
        "Analysis Method",
        [
            "Classical",
            "Matrix Stiffness",
        ],
        index=1,
        disabled=st.session_state.locked,
    )

with engine_col2:

    if analysis_method == "Matrix Stiffness":

        st.info(
            "Matrix Stiffness Method — "
            "2D Euler-Bernoulli beam "
            "dengan DOF UX, UY, RZ."
        )

    else:

        st.info(
            "Classical Beam Analysis — "
            "engine analisis V1.0."
        )


# ============================================================
# MAIN WORKSPACE
# ============================================================

model_col, view_col = st.columns(
    [0.95, 1.6],
    gap="large",
)


# ============================================================
# LEFT COLUMN — MODEL INPUT
# ============================================================

with model_col:

    st.subheader(
        "Model Balok"
    )

    if st.session_state.locked:

        st.success(
            "🔒 MODEL LOCKED — "
            "hasil analisis tersedia."
        )

    else:

        st.warning(
            "🔓 MODEL UNLOCKED — "
            "lakukan LOCK & ANALYZE untuk "
            "menjalankan analisis."
        )

    # ========================================================
    # GEOMETRY
    # ========================================================

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Geometry</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        L = c1.number_input(
            "Panjang L (m)",
            min_value=0.10,
            value=10.0,
            step=0.10,
            disabled=st.session_state.locked,
        )

        b = c2.number_input(
            "Lebar b (m)",
            min_value=0.05,
            value=0.25,
            step=0.05,
            disabled=st.session_state.locked,
        )

        h = c1.number_input(
            "Tinggi h (m)",
            min_value=0.05,
            value=0.40,
            step=0.05,
            disabled=st.session_state.locked,
        )

        A = b * h

        I = (
            b * h**3
            / 12
        )

        c1.metric(
            "Area",
            f"{A:.4f} m²",
        )

        c2.metric(
            "I",
            f"{I:.6f} m⁴",
        )

        # ========================================================
    # SUPPORTS
    # ========================================================

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Tumpuan</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Default supports
        # ----------------------------------------------------

        if "supports" not in st.session_state:

            st.session_state.supports = [
                {
                    "ID": "",
                    "Position": 0.0,
                    "Type": "Pin",
                },
                {
                    "ID": "",
                    "Position": float(L),
                    "Type": "Roll",
                },
            ]

        # ----------------------------------------------------
        # DataFrame
        # ----------------------------------------------------

        support_df = pd.DataFrame(
            st.session_state.supports
        )

        # ----------------------------------------------------
        # Data Editor
        # ----------------------------------------------------

        edited_supports = st.data_editor(

            support_df,

            num_rows="dynamic",

            use_container_width=True,

            disabled=st.session_state.locked,

            column_config={

                "ID": st.column_config.TextColumn(
                    "ID",
                    width="small",
                    required=True,
                ),

                "Position": st.column_config.NumberColumn(
                    "Posisi x (m)",
                    min_value=0.0,
                    max_value=float(L),
                    step=0.10,
                    format="%.2f",
                ),

                "Type": st.column_config.SelectboxColumn(
                    "Tipe Tumpuan",
                    options=[
                        "Pin",
                        "Roll",
                        "Jepit",
                    ],
                    required=True,
                ),
            },

            key="support_editor",
        )

        # ----------------------------------------------------
        # Simpan perubahan
        # ----------------------------------------------------

        if not st.session_state.locked:

            new_supports = (
                edited_supports
                .fillna("")
                .to_dict("records")
            )

            if new_supports != st.session_state.supports:

                st.session_state.supports = new_supports

    # ========================================================
    # MATERIAL
    # ========================================================

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Material</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        fc = c1.number_input(
            "f'c (MPa)",
            min_value=10.0,
            value=25.0,
            step=1.0,
            disabled=st.session_state.locked,
        )

        E_MPa = 4700.0 * math.sqrt(fc)

        st.metric(
            "Modulus Elastisitas Ec (MPa)",
            f"{E_MPa:,.0f}"
        )

        fy = c2.number_input(
            "fy Longitudinal (MPa)",
            min_value=200.0,
            value=420.0,
            step=10.0,
            disabled=st.session_state.locked,
        )

    # ========================================================
    # REINFORCEMENT
    # ========================================================

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">'
            'Potongan Melintang & Penulangan'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Penampang persegi • "
            "tulangan nonprategang"
        )

        c1, c2 = st.columns(2)

        cover = c1.number_input(
            "Selimut Beton (mm)",
            min_value=10.0,
            value=40.0,
            step=5.0,
            disabled=st.session_state.locked,
        )

        stirrup_dia = c2.number_input(
            "Ø Sengkang (mm)",
            min_value=6.0,
            value=10.0,
            step=2.0,
            disabled=st.session_state.locked,
        )

        c1, c2 = st.columns(2)

        bar_dia = c1.number_input(
            "Ø Tulangan longitudinal (mm)",
            min_value=6.0,
            value=16.0,
            step=2.0,
            disabled=st.session_state.locked,
        )

        n_bottom = c2.number_input(
            "Jumlah tulangan bawah",
            min_value=1,
            value=4,
            step=1,
            disabled=st.session_state.locked,
        )

        c1, c2 = st.columns(2)

        n_top = c1.number_input(
            "Jumlah tulangan atas",
            min_value=0,
            value=2,
            step=1,
            disabled=st.session_state.locked,
        )

        fyv = c2.number_input(
            "fy sengkang (MPa)",
            min_value=200.0,
            value=280.0,
            step=10.0,
            disabled=st.session_state.locked,
        )

        c1, c2 = st.columns(2)

        stirrup_legs = c1.number_input(
            "Jumlah kaki sengkang",
            min_value=2,
            value=2,
            step=1,
            disabled=st.session_state.locked,
        )

        stirrup_spacing = c2.number_input(
            "Spacing sengkang s (mm)",
            min_value=50.0,
            value=150.0,
            step=25.0,
            disabled=st.session_state.locked,
        )

    # ========================================================
    # LOAD CASE
    # ========================================================

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Load Case</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(
            [2, 1]
        )

        active_case = c1.selectbox(
            "Active Case",
            st.session_state.case_order,
            index=st.session_state.case_order.index(
                st.session_state.active_case
            ),
            disabled=st.session_state.locked,
        )

        st.session_state.active_case = (
            active_case
        )

        if not st.session_state.locked:

            if c2.button(
                "➕ New Case",
                use_container_width=True,
            ):

                new_name = (
                    f"LC{len(st.session_state.case_order) + 1}"
                )

                st.session_state.cases[
                    new_name
                ] = []

                st.session_state.case_order.append(
                    new_name
                )

                st.session_state.active_case = (
                    new_name
                )

                rerun()

        if (
            not st.session_state.locked
            and len(st.session_state.case_order) > 1
        ):

            if st.button(
                "🗑️ Delete Active Case",
                use_container_width=True,
            ):

                st.session_state.cases.pop(
                    active_case,
                    None,
                )

                st.session_state.case_order.remove(
                    active_case
                )

                st.session_state.active_case = (
                    st.session_state.case_order[0]
                )

                rerun()

    # ========================================================
    # LOADS
    # ========================================================

    with st.container(border=True):

        st.markdown(
            f'<div class="section-title">'
            f'Loads — {active_case}'
            f'</div>',
            unsafe_allow_html=True,
        )

        case_loads = (
            st.session_state.cases[
                active_case
            ]
        )

        if not st.session_state.locked:

            if st.button(
                "➕ Add Load",
                use_container_width=True,
            ):

                case_loads.append(
                    {
                        "name": (
                            f"Load {len(case_loads) + 1}"
                        ),
                        "type": "Point Load",
                        "magnitude": 20.0,
                        "position": min(
                            5.0,
                            float(L),
                        ),
                    }
                )

                rerun()

        if not case_loads:

            st.info(
                "Belum ada beban. "
                "Klik Add Load."
            )

        for idx, load in enumerate(
            case_loads
        ):

            with st.container(
                border=True
            ):

                c1, c2, c3 = st.columns(
                    [1.1, 1.1, 0.45]
                )

                if not st.session_state.locked:

                    load["name"] = c1.text_input(
                        "Nama",
                        value=load["name"],
                        key=(
                            f"name_{active_case}_{idx}"
                        ),
                    )

                    type_options = [
                        "Point Load",
                        "UDL",
                        "Moment",
                    ]

                    load["type"] = c2.selectbox(
                        "Tipe",
                        type_options,
                        index=type_options.index(
                            load["type"]
                        ),
                        key=(
                            f"type_{active_case}_{idx}"
                        ),
                    )

                    if c3.button(
                        "🗑️",
                        key=(
                            f"delete_load_"
                            f"{active_case}_{idx}"
                        ),
                    ):

                        case_loads.pop(idx)

                        rerun()

                else:

                    c1.write(
                        f"**{load['name']}**"
                    )

                    c2.write(
                        load["type"]
                    )

                # ------------------------------------------------
                # Point Load
                # ------------------------------------------------

                if load["type"] == "Point Load":

                    c1, c2 = st.columns(2)

                    load["magnitude"] = (
                        c1.number_input(
                            "P (kN) — + = ke bawah",
                            value=float(
                                load["magnitude"]
                            ),
                            step=1.0,
                            key=(
                                f"P_"
                                f"{active_case}_{idx}"
                            ),
                            disabled=(
                                st.session_state.locked
                            ),
                        )
                    )

                    load["position"] = (
                        c2.number_input(
                            "Posisi x (m)",
                            min_value=0.0,
                            max_value=float(L),
                            value=min(
                                float(
                                    load["position"]
                                ),
                                float(L),
                            ),
                            step=0.5,
                            key=(
                                f"x_"
                                f"{active_case}_{idx}"
                            ),
                            disabled=(
                                st.session_state.locked
                            ),
                        )
                    )

                # ------------------------------------------------
                # UDL
                # ------------------------------------------------

                elif load["type"] == "UDL":

                    c1, c2, c3 = st.columns(3)

                    load["magnitude"] = (
                        c1.number_input(
                            "w (kN/m)",
                            value=float(
                                load["magnitude"]
                            ),
                            step=1.0,
                            key=(
                                f"w_"
                                f"{active_case}_{idx}"
                            ),
                            disabled=(
                                st.session_state.locked
                            ),
                        )
                    )

                    load["start"] = (
                        c2.number_input(
                            "x₁ (m)",
                            min_value=0.0,
                            max_value=float(L),
                            value=min(
                                float(
                                    load.get(
                                        "start",
                                        0.0
                                    )
                                ),
                                float(L),
                            ),
                            step=0.5,
                            key=(
                                f"x1_"
                                f"{active_case}_{idx}"
                            ),
                            disabled=(
                                st.session_state.locked
                            ),
                        )
                    )

                    load["end"] = (
                        c3.number_input(
                            "x₂ (m)",
                            min_value=0.0,
                            max_value=float(L),
                            value=min(
                                float(
                                    load.get(
                                        "end",
                                        L
                                    )
                                ),
                                float(L),
                            ),
                            step=0.5,
                            key=(
                                f"x2_"
                                f"{active_case}_{idx}"
                            ),
                            disabled=(
                                st.session_state.locked
                            ),
                        )
                    )

                # ------------------------------------------------
                # Moment
                # ------------------------------------------------

                elif load["type"] == "Moment":

                    c1, c2 = st.columns(2)

                    load["magnitude"] = (
                        c1.number_input(
                            "M (kNm) — + = clockwise",
                            value=float(
                                load["magnitude"]
                            ),
                            step=1.0,
                            key=(
                                f"M_"
                                f"{active_case}_{idx}"
                            ),
                            disabled=(
                                st.session_state.locked
                            ),
                        )
                    )

                    load["position"] = (
                        c2.number_input(
                            "Posisi x (m)",
                            min_value=0.0,
                            max_value=float(L),
                            value=min(
                                float(
                                    load["position"]
                                ),
                                float(L),
                            ),
                            step=0.5,
                            key=(
                                f"Mx_"
                                f"{active_case}_{idx}"
                            ),
                            disabled=(
                                st.session_state.locked
                            ),
                        )
                    )

    # ========================================================
    # ACTION BUTTON
    # ========================================================

    if not st.session_state.locked:

        if st.button(
            "🔒 LOCK & ANALYZE",
            type="primary",
            use_container_width=True,
        ):

            errors = validate_model(
                L=L,
                supports=st.session_state.supports,
                cases=st.session_state.cases,
                case_order=st.session_state.case_order,
            )

            if errors:

                for error in errors:
                    st.error(error)

            else:

                combo_results = {}
                matrix_results = {}

                # ------------------------------------------------
                # Analyze every combination
                # ------------------------------------------------

                for combo in (
                    st.session_state.combinations
                ):

                    combo_name = combo["name"]

                    loads = (
                        build_combination_loads(
                            combo
                        )
                    )

                    # ============================================
                    # CLASSICAL ANALYSIS
                    # ============================================

                    combo_results[
                        combo_name
                    ] = solve_beam(
                        L=L,
                        b=b,
                        h=h,
                        E_MPa=E_MPa,
                        loads=loads,
                    )

                    # ============================================
                    # MATRIX STIFFNESS
                    # ============================================

                    if (
                        analysis_method
                        == "Matrix Stiffness"
                    ):

                        try:

                            matrix_results[
                                combo_name
                            ] = run_matrix_analysis(
                                L=L,
                                supports=(
                                    st.session_state.supports
                                ),
                                loads=loads,
                                E_MPa=E_MPa,
                                A=A,
                                I=I,
                            )

                        except Exception as e:

                            matrix_results[
                                combo_name
                            ] = {
                                "error": str(e)
                            }

                # ------------------------------------------------
                # Save results
                # ------------------------------------------------

                st.session_state.results = {
                    "combos": combo_results,
                    "L": L,
                    "b": b,
                    "h": h,
                    "E_MPa": E_MPa,
                    "fc": fc,
                    "fy": fy,
                }

                st.session_state.matrix_results = (
                    matrix_results
                )

                st.session_state.rc_design = None

                st.session_state.locked = True

                rerun()

    else:

        if st.button(
            "🔓 UNLOCK MODEL",
            use_container_width=True,
        ):

            st.session_state.locked = False

            st.session_state.results = None

            st.session_state.matrix_results = {}

            st.session_state.rc_design = None

            rerun()


# ============================================================
# RIGHT COLUMN — MODEL VIEW
# ============================================================

with view_col:

    st.subheader("Model Struktur")

    # ========================================================
    # VISUAL SETTINGS
    # ========================================================

    LOAD_COLOR = "#C62828"
    LOAD_FILL = "rgba(198, 40, 40, 0.15)"

    BEAM_COLOR = "#00CCFF"
    SUPPORT_COLOR = "#1565C0"
    DIM_COLOR = "#F70000"

    # ========================================================
    # FIGURE
    # ========================================================

    fig = go.Figure()

    # ========================================================
    # VISUAL SCALE
    # ========================================================

    beam_y = 0.0

    # Tinggi visual panah beban titik
    point_arrow_y = 1.00

    # Tinggi visual UDL
    udl_arrow_y = 0.5

    # Posisi garis dimensi
    dim_y = -0.5

    # ========================================================
    # BEAM
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=[0, L],
            y=[beam_y, beam_y],
            mode="lines",
            line=dict(
                width=8,
                color=BEAM_COLOR,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # ========================================================
    # SUPPORTS
    # ========================================================

    supports = st.session_state.supports

    for support in supports:

        x = float(
            support["Position"]
        )

        support_type = support.get(
            "Type",
            "Pin",
        )


        # --------------------------------------------------------
        # PIN
        # --------------------------------------------------------

        if support_type == "Pin":

                    fig.add_trace(
                        go.Scatter(
                            x=[x],
                            y=[-0.1],
                            mode="markers",
                            marker=dict(
                                symbol="triangle-up",
                                size=20,
                            ),
                            showlegend=False,
                        )
                    )

        # --------------------------------------------------------
        # ROLL
        # --------------------------------------------------------

        elif support_type == "Roll":

                    fig.add_trace(
                        go.Scatter(
                            x=[x],
                            y=[-0.1],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                size=15,
                            ),
                            showlegend=False,
                        )
                    )

        # --------------------------------------------------------
        # JEPIT
        # --------------------------------------------------------

        elif support_type == "Jepit":

                    fig.add_shape(
                        type="rect",

                        x0=x - 0.1,
                        x1=x + 0.1,

                        y0=-0.1,
                        y1=+0.1,

                        line=dict(
                            width=1,
                        ),

                        fillcolor="#1565C0",
                    )

                    # Garis hatch jepit
                    hatch_x = np.linspace(
                        x - 0.05 * L,
                        x + 0.05 * L,
                        6,
                    )

                    for hx in hatch_x:

                        fig.add_shape(
                            type="line",

                            x0=hx,
                            y0=-0.01,

                            x1=hx - 0.025 * L,
                            y1=-0.22,

                            line=dict(
                                width=0,
                            ),
                        )

        # --------------------------------------------------------
        # SUPPORT LABEL
        # --------------------------------------------------------

        fig.add_annotation(
            x=x,
            y=-0.2,

            text=(
                f"<b>{support['ID']}</b>"
                f"<br>{support_type}"
            ),

            showarrow=False,
        )

        # ----------------------------------------------------
        # Support ID
        # ----------------------------------------------------

        fig.add_annotation(
            x=x,
            y=-0.38,
            text=f"<b>{support['ID']}</b>",
            showarrow=False,
            font=dict(
                size=13,
                color=SUPPORT_COLOR,
            ),
        )

    # ========================================================
    # DIMENSION LINE
    # ========================================================

    # --------------------------------------------------------
    # Main dimension line
    # --------------------------------------------------------

    fig.add_shape(
        type="line",
        x0=0,
        x1=L,
        y0=dim_y,
        y1=dim_y,
        line=dict(
            width=2,
            color=DIM_COLOR,
        ),
    )

    # ========================================================
    # DIMENSION EXTENSION OFFSET
    # ========================================================

    dim_offset = 0.30

    # Left extension line
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=-dim_offset,
        y1=dim_y,
        line=dict(
            width=1,
            color=DIM_COLOR,
        ),
    )

    # Right extension line
    fig.add_shape(
        type="line",
        x0=L,
        x1=L,
        y0=-dim_offset,
        y1=dim_y,
        line=dict(
            width=1,
            color=DIM_COLOR,
        ),
    )

    # --------------------------------------------------------
    # Dimension arrow — left
    # --------------------------------------------------------

    fig.add_annotation(
        x=0,
        y=dim_y,
        ax=0.30 * L,
        ay=dim_y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=DIM_COLOR,
    )

    # --------------------------------------------------------
    # Dimension arrow — right
    # --------------------------------------------------------

    fig.add_annotation(
        x=L,
        y=dim_y,
        ax=0.70 * L,
        ay=dim_y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=DIM_COLOR,
    )

    # --------------------------------------------------------
    # Dimension value
    # --------------------------------------------------------

    fig.add_annotation(
        x=L / 2,
        y=dim_y + 0.12,
        text=f"<b>L = {L:.2f} m</b>",
        showarrow=False,
        font=dict(
            size=14,
            color=DIM_COLOR,
        ),
    )

    # ========================================================
    # LOADS
    # ========================================================

    for case_name in st.session_state.case_order:

        for load in st.session_state.cases.get(
            case_name,
            []
        ):

            load_type = load["type"]

            # =================================================
            # POINT LOAD
            # =================================================

            if load_type == "Point Load":

                x_load = float(
                    load["position"]
                )

                magnitude = float(
                    load["magnitude"]
                )

                # ------------------------------------------------
                # Direction
                # ------------------------------------------------

                if magnitude >= 0:

                    # Positive = downward
                    arrow_y = point_arrow_y
                    label_y = 1.30

                else:

                    # Negative = upward
                    arrow_y = -point_arrow_y
                    label_y = -1.30

                # ------------------------------------------------
                # Load arrow
                # ------------------------------------------------

                fig.add_annotation(
                    x=x_load,
                    y=beam_y,
                    ax=x_load,
                    ay=arrow_y,

                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",

                    text="",

                    showarrow=True,

                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=2.5,
                    arrowcolor=LOAD_COLOR,
                )

                # ------------------------------------------------
                # Load label
                # ------------------------------------------------

                fig.add_annotation(
                    x=x_load,
                    y=label_y,

                    text=(
                        f"<b>{case_name}</b><br>"
                        f"{load['name']}<br>"
                        f"<b>{magnitude:.2f} kN</b>"
                    ),

                    showarrow=False,

                    font=dict(
                        size=12,
                        color=LOAD_COLOR,
                    ),
                )

            # =================================================
            # UDL
            # =================================================

            elif load_type == "UDL":

                start = float(
                    load["start"]
                )

                end = float(
                    load["end"]
                )

                magnitude = float(
                    load["magnitude"]
                )

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                if end <= start:

                    continue

                # ------------------------------------------------
                # Direction
                # ------------------------------------------------

                if magnitude >= 0:

                    # Positive = downward
                    fill_y = udl_arrow_y
                    label_y = 1.25

                else:

                    # Negative = upward
                    fill_y = -udl_arrow_y
                    label_y = -1.25

                # =================================================
                # UDL FILL
                # =================================================

                fig.add_trace(
                    go.Scatter(
                        x=[
                            start,
                            end,
                            end,
                            start,
                            start,
                        ],

                        y=[
                            beam_y,
                            beam_y,
                            fill_y,
                            fill_y,
                            beam_y,
                        ],

                        mode="lines",

                        fill="toself",

                        fillcolor=LOAD_FILL,

                        line=dict(
                            width=1,
                            color=LOAD_COLOR,
                        ),

                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                # =================================================
                # UDL TOP LINE
                # =================================================

                fig.add_trace(
                    go.Scatter(
                        x=[
                            start,
                            end,
                        ],

                        y=[
                            fill_y,
                            fill_y,
                        ],

                        mode="lines",

                        line=dict(
                            width=2,
                            color=LOAD_COLOR,
                        ),

                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                # =================================================
                # UDL ARROWS
                # =================================================

                xs = np.linspace(
                    start,
                    end,
                    17,
                )

                for x_load in xs:

                    fig.add_annotation(
                        x=x_load,
                        y=beam_y,

                        ax=x_load,
                        ay=fill_y,

                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",

                        text="",

                        showarrow=True,

                        arrowhead=2,
                        arrowsize=0.7,
                        arrowwidth=1.4,
                        arrowcolor=LOAD_COLOR,
                    )

                # =================================================
                # UDL LABEL
                # =================================================

                fig.add_annotation(
                    x=(start + end) / 2,
                    y=label_y,

                    text=(
                        f"<b>{case_name}</b><br>"
                        f"{load['name']}<br>"
                        f"<b>{magnitude:.2f} kN/m</b>"
                    ),

                    showarrow=False,

                    font=dict(
                        size=12,
                        color=LOAD_COLOR,
                    ),
                )

            # =================================================
            # MOMENT
            # =================================================

            elif load_type == "Moment":

                x_load = float(
                    load["position"]
                )

                magnitude = float(
                    load["magnitude"]
                )

                # ------------------------------------------------
                # Moment symbol
                # ------------------------------------------------

                symbol = (
                    "↻"
                    if magnitude >= 0
                    else "↺"
                )

                # ------------------------------------------------
                # Moment label
                # ------------------------------------------------

                fig.add_annotation(
                    x=x_load,
                    y=0.65,

                    text=(
                        f"<b>{symbol} "
                        f"{magnitude:.2f} kNm</b><br>"
                        f"{case_name}: "
                        f"{load['name']}"
                    ),

                    showarrow=False,

                    font=dict(
                        size=12,
                        color=LOAD_COLOR,
                    ),
                )

            # =================================================
            # END LOAD TYPE
            # =================================================

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(

        height=550,

        margin=dict(
            l=30,
            r=30,
            t=30,
            b=30,
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        # ----------------------------------------------------
        # X AXIS
        # ----------------------------------------------------

        xaxis=dict(
            visible=False,

            range=[
                -0.10 * L,
                1.10 * L,
            ],

            fixedrange=True,
        ),

        # ----------------------------------------------------
        # Y AXIS
        # ----------------------------------------------------

        yaxis=dict(
            visible=False,

            range=[
                -1.60,
                1.60,
            ],

            fixedrange=True,
        ),

        # ----------------------------------------------------
        # General
        # ----------------------------------------------------

        showlegend=False,


    )

    # ========================================================
    # DISPLAY
    # ========================================================

    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar": False,
        },
    )

    # ============================================================
    # CROSS SECTION PREVIEW
    # ============================================================

    st.markdown("#### Penampang Balok")

    sec = go.Figure()

    # ------------------------------------------------------------
    # Dimension in mm
    # ------------------------------------------------------------

    bm = b * 1000
    hm = h * 1000

    # ------------------------------------------------------------
    # Concrete section
    # ------------------------------------------------------------

    sec.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=bm,
        y1=hm,
        line=dict(width=2),
        fillcolor="rgba(180,180,180,0.18)",
    )

    # ------------------------------------------------------------
    # Stirrup
    # ------------------------------------------------------------

    sx0 = cover + stirrup_dia / 2
    sy0 = cover + stirrup_dia / 2

    sx1 = bm - cover - stirrup_dia / 2
    sy1 = hm - cover - stirrup_dia / 2

    if sx1 > sx0 and sy1 > sy0:

        sec.add_shape(
            type="rect",
            x0=sx0,
            y0=sy0,
            x1=sx1,
            y1=sy1,
            line=dict(
                width=2,
            ),
            fillcolor="rgba(0,0,0,0)",
        )


    # ------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------

    def add_bars_mm(n, y, label):

        if n <= 0:
            return

        xl = (
            cover
            + stirrup_dia
            + bar_dia / 2
        )

        xr = (
            bm
            - cover
            - stirrup_dia
            - bar_dia / 2
        )

        if n == 1:

            xs = [bm / 2]

        elif xr > xl:

            xs = np.linspace(
                xl,
                xr,
                int(n),
            )

        else:

            xs = np.linspace(
                max(bar_dia / 2, bm * 0.2),
                min(bm - bar_dia / 2, bm * 0.8),
                int(n),
            )

        sec.add_trace(
            go.Scatter(
                x=xs,
                y=[y] * len(xs),
                mode="markers",
                marker=dict(
                    size=max(
                        7,
                        min(18, bar_dia / 1.1),
                    ),
                    symbol="circle",
                ),
                name=label,
                hovertemplate=(
                    f"{label} "
                    f"Ø{bar_dia:g} mm"
                    "<extra></extra>"
                ),
            )
        )


    # Bottom reinforcement
    add_bars_mm(
        n_bottom,
        cover + stirrup_dia + bar_dia / 2,
        "Tulangan bawah",
    )

    # Top reinforcement
    add_bars_mm(
        n_top,
        hm - cover - stirrup_dia - bar_dia / 2,
        "Tulangan atas",
    )


    # ============================================================
    # DIMENSION — WIDTH b
    # ============================================================

    dim_y = -55

    # Main dimension line
    sec.add_shape(
        type="line",
        x0=0,
        y0=dim_y,
        x1=bm,
        y1=dim_y,
        line=dict(
            width=1.5,
        ),
    )

    # Extension line kiri
    sec.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=0,
        y1=dim_y - 5,
        line=dict(
            width=1,
        ),
    )

    # Extension line kanan
    sec.add_shape(
        type="line",
        x0=bm,
        y0=0,
        x1=bm,
        y1=dim_y - 5,
        line=dict(
            width=1,
        ),
    )

    # Arrow kiri
    sec.add_annotation(
        x=0,
        y=dim_y,
        ax=12,
        ay=dim_y,
        text="",
        showarrow=False,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
    )

    # Arrow kanan
    sec.add_annotation(
        x=bm,
        y=dim_y,
        ax=bm - 12,
        ay=dim_y,
        text="",
        showarrow=False,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
    )

    # Label b
    sec.add_annotation(
        x=bm / 2,
        y=dim_y - 22,
        text=f"<b>b = {bm:.0f} mm</b>",
        showarrow=False,
    )


    # ============================================================
    # DIMENSION — HEIGHT h
    # ============================================================

    dim_x = -55

    # Main dimension line
    sec.add_shape(
        type="line",
        x0=dim_x,
        y0=0,
        x1=dim_x,
        y1=hm,
        line=dict(
            width=1.5,
        ),
    )

    # Extension line bawah
    sec.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=dim_x - 5,
        y1=0,
        line=dict(
            width=1,
        ),
    )

    # Extension line atas
    sec.add_shape(
        type="line",
        x0=0,
        y0=hm,
        x1=dim_x - 5,
        y1=hm,
        line=dict(
            width=1,
        ),
    )

    # Arrow bawah
    sec.add_annotation(
        x=dim_x,
        y=0,
        ax=dim_x,
        ay=12,
        text="",
        showarrow=False,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
    )

    # Arrow atas
    sec.add_annotation(
        x=dim_x,
        y=hm,
        ax=dim_x,
        ay=hm - 12,
        text="",
        showarrow=False,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
    )

    # Label h
    sec.add_annotation(
        x=dim_x - 25,
        y=hm / 2,
        text=f"<b>h = {hm:.0f} mm</b>",
        textangle=-90,
        showarrow=False,
    )


    # ============================================================
    # LAYOUT
    # ============================================================

    # Beri ruang untuk dimensi
    x_margin = max(80, bm * 0.15)
    y_margin = max(100, hm * 0.15)

    sec.update_layout(

        height=430,

        margin=dict(
            l=70,
            r=30,
            t=20,
            b=70,
        ),

        # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        # Hilangkan axis sepenuhnya
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            ticks="",
            fixedrange=True,
            range=[
                -x_margin,
                bm + x_margin,
            ],
        ),

        yaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            ticks="",
            fixedrange=True,
            range=[
                -y_margin,
                hm + y_margin,
            ],
            scaleanchor="x",
            scaleratio=1,
        ),

        showlegend=False,

        # Nonaktifkan interaksi
        dragmode=False,

        hovermode=False,
    )


    # ============================================================
    # DISPLAY
    # ============================================================

    st.plotly_chart(
        sec,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "showTips": False,
        },
    )

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "L",
        f"{L:.2f} m",
    )

    c2.metric(
        "b × h",
        f"{b:.2f} × {h:.2f} m",
    )

    c3.metric(
        "E",
        f"{E_MPa:,.0f} MPa",
    )

    st.info(
        "Konvensi: beban vertikal positif = "
        "ke bawah. Momen positif = clockwise. "
        "Input di kiri langsung memengaruhi "
        "model di kanan."
    )


# ============================================================
# ANALYSIS RESULT
# ============================================================

st.divider()

st.header(
    "📊 Analysis"
)

if st.session_state.results:

    combo_names = list(
        st.session_state.results[
            "combos"
        ].keys()
    )

    display_combo = st.selectbox(
        "Combination",
        combo_names,
        key="display_combo",
    )

    result = (
        st.session_state.results[
            "combos"
        ][display_combo]
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "RA",
        f"{result['RA']:.2f} kN",
    )

    c2.metric(
        "RB",
        f"{result['RB']:.2f} kN",
    )

    c3.metric(
        "Vmax",
        f"{result['max_shear']:.2f} kN",
    )

    c4.metric(
        "Mmax",
        f"{result['max_moment']:.2f} kNm",
    )

    c5.metric(
        "δmax",
        f"{result['max_deflection'] * 1000:.3f} mm",
    )

    # --------------------------------------------------------
    # Diagrams
    # --------------------------------------------------------

    tab_v, tab_m, tab_d = st.tabs(
        [
            "Shear Force Diagram",
            "Bending Moment Diagram",
            "Deflection",
        ]
    )

    with tab_v:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=result["x"],
                y=result["V"],
                mode="lines",
                fill="tozeroy",
                name="V",
            )
        )

        fig.add_hline(
            y=0
        )

        fig.update_layout(
            height=430,
            xaxis_title="x (m)",
            yaxis_title="V (kN)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                        "displayModeBar": False,
                    },
        )

    with tab_m:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=result["x"],
                y=result["M"],
                mode="lines",
                fill="tozeroy",
                name="M",
            )
        )

        fig.add_hline(
            y=0
        )

        fig.update_layout(
            height=430,
            xaxis_title="x (m)",
            yaxis_title="M (kNm)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                        "displayModeBar": False,
                    },
        )

    with tab_d:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=result["x"],
                y=result["deflection"] * 1000,
                mode="lines",
                fill="tozeroy",
                name="Deflection",
            )
        )

        fig.add_hline(
            y=0
        )

        fig.update_layout(
            height=430,
            xaxis_title="x (m)",
            yaxis_title="Deflection (mm)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                        "displayModeBar": False,
                    },
        )

    st.caption(
        f"Mmax = {result['max_moment']:.3f} kNm "
        f"at x = {result['max_moment_x']:.3f} m | "
        f"δmax = "
        f"{result['max_deflection'] * 1000:.3f} mm "
        f"at x = "
        f"{result['max_deflection_x']:.3f} m"
    )

else:

    st.info(
        "Belum ada hasil analisis. "
        "Isi model dan beban, kemudian "
        "klik LOCK & ANALYZE."
    )


# ============================================================
# MATRIX STIFFNESS RESULT
# ============================================================

if (
    st.session_state.matrix_results
    and analysis_method == "Matrix Stiffness"
):

    st.divider()

    st.header(
        "🔢 Matrix Stiffness Analysis"
    )

    matrix_combo_names = list(
        st.session_state.matrix_results.keys()
    )

    matrix_combo = st.selectbox(
        "Matrix Combination",
        matrix_combo_names,
        key="matrix_display_combo",
    )

    matrix_result = (
        st.session_state.matrix_results[
            matrix_combo
        ]
    )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    if "error" in matrix_result:

        st.error(
            "Matrix Stiffness Analysis gagal: "
            + matrix_result["error"]
        )

    else:

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Nodes",
                "Reactions",
                "Displacement",
                "Matrices",
            ]
        )

        # ----------------------------------------------------
        # Nodes
        # ----------------------------------------------------

        with tab1:

            st.subheader(
                "Node & Element"
            )

            node_data = []

            nodes = matrix_result.get(
                "nodes",
                []
            )

            for node in nodes:

                node_id = node["id"]

                node_data.append(
                    {
                        "Node": node_id,
                        "x (m)": node["x"],
                        "UX DOF": node_id * 3,
                        "UY DOF": node_id * 3 + 1,
                        "RZ DOF": node_id * 3 + 2,
                    }
                )

            st.dataframe(
                pd.DataFrame(node_data),
                hide_index=True,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Reactions
        # ----------------------------------------------------

        with tab2:

            st.subheader(
                "Reactions"
            )

            try:

                reactions = get_reactions(
                    matrix_result.get(
                        "nodes",
                        []
                    ),
                    matrix_result["R"],
                )

                st.dataframe(
                    pd.DataFrame(
                        reactions
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            except Exception as e:

                st.error(
                    f"Reaction extraction gagal: {e}"
                )

        # ----------------------------------------------------
        # Displacement
        # ----------------------------------------------------

        with tab3:

            st.subheader(
                "Nodal Displacement"
            )

            displacement = []

            U = matrix_result["U"]

            n_nodes = (
                len(U) // 3
            )

            for i in range(
                n_nodes
            ):

                displacement.append(
                    {
                        "Node": i,
                        "UX": U[i * 3],
                        "UY": U[i * 3 + 1],
                        "RZ": U[i * 3 + 2],
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    displacement
                ),
                hide_index=True,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Matrices
        # ----------------------------------------------------

        with tab4:

            st.subheader(
                "Global Stiffness Matrix [K]"
            )

            K = matrix_result["K"]

            st.dataframe(
                pd.DataFrame(
                    K,
                    index=[
                        f"DOF {i}"
                        for i in range(
                            len(K)
                        )
                    ],
                    columns=[
                        f"DOF {i}"
                        for i in range(
                            len(K)
                        )
                    ],
                ),
                use_container_width=True,
            )

            st.subheader(
                "Global Load Vector [F]"
            )

            F = matrix_result["F"]

            st.dataframe(
                pd.DataFrame(
                    {
                        "DOF": range(
                            len(F)
                        ),
                        "F": F,
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.subheader(
                "Displacement Vector [U]"
            )

            U = matrix_result["U"]

            st.dataframe(
                pd.DataFrame(
                    {
                        "DOF": range(
                            len(U)
                        ),
                        "U": U,
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            if (
                "condition_number"
                in matrix_result
            ):

                st.metric(
                    "Condition Number",
                    f"{matrix_result['condition_number']:.3e}",
                )


# ============================================================
# LOAD COMBINATION
# ============================================================

st.divider()

st.header(
    "🧮 Load Combination"
)

for i, combo in enumerate(
    st.session_state.combinations
):

    with st.container(
        border=True
    ):

        c1, c2 = st.columns(
            [1, 3]
        )

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        if not st.session_state.locked:

            combo["name"] = (
                c1.text_input(
                    "Nama",
                    value=combo["name"],
                    key=f"combo_name_{i}",
                )
            )

        else:

            c1.write(
                f"**{combo['name']}**"
            )

        # ----------------------------------------------------
        # Formula
        # ----------------------------------------------------

        c2.write(
            combo_text(combo)
        )

        # ----------------------------------------------------
        # Factors
        # ----------------------------------------------------

        factor_cols = st.columns(
            max(
                1,
                len(
                    st.session_state.case_order
                ),
            )
        )

        for j, case_name in enumerate(
            st.session_state.case_order
        ):

            # Make sure factor exists
            if case_name not in combo[
                "factors"
            ]:

                combo["factors"][
                    case_name
                ] = 0.0

            combo["factors"][
                case_name
            ] = factor_cols[j].number_input(
                case_name,
                value=float(
                    combo["factors"].get(
                        case_name,
                        0.0,
                    )
                ),
                step=0.1,
                key=(
                    f"factor_combo_"
                    f"{i}_{j}_{case_name}"
                ),
                disabled=(
                    st.session_state.locked
                ),
            )


if not st.session_state.locked:

    if st.button(
        "➕ Add Load Combination"
    ):

        st.session_state.combinations.append(
            {
                "name": (
                    f"COMB "
                    f"{len(st.session_state.combinations) + 1}"
                ),
                "factors": {
                    case_name: 0.0
                    for case_name in (
                        st.session_state.case_order
                    )
                },
            }
        )

        rerun()


# ============================================================
# RC DESIGN — SNI 2847:2019
# ============================================================

st.divider()

st.header(
    "🔩 Desain Balok Beton Bertulang — "
    "SNI 2847:2019"
)

st.caption(
    "Scope V1.1: balok persegi nonprategang, "
    "beton normal, tanpa gaya aksial/torsi, "
    "dengan desain lentur dan geser dasar. "
    "Detail seismik, torsi, T-beam, "
    "development/lap splice lengkap, dan "
    "serviceability lanjutan akan ditambahkan "
    "bertahap."
)


if st.session_state.results:

    # --------------------------------------------------------
    # Find ULS combinations
    # --------------------------------------------------------

    uls_names = [
        name
        for name in (
            st.session_state.results[
                "combos"
            ]
        )
        if (
            "ULS" in name.upper()
            or "ULT" in name.upper()
        )
    ]

    if not uls_names:

        uls_names = list(
            st.session_state.results[
                "combos"
            ].keys()
        )

    selected_uls = st.multiselect(
        "Combination desain",
        uls_names,
        default=uls_names[:1],
    )

    if selected_uls:

        Mu = max(
            st.session_state.results[
                "combos"
            ][name]["max_moment"]
            for name in selected_uls
        )

        Vu = max(
            st.session_state.results[
                "combos"
            ][name]["max_shear"]
            for name in selected_uls
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Mu Envelope",
            f"{Mu:.2f} kNm",
        )

        c2.metric(
            "Vu Envelope",
            f"{Vu:.2f} kN",
        )

        # ----------------------------------------------------
        # Run Design
        # ----------------------------------------------------

        if st.button(
            "🔩 RUN SNI 2847:2019 DESIGN",
            type="primary",
        ):

            try:

                st.session_state.rc_design = (
                    design_rc_beam_sni(
                        b=b * 1000,
                        h=h * 1000,
                        fc=fc,
                        fy=fy,
                        fyv=fyv,
                        cover=cover,
                        Mu=Mu,
                        Vu=Vu,
                        bar_dia=bar_dia,
                        n_bottom=n_bottom,
                        n_top=n_top,
                        stirrup_dia=stirrup_dia,
                        stirrup_legs=stirrup_legs,
                        stirrup_spacing=(
                            stirrup_spacing
                        ),
                    )
                )

            except ValueError as e:

                st.error(
                    str(e)
                )

                st.session_state.rc_design = (
                    None
                )

        # ----------------------------------------------------
        # Design Result
        # ----------------------------------------------------

        if st.session_state.rc_design:

            design = (
                st.session_state.rc_design
            )

            # =================================================
            # FLEXURE
            # =================================================

            st.subheader(
                "Hasil Lentur"
            )

            c1, c2, c3, c4, c5 = (
                st.columns(5)
            )

            c1.metric(
                "d",
                f"{design['d']:.1f} mm",
            )

            c2.metric(
                "As,min",
                f"{design['As_min']:.0f} mm²",
            )

            c3.metric(
                "As perlu",
                (
                    f"{design['As_required']:.0f} mm²"
                    if np.isfinite(
                        design["As_required"]
                    )
                    else "N/A"
                ),
            )

            c4.metric(
                "As tersedia",
                f"{design['As_provided']:.0f} mm²",
            )

            c5.metric(
                "φMn",
                f"{design['phi_Mn']:.2f} kNm",
            )

            st.write(
                f"β₁ = **{design['beta1']:.3f}**, "
                f"c = **{design['c']:.1f} mm**, "
                f"εt = **{design['eps_t']:.5f}**, "
                f"φ = **{design['phi_flex']:.3f}**"
            )

            if design["flexure_ok"]:

                st.success(
                    "✅ Lentur OK — "
                    "φMn ≥ Mu dan As tersedia "
                    "≥ As,min"
                )

            else:

                st.error(
                    "❌ Lentur TIDAK OK — "
                    "periksa As, dimensi, "
                    "atau kapasitas penampang."
                )

            # =================================================
            # SHEAR
            # =================================================

            st.subheader(
                "Hasil Geser"
            )

            c1, c2, c3, c4, c5 = (
                st.columns(5)
            )

            c1.metric(
                "Vc",
                f"{design['Vc']:.2f} kN",
            )

            c2.metric(
                "Vs",
                f"{design['Vs']:.2f} kN",
            )

            c3.metric(
                "φVn",
                f"{design['phi_Vn']:.2f} kN",
            )

            c4.metric(
                "Av,min",
                f"{design['Av_min']:.1f} mm²",
            )

            c5.metric(
                "s max",
                f"{design['s_max']:.0f} mm",
            )

            checks = pd.DataFrame(
                {
                    "Check": [
                        "φVn ≥ Vu",
                        "Av ≥ Av,min",
                        "Vs ≤ Vs,max",
                        "s ≤ smax",
                        "Clear spacing longitudinal",
                    ],

                    "Hasil": [
                        (
                            "OK"
                            if design[
                                "shear_strength_ok"
                            ]
                            else "NOT OK"
                        ),

                        (
                            "OK"
                            if design[
                                "shear_reinf_ok"
                            ]
                            else "NOT OK"
                        ),

                        (
                            "OK"
                            if design[
                                "shear_vs_limit_ok"
                            ]
                            else "NOT OK"
                        ),

                        (
                            "OK"
                            if design[
                                "spacing_ok"
                            ]
                            else "NOT OK"
                        ),

                        (
                            "OK"
                            if design[
                                "spacing_long_ok"
                            ]
                            else "NOT OK"
                        ),
                    ],
                }
            )

            st.dataframe(
                checks,
                hide_index=True,
                use_container_width=True,
            )

            if design["shear_ok"]:

                st.success(
                    "✅ Geser OK"
                )

            else:

                st.error(
                    "❌ Geser TIDAK OK — "
                    "salah satu pemeriksaan "
                    "geser/detailing belum "
                    "terpenuhi."
                )

            st.info(
                "Implementasi ini mengacu pada "
                "SNI 2847:2019 untuk konsep desain "
                "kekuatan, minimum tulangan lentur, "
                "faktor reduksi berbasis regangan, "
                "dan desain geser. Implementasi ini "
                "belum mencakup seluruh persyaratan "
                "SNI untuk semua kondisi."
            )

else:

    st.info(
        "Analisis harus dijalankan terlebih dahulu."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BURGAM.STUD / Balok Sederhana — "
    "Dukung kami https://instagram.com/buruhgambar.id"
)