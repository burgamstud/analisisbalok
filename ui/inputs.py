import streamlit as st


def beam_inputs():

    st.sidebar.header("Input Struktur")

    st.sidebar.subheader("Geometri")

    L = st.sidebar.number_input(
        "Panjang Balok L (m)",
        min_value=0.1,
        value=10.0,
        step=0.5
    )

    st.sidebar.subheader("Penampang")

    b = st.sidebar.number_input(
        "Lebar Penampang b (m)",
        min_value=0.01,
        value=0.30,
        step=0.01
    )

    h = st.sidebar.number_input(
        "Tinggi Penampang h (m)",
        min_value=0.01,
        value=0.40,
        step=0.01
    )

    st.sidebar.subheader("Material")

    E = st.sidebar.number_input(
    "Modulus Elastisitas E (MPa)",
    min_value=1.0,
    value=200_000.0,
    step=1_000.0
    )

    st.sidebar.subheader("Beban Titik")

    P = st.sidebar.number_input(
        "Beban Titik P (kN)",
        min_value=0.0,
        value=20.0,
        step=1.0
    )

    a = st.sidebar.number_input(
        "Posisi Beban dari A (m)",
        min_value=0.0,
        max_value=float(L),
        value=float(L / 2),
        step=0.5
    )

    analyze = st.sidebar.button(
        "🔍 ANALYZE",
        type="primary",
        use_container_width=True
    )

    return {
    "L": L,
    "b": b,
    "h": h,
    "E": E,
    "P": P,
    "a": a,
    "analyze": analyze
    }