import plotly.graph_objects as go


def create_beam_model(L, P, a):

    fig = go.Figure()

    # ========================================================
    # BEAM
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=[0, L],
            y=[0, 0],
            mode="lines",
            line=dict(width=8),
            showlegend=False
        )
    )


    # ========================================================
    # SUPPORT A
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[-0.12],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=18
            ),
            showlegend=False
        )
    )


    # ========================================================
    # SUPPORT B
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=[L],
            y=[-0.1],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=16
            ),
            showlegend=False
        )
    )


    # ========================================================
    # LOAD ARROW
    # ========================================================

    fig.add_annotation(
    x=a,
    y=0.08,
    ax=a,
    ay=1.25,
    text="",
    showarrow=True,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=3
    )

    # Load label
    fig.add_annotation(
    x=a,
    y=1.45,
    text=f"P = {P:.2f} kN",
    showarrow=False,
    font=dict(
        size=14
    )
    )


    # ========================================================
    # DIMENSION L
    # ========================================================

    fig.add_annotation(
        x=L / 2,
        y=-0.7,
        text=f"L = {L:.2f} m",
        showarrow=False
    )


    # ========================================================
    # DIMENSION a
    # ========================================================

    b = L - a
    fig.add_annotation(
    x=(a + L) / 2,
    y=-0.3,
    text=f"b = {b:.2f} m",
    showarrow=False
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(
        height=350,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),

        xaxis=dict(
            visible=False,
            range=[
                -L * 0.1,
                L * 1.1
            ]
        ),

        yaxis=dict(
            visible=False,
            range=[
                -1.2,
                2
            ]
        ),

        showlegend=False
    )

    return fig