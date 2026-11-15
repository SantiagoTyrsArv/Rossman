"""Figuras Plotly reutilizables por las páginas de la app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.utils.settings import MODEL_LABELS

# Paleta categórica fija por modelo (misma en todas las páginas)
MODEL_COLORS = {
    "naive": "#8c8c8c",
    "sarima": "#2a78d6",
    "sarimax": "#d1495b",
    "gru": "#1f9e76",
}
REAL_COLOR = "#1b1b1f"
SCENARIO_COLOR = "#e08a1e"
BASE_COLOR = "#2a78d6"

_LAYOUT = dict(
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    yaxis=dict(tickformat=",.0f", title="Ventas (€)"),
)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def add_band(fig: go.Figure, ds, lo, hi, color: str, name: str, **kw) -> None:
    """Banda semitransparente entre `lo` y `hi` (se omite si todo es NaN)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if np.isnan(lo).all() or np.isnan(hi).all():
        return
    fig.add_trace(
        go.Scatter(
            x=ds,
            y=hi,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=name,
            **kw,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ds,
            y=lo,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=_rgba(color, 0.15),
            name=f"{name} · IC 95 %",
            hoverinfo="skip",
            legendgroup=name,
            **kw,
        )
    )


def forecast_figure(
    history: pd.DataFrame, fc: pd.DataFrame, models: list[str], title: str = ""
) -> go.Figure:
    """Historia reciente + pronóstico de cada modelo con su intervalo al 95 %."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["ds"],
            y=history["y"],
            name="Real",
            mode="lines",
            line=dict(color=REAL_COLOR, width=1.6),
        )
    )
    if fc["y"].notna().any():
        real = fc.drop_duplicates("ds").sort_values("ds")
        fig.add_trace(
            go.Scatter(
                x=real["ds"],
                y=real["y"],
                name="Real (periodo)",
                mode="lines",
                line=dict(color=REAL_COLOR, width=1.6, dash="dot"),
            )
        )
    for m in models:
        g = fc[fc["model"] == m].sort_values("ds")
        if g.empty:
            continue
        color = MODEL_COLORS.get(m, "#555")
        add_band(fig, g["ds"], g["lo_95"], g["hi_95"], color, MODEL_LABELS.get(m, m))
        fig.add_trace(
            go.Scatter(
                x=g["ds"],
                y=g["y_hat"],
                name=MODEL_LABELS.get(m, m),
                mode="lines",
                line=dict(color=color, width=2),
                legendgroup=MODEL_LABELS.get(m, m),
            )
        )
    fig.update_layout(title=title, height=440, **_LAYOUT)
    return fig


def scenario_figure(sim: pd.DataFrame) -> go.Figure:
    """Curvas base vs escenario con sus intervalos."""
    fig = go.Figure()
    add_band(fig, sim["ds"], sim["base_lo"], sim["base_hi"], BASE_COLOR, "Base")
    fig.add_trace(
        go.Scatter(
            x=sim["ds"],
            y=sim["base"],
            name="Base (patrón histórico)",
            line=dict(color=BASE_COLOR, width=2),
            legendgroup="Base",
        )
    )
    add_band(fig, sim["ds"], sim["scen_lo"], sim["scen_hi"], SCENARIO_COLOR, "Escenario")
    fig.add_trace(
        go.Scatter(
            x=sim["ds"],
            y=sim["scen"],
            name="Escenario",
            line=dict(color=SCENARIO_COLOR, width=2.4),
            legendgroup="Escenario",
        )
    )
    fig.update_layout(height=440, **_LAYOUT)
    return fig


def history_figure(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Serie histórica con rangeslider."""
    fig = go.Figure(
        go.Scatter(
            x=df["ds"],
            y=df["y"],
            mode="lines",
            name="Ventas",
            line=dict(color=BASE_COLOR, width=1.4),
        )
    )
    fig.update_layout(title=title, height=420, **_LAYOUT)
    fig.update_layout(hovermode="x")
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=3, label="3 m", step="month", stepmode="backward"),
                dict(count=6, label="6 m", step="month", stepmode="backward"),
                dict(count=1, label="1 a", step="year", stepmode="backward"),
                dict(step="all", label="Todo"),
            ]
        ),
    )
    return fig


def stl_figure(ds, observed, trend, seasonal, resid) -> go.Figure:
    """Descomposición STL en 4 paneles."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Observada", "Tendencia", "Estacional (s = 7)", "Residuo"],
    )
    for i, (y, color) in enumerate(
        [(observed, REAL_COLOR), (trend, BASE_COLOR), (seasonal, "#1f9e76"), (resid, "#8c8c8c")],
        start=1,
    ):
        mode = "markers" if i == 4 else "lines"
        fig.add_trace(
            go.Scatter(
                x=ds,
                y=y,
                mode=mode,
                showlegend=False,
                line=dict(color=color, width=1.2),
                marker=dict(color=color, size=2.5),
            ),
            row=i,
            col=1,
        )
    fig.update_layout(template="plotly_white", height=720, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def correlogram(values: np.ndarray, conf: float, title: str) -> go.Figure:
    """ACF o PACF como barras, con bandas de significancia ±conf."""
    lags = np.arange(len(values))
    fig = go.Figure(
        go.Bar(
            x=lags,
            y=values,
            marker_color=BASE_COLOR,
            name=title,
            hovertemplate="lag %{x}: %{y:.3f}<extra></extra>",
        )
    )
    for sign in (1, -1):
        fig.add_hline(y=sign * conf, line=dict(color="#d1495b", dash="dash", width=1))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Rezago (días)",
        yaxis=dict(range=[-1, 1]),
    )
    return fig


def heatmap_figure(pivot: pd.DataFrame, title: str) -> go.Figure:
    """Heatmap día de la semana × mes."""
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Blues",
            colorbar=dict(title="€"),
            hovertemplate="%{y} · %{x}: %{z:,.0f} €<extra></extra>",
        )
    )
    fig.update_layout(
        title=title, template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig
