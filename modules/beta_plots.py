"""
Time-varying beta charts — one chart per variable, picked from a dropdown
(same pattern as the Response Curves section).

For the selected variable:
  primary   (left) axis : beta_t, the posterior-mean time-varying coefficient,
                          with its 95% credible band when the result has one
  secondary (right) axis: the variable's raw input series over the same periods

The Intercept has no input series, so only its beta_t (the base level) is drawn.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from modules.contrib_tables import _variable_index

_TYPE_LABEL = {
    "intercept": "Intercept", "media": "Own media", "comp_media": "Competitor media",
    "own_nonmedia": "Own non-media", "comp_nonmedia": "Competitor non-media",
    "price": "Price",
}


def beta_series(res, state_idx):
    """(beta, lo, hi) for one state; lo/hi are None when the result has no band."""
    beta = np.asarray(res["x_smooth"])[:, state_idx]
    lo = hi = None
    if res.get("beta_lo") is not None and res.get("beta_hi") is not None:
        lo = np.asarray(res["beta_lo"])[:, state_idx]
        hi = np.asarray(res["beta_hi"])[:, state_idx]
    return beta, lo, hi


def make_beta_fig(name, group, beta, lo, hi, x_input, n_train, show_band=True):
    T = len(beta)
    x = np.arange(T)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if n_train is not None and 0 < n_train < T:
        fig.add_vrect(x0=n_train, x1=T - 1, fillcolor="#fef3c7", opacity=0.35,
                      layer="below", line_width=0,
                      annotation_text="Test period", annotation_position="top left")

    if show_band and lo is not None and hi is not None:
        fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(37,99,235,0.15)",
                                 name="β 95% band", hoverinfo="skip"), secondary_y=False)

    fig.add_trace(go.Scatter(x=x, y=beta, mode="lines", name=f"β_t ({name})",
                             line=dict(color="#1d4ed8", width=2.5)), secondary_y=False)

    if x_input is not None:
        fig.add_trace(go.Scatter(x=x, y=x_input, mode="lines", name=f"{name} (input)",
                                 line=dict(color="#f59e0b", width=1.6)), secondary_y=True)

    fig.update_xaxes(title_text="Period")
    fig.update_yaxes(title_text=f"β_t — {name}", secondary_y=False, color="#1d4ed8")
    if x_input is not None:
        fig.update_yaxes(title_text=f"{name} (input)", secondary_y=True,
                         color="#b45309", showgrid=False)
    fig.update_layout(template="plotly_white", height=460,
                      title=f"Time-varying β and input — {name}",
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    return fig


def render_beta_charts(df, config, res, g, kp=""):
    """Dropdown + chart for every variable of this result (own/competitor
    media, non-media, price, intercept)."""
    items = _variable_index(g)
    names = [n for n, _grp, _si in items]
    info = {n: (grp, si) for n, grp, si in items}
    if not names:
        st.info("No variables configured.")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        sel = st.selectbox("Select variable", names, key=f"{kp}beta_sel",
                           format_func=lambda n: f"{n}  ·  {_TYPE_LABEL[info[n][0]]}")
    group, si = info[sel]
    beta, lo, hi = beta_series(res, si)
    has_band = lo is not None
    with c2:
        show_band = st.checkbox("Show 95% band", value=has_band, disabled=not has_band,
                                key=f"{kp}beta_band")
    if not has_band:
        st.caption("95% bands aren't stored on this result (fitted with an older build) — "
                   "re-run the model to see them.")

    x_input = None
    if group != "intercept" and sel in df.columns:
        x_input = df[sel].values.astype(float)

    fig = make_beta_fig(sel, group, beta, lo, hi, x_input, config.get("n_train"), show_band)
    st.plotly_chart(fig, use_container_width=True, key=f"{kp}fig_beta")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average β", f"{np.mean(beta):.6g}")
    m2.metric("Min β", f"{np.min(beta):.6g}")
    m3.metric("Max β", f"{np.max(beta):.6g}")
    m4.metric("Std of β", f"{np.std(beta):.6g}")

    out = pd.DataFrame({"Period": np.arange(len(beta)), "beta": beta})
    if has_band:
        out["beta_lo_2.5%"], out["beta_hi_97.5%"] = lo, hi
    if x_input is not None:
        out[f"{sel}_input"] = x_input
    d1, d2 = st.columns(2)
    d1.download_button(f"📥 Download β series ({sel})", out.to_csv(index=False).encode(),
                       f"beta_{sel}.csv", "text/csv", key=f"{kp}dl_beta_csv")
    try:
        png = fig.to_image(format="png", scale=2)
        d2.download_button(f"🖼️ Download chart image ({sel})", png, f"beta_{sel}.png",
                           "image/png", key=f"{kp}dl_beta_png")
    except Exception as e:
        d2.caption(f"⚠️ Could not render a PNG ({e}). Install `kaleido` for image export.")
