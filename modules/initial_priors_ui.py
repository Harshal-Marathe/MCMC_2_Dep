"""
Initial-β prior window (Tab 4 · D4).

The latent state path starts at  x_0 ~ N(x0, sd0^2)  (modules/mcmc.py).
By default every regression beta starts at x0 = 0 with a very wide
sd0 = 3 * std(y) / mean|x|  ("3 target-sds of effect at typical input level").
For a small-input channel that sd is huge, so the first-period beta can
land far above the later betas (the spike at period 0 in the beta chart).

This window lets the user, per variable, set
  * Initial mean  - where the beta starts (default 0)
  * Initial sd    - how tightly it is held there (blank/0 = automatic)
and a global multiplier that shrinks every automatic sd at once.
Result is stored in the config as
  initial_beta_priors  = {var: {"mean": float|None, "sd": float|None}}
  initial_beta_sd_mult = float in (0, 1]
and applied in mcmc.build_static().
"""

import numpy as np
import pandas as pd
import streamlit as st


def auto_initial_sd(df_train, target, col):
    """Same formula as the default in modules/mcmc.py (kept in sync)."""
    y = df_train[target].values.astype(float)
    y_std = float(np.std(y))
    y_std = y_std if y_std > 1e-12 else 1.0
    reg_mean = float(np.mean(np.abs(df_train[col].values.astype(float))))
    return 3.0 * y_std / reg_mean if reg_mean > 1e-9 else 1e-2


def render_initial_beta_priors(df, n_train, target, target2, var_groups, key_prefix="d4_",
                               current=None, current_mult=1.0):
    """
    var_groups: list of (type_label, [col, ...], target_for_these_cols).
    current / current_mult: previously saved priors, used to pre-fill the table.
    Returns (priors_dict, sd_mult).
    """
    current = current or {}
    df_tr = df.iloc[:n_train]
    sd_mult = st.slider(
        "Shrink all automatic initial sds by", 0.01, 1.0,
        float(min(max(current_mult or 1.0, 0.01), 1.0)), 0.01,
        key=f"{key_prefix}sd_mult",
        help="1.0 = default (very wide). e.g. 0.1 makes every variable's starting "
             "beta 10x tighter around its initial mean. Per-variable sds typed "
             "below override this for that variable.")

    rows, seen = [], set()
    for label, cols, tgt in var_groups:
        for c in cols:
            if c in seen or c not in df_tr.columns or tgt not in df_tr.columns:
                continue
            seen.add(c)
            rows.append({"Variable": c, "Type": label,
                         "Auto initial sd": auto_initial_sd(df_tr, tgt, c) * sd_mult,
                         "Initial mean": float((current.get(c) or {}).get("mean") or 0.0),
                         "Initial sd (0 = auto)": float((current.get(c) or {}).get("sd") or 0.0)})
    if not rows:
        st.caption("No regression variables selected yet.")
        return {}, float(sd_mult)

    base = pd.DataFrame(rows)
    sig = f"{key_prefix}tbl_{abs(hash(tuple(base['Variable'])))}"
    edited = st.data_editor(
        base, key=sig, hide_index=True, use_container_width=True,
        disabled=["Variable", "Type", "Auto initial sd"],
        column_config={
            "Auto initial sd": st.column_config.NumberColumn(format="%.4g"),
            "Initial mean": st.column_config.NumberColumn(
                format="%.6g", help="Starting value of this variable's beta (period 0)."),
            "Initial sd (0 = auto)": st.column_config.NumberColumn(
                min_value=0.0, format="%.6g",
                help="Prior sd of the period-0 beta. Smaller = beta is held closer "
                     "to the initial mean at the start. 0 = use the automatic sd."),
        })

    priors = {}
    for _, r in edited.iterrows():
        m = float(r["Initial mean"]) if pd.notna(r["Initial mean"]) else 0.0
        s = float(r["Initial sd (0 = auto)"]) if pd.notna(r["Initial sd (0 = auto)"]) else 0.0
        if m != 0.0 or s > 0.0:
            priors[r["Variable"]] = {"mean": m, "sd": s if s > 0.0 else None}
    return priors, float(sd_mult)
