"""
Shared UI helpers for the MRV portal Streamlit application.
"""
from typing import Optional

import pandas as pd
import streamlit as st


def _render_empty_message(message: str) -> None:
    """Standard info message when no data exists for a section."""
    st.info(message)


def _kpi_card(title: str, value: str, help_text: Optional[str] = None, delta: Optional[str] = None) -> None:
    """Consistent metric card layout reused across pages."""
    with st.container():
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)
        if delta:
            st.caption(delta)
        st.markdown("</div>", unsafe_allow_html=True)


def _apply_common_filters(df: pd.DataFrame, flt: dict) -> pd.DataFrame:
    """Filter helper shared between pages for project/version/method/site/stratum selections."""
    if df.empty:
        return df
    out = df.copy()
    if "project" in out.columns and "project" in flt:
        out = out[out["project"] == flt["project"]]
    if "dataset_version" in out.columns and flt.get("dataset_version"):
        out = out[out["dataset_version"] == flt["dataset_version"]]
    if "methodology" in out.columns and flt.get("methodologies"):
        out = out[out["methodology"].isin(flt["methodologies"])]
    if "site" in out.columns and flt.get("sites"):
        out = out[out["site"].isin(flt["sites"])]
    if "stratum" in out.columns and flt.get("strata"):
        out = out[out["stratum"].isin(flt["strata"])]
    return out

