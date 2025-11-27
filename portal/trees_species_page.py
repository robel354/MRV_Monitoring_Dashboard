"""
Trees and Species page for the MRV portal.
"""
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from .ui_helpers import _apply_common_filters, _kpi_card, _render_empty_message


def page_trees_species(data: Dict[str, object], flt: dict) -> None:
    """Entry point for the Trees & Species page."""
    st.title("Trees & Species")
    st.caption("Tree planting metrics, species breakdown, and agroforestry/reforestation statistics.")

    monitoring_df: pd.DataFrame = data["monitoring"]
    species_df = data.get("species", pd.DataFrame())

    mon = _apply_common_filters(monitoring_df, flt)
    if flt.get("periods"):
        mon = mon[mon["period"].isin(flt["periods"])].copy()

    # Tree planting KPIs
    st.markdown("### Tree Planting Summary")
    row1 = st.columns(3)
    
    trees_af = f"{int(mon['agroforestry_trees_planted'].sum()):,}" if not mon.empty and 'agroforestry_trees_planted' in mon.columns else "—"
    trees_re = f"{int(mon['reforestation_trees_planted'].sum()):,}" if not mon.empty and 'reforestation_trees_planted' in mon.columns else "—"
    trees_total = f"{int(mon['trees_planted'].sum()):,}" if not mon.empty and 'trees_planted' in mon.columns else "—"

    with row1[0]:
        _kpi_card("Agroforestry trees planted", trees_af)
    with row1[1]:
        _kpi_card("Reforestation trees planted", trees_re)
    with row1[2]:
        _kpi_card("Total trees planted", trees_total)

    # Additional metrics row
    row2 = st.columns(3)
    hectares_reforested = f"{mon['hectares_reforested'].sum():,.1f} ha" if not mon.empty and 'hectares_reforested' in mon.columns else "—"
    hectares_af = f"{mon['hectares_agroforestry'].sum():,.1f} ha" if not mon.empty and 'hectares_agroforestry' in mon.columns else "—"
    species_count = f"{int(mon['species_count'].max()):,}" if not mon.empty and 'species_count' in mon.columns else "—"

    with row2[0]:
        _kpi_card("Hectares reforested", hectares_reforested)
    with row2[1]:
        _kpi_card("Area under agroforestry", hectares_af)
    with row2[2]:
        _kpi_card("Species count", species_count)

    # Trees per species section
    st.markdown("### Trees Per Species")
    if species_df.empty:
        _render_empty_message("No species data available.")
    else:
        # Filter species by current project/version/sites
        sdf = species_df.copy()
        if "project" in sdf.columns:
            sdf = sdf[sdf["project"] == flt["project"]]
        if flt.get("sites"):
            sdf = sdf[sdf["site"].isin(flt["sites"])]
        if "dataset_version" in sdf.columns and flt.get("dataset_version"):
            sdf = sdf[sdf["dataset_version"] == flt["dataset_version"]]

        tabs_sp = st.tabs(["Agroforestry", "Reforestation"])
        with tabs_sp[0]:
            st.markdown("#### Agroforestry Species")
            af = sdf[sdf["category"] == "Agroforestry"].groupby("species", as_index=False)["trees_planted"].sum().sort_values("trees_planted", ascending=False)
            if af.empty:
                _render_empty_message("No agroforestry species data for the selected filters.")
            else:
                # Rename columns for display
                af_display = af.copy()
                af_display = af_display.rename(columns={"species": "Species", "trees_planted": "Trees Planted"})
                # Create chart with proper labels
                fig_af = px.bar(
                    af,
                    x="species",
                    y="trees_planted",
                    title="Agroforestry: Trees Per Species",
                    labels={"species": "Species", "trees_planted": "Trees Planted"}
                )
                st.plotly_chart(fig_af, use_container_width=True)
                st.dataframe(af_display, use_container_width=True)

        with tabs_sp[1]:
            st.markdown("#### Reforestation Species")
            re = sdf[sdf["category"] == "Reforestation"].groupby("species", as_index=False)["trees_planted"].sum().sort_values("trees_planted", ascending=False)
            if re.empty:
                _render_empty_message("No reforestation species data for the selected filters.")
            else:
                # Rename columns for display
                re_display = re.copy()
                re_display = re_display.rename(columns={"species": "Species", "trees_planted": "Trees Planted"})
                # Create chart with proper labels
                fig_re = px.bar(
                    re,
                    x="species",
                    y="trees_planted",
                    title="Reforestation: Trees Per Species",
                    labels={"species": "Species", "trees_planted": "Trees Planted"}
                )
                st.plotly_chart(fig_re, use_container_width=True)
                st.dataframe(re_display, use_container_width=True)

