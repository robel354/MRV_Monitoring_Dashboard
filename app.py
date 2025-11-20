import io
import json
import math
import random
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


# ---------- Page Configuration ----------
st.set_page_config(
    page_title="MRV Verification & Insights Portal",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Lightweight Styling ----------
_APP_CSS = """
/* Subtle professional styling */
.metric-card {
  border: none;
  border-radius: 8px;
  padding: 0;
  background: transparent;
  min-height: 0;
  display: block;
}
.metric-title {
  font-size: 13px;
  color: #374151;
  margin-bottom: 6px;
}
.metric-value {
  font-size: 28px;
  font-weight: 700;
  line-height: 1.2;
  color: #111827;
}
.legend-pill { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; margin-right:6px; border:1px solid #e5e7eb; }
.alert-banner { border-left: 4px solid #dc2626; padding: 8px 12px; background:#FEF2F2; }
/* Sticky quick filter bar */
.filter-bar { position: sticky; top: 0; z-index: 50; background: #ffffff; border: 1px solid #E5E7EB; border-radius: 12px; padding: 10px 12px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.chip { display:inline-block; padding:4px 8px; border-radius:999px; background:#F3F4F6; font-size:12px; margin-right:6px; }
"""
st.markdown(f"<style>{_APP_CSS}</style>", unsafe_allow_html=True)


# ---------- Types & Constants ----------
METHODOLOGIES = ["VM0042", "VM0047", "CCB"]
VM0042_BASELINE_METRICS = [
    "Aboveground biomass (AGB) (tCO2e)",
    "Belowground biomass (BGB) (tCO2e)",
    "Soil organic carbon (tCO2e)",
    "GHG Emissions — Fossil fuels (tCO2e)",
    "GHG Emissions — Biomass Burning (tCO2e)",
    "GHG Emissions — Nitrogen inputs to Soils (tCO2e)",
]
VM0047_BASELINE_METRICS = [
    "Aboveground biomass (AGB) (tCO2e)",
    "Belowground biomass (BGB) (tCO2e)",
    "Soil organic carbon (tCO2e)",
    "GHG Emissions — Fossil fuels (tCO2e)",
    "GHG Emissions — Biomass Burning (tCO2e)",
    "GHG Emissions — Nitrogen inputs to Soils (tCO2e)",
]
LEAKAGE_METRIC = "Leakage (tCO2e)"
EMISSIONS_METRICS = [
    "GHG Emissions — Fossil fuels (tCO2e)",
    "GHG Emissions — Biomass Burning (tCO2e)",
    "GHG Emissions — Nitrogen inputs to Soils (tCO2e)",
]
COMPLETION_STATUSES = ["Complete", "Partial", "Pending"]


@st.cache_data(show_spinner=False)
def _maybe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _maybe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _maybe_read_geojson(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_data() -> Dict[str, object]:
    """Load data from ./data if available; otherwise create realistic sample data.
    Returns a dictionary of dataframes/geojson objects.
    """
    # Try load from local data folder
    baseline_df = _maybe_read_csv("data/baseline.csv")
    monitoring_df = _maybe_read_csv("data/monitoring.csv")
    qc_df = _maybe_read_csv("data/qc_summary.csv")
    plots_geojson = _maybe_read_geojson("data/plots.geojson")
    boundaries_geojson = _maybe_read_geojson("data/boundaries.geojson")
    strata_geojson = _maybe_read_geojson("data/strata.geojson")
    metadata = _maybe_read_json("data/metadata.json") or {}

    if baseline_df is not None and monitoring_df is not None and qc_df is not None:
        # Harmonize expected columns if needed
        for col in [
            "methodology",
            "site",
            "stratum",
            "dataset_version",
            "analyst",
            "date",
            "metric",
            "value",
            "uncertainty_percent",
            "sampling_uncertainty_percent",
        ]:
            if col not in baseline_df.columns:
                baseline_df[col] = np.nan

        for col in [
            "methodology",
            "site",
            "stratum",
            "dataset_version",
            "period",
            "date",
            "tco2e_this_period",
            "cumulative_tco2e",
            "uncertainty_percent",
            "trees_planted",
            "species_count",
            "plots_measured",
            "run_id",
        ]:
            if col not in monitoring_df.columns:
                monitoring_df[col] = np.nan

        for col in [
            "dataset",
            "methodology",
            "dataset_version",
            "run_id",
            "reviewer",
            "date",
            "total_records",
            "failed_records",
            "severity",
        ]:
            if col not in qc_df.columns:
                qc_df[col] = np.nan

        return {
            "baseline": baseline_df,
            "monitoring": monitoring_df,
            "qc": qc_df,
            "plots_geojson": plots_geojson or {"type": "FeatureCollection", "features": []},
            "boundaries_geojson": boundaries_geojson or {"type": "FeatureCollection", "features": []},
            "strata_geojson": strata_geojson or {"type": "FeatureCollection", "features": []},
            "metadata": metadata,
        }

    # Create sample data when local files are absent
    random.seed(42)
    np.random.seed(42)
    # Align names to examples
    sites = ["Luxiha", "1st May", "Kamwenyetulo"]
    projects = ["Angola"]
    strata = ["Upland", "Lowland", "Riparian"]
    dataset_versions = ["v2025.10.01", "v2025.11.01"]
    analysts = ["Analyst One", "Analyst Two"]
    base_date = datetime(2025, 1, 1)

    # Baseline records
    baseline_rows: List[dict] = []
    for methodology in METHODOLOGIES:
        metrics = VM0042_BASELINE_METRICS if methodology == "VM0042" else VM0047_BASELINE_METRICS
        for version in dataset_versions:
            for site in sites:
                for stratum in strata:
                    # Random performance benchmark fraction for VM0047 (applies to biomass)
                    benchmark_fraction = float(np.clip(np.random.normal(loc=0.8, scale=0.08), 0.6, 0.95)) if methodology == "VM0047" else np.nan
                    for metric in metrics + [LEAKAGE_METRIC]:
                        # Slightly smaller values for emissions/leakage than pools
                        is_emission_or_leakage = metric in EMISSIONS_METRICS + [LEAKAGE_METRIC]
                        mean_val = 4.6 if is_emission_or_leakage else 5.0
                        value = float(np.random.lognormal(mean=mean_val, sigma=0.4))
                        uncert = float(np.clip(np.random.normal(loc=12, scale=4), 3, 30))
                        samp_uncert = float(np.clip(np.random.normal(loc=10, scale=3), 2, 25))
                        baseline_rows.append({
                            "project": "Angola",
                            "methodology": methodology,
                            "site": site,
                            "stratum": stratum,
                            "dataset_version": version,
                            "analyst": random.choice(analysts),
                            "date": (base_date - timedelta(days=30)).date().isoformat(),
                            "metric": metric,
                            "value": round(value, 2),
                            "uncertainty_percent": round(uncert, 2),
                            "sampling_uncertainty_percent": round(samp_uncert, 2),
                            "sample_count": int(np.random.randint(20, 90)),
                            "benchmark_fraction": benchmark_fraction if (methodology == "VM0047" and metric == "Aboveground biomass (AGB) (tCO2e)") else np.nan,
                        })

    baseline_df = pd.DataFrame(baseline_rows)

    # Monitoring records across four periods
    periods = ["2025T0", "2025T1", "2025T2", "2025T3"]
    monitoring_rows: List[dict] = []
    for methodology in METHODOLOGIES:
        for version in dataset_versions:
            for site in sites:
                for stratum in strata:
                    cumulative = 0.0
                    for i, period in enumerate(periods):
                        tco2e = float(max(0, np.random.normal(loc=800, scale=140)))
                        cumulative += tco2e
                        uncert = float(np.clip(np.random.normal(loc=10, scale=3), 2, 25))
                        monitoring_rows.append({
                            "project": "Angola",
                            "methodology": methodology,
                            "site": site,
                            "stratum": stratum,
                            "dataset_version": version,
                            "period": period,
                            "date": (base_date + timedelta(days=90 * i)).date().isoformat(),
                            "tco2e_this_period": round(tco2e, 2),
                            "cumulative_tco2e": round(cumulative, 2),
                            "uncertainty_percent": round(uncert, 2),
                            "trees_planted": int(np.random.randint(1000, 6000)),
                            "agroforestry_trees_planted": int(np.random.randint(500, 3000)),
                            "reforestation_trees_planted": int(np.random.randint(500, 3000)),
                            "species_count": int(np.random.randint(5, 25)),
                            "plots_measured": int(np.random.randint(20, 90)),
                            "hectares_reforested": float(np.random.uniform(5, 120)),
                            "hectares_ag_improved": float(np.random.uniform(10, 200)),
                            "hectares_agroforestry": float(np.random.uniform(5, 150)),
                            "households_enrolled": int(np.random.randint(30, 600)),
                            "run_id": f"RUN-{period}-{random.randint(100,999)}",
                        })

    monitoring_df = pd.DataFrame(monitoring_rows)

    # Species breakdown (synthetic) for AF and RE trees
    species_catalog = {
        "Agroforestry": ["Faidherbia albida", "Grevillea robusta", "Acacia spp.", "Moringa oleifera"],
        "Reforestation": ["Eucalyptus spp.", "Pinus spp.", "Miombo spp.", "Terminalia sericea"],
    }
    species_rows: List[dict] = []
    for version in dataset_versions:
        for site in sites:
            for stratum in strata:
                for category, sp_list in species_catalog.items():
                    for sp in sp_list:
                        species_rows.append({
                            "project": "Angola",
                            "dataset_version": version,
                            "site": site,
                            "stratum": stratum,
                            "category": category,  # Agroforestry or Reforestation
                            "species": sp,
                            "trees_planted": int(np.random.randint(100, 2000)),
                        })
    species_df = pd.DataFrame(species_rows)

    # QC summary
    qc_rows: List[dict] = []
    reviewers = ["QC Reviewer A", "QC Reviewer B"]
    for methodology in METHODOLOGIES:
        for version in dataset_versions:
            for dataset in ["baseline", "monitoring"]:
                total = int(np.random.randint(800, 2500))
                failed = int(np.random.binomial(total, p=0.03))
                severity = "high" if failed / max(total, 1) > 0.05 else "low"
                qc_rows.append({
                    "dataset": dataset,
                    "methodology": methodology,
                    "dataset_version": version,
                    "run_id": f"QC-{random.randint(10000,99999)}",
                    "reviewer": random.choice(reviewers),
                    "date": (base_date + timedelta(days=random.randint(1, 330))).date().isoformat(),
                    "total_records": total,
                    "failed_records": failed,
                    "severity": severity,
                })

    qc_df = pd.DataFrame(qc_rows)

    # Simple spatial data (Angola extents approximate)
    base_lon, base_lat = 16.0, -12.5

    def _point_feature(lon: float, lat: float, props: dict) -> dict:
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        }

    plot_features: List[dict] = []
    for methodology in METHODOLOGIES:
        for version in dataset_versions:
            for site in sites:
                for stratum in strata:
                    for _ in range(20):
                        lon = base_lon + np.random.uniform(-2.0, 2.0)
                        lat = base_lat + np.random.uniform(-2.0, 2.0)
                        component = "biomass" if random.random() < 0.5 else "SOC"
                        value = float(np.random.lognormal(mean=4.5, sigma=0.5))
                        u = float(np.clip(np.random.normal(loc=10, scale=4), 2, 30))
                        plot_features.append(
                            _point_feature(
                                lon,
                                lat,
                                {
                                    "project": "Angola",
                                    "methodology": methodology,
                                    "dataset": component,
                                    "component": component,
                                    "site": site,
                                    "stratum": stratum,
                                    "completion_status": random.choice(COMPLETION_STATUSES),
                                    "collection_date": (base_date + timedelta(days=random.randint(1, 330))).date().isoformat(),
                                    "dataset_version": version,
                                    "value": round(value, 2),
                                    "uncertainty_percent": round(u, 2),
                                },
                            )
                        )

    plots_geojson = {"type": "FeatureCollection", "features": plot_features}

    # Minimal boundary and stratum polygons (rectangles)
    def _polygon_feature(coords: List[Tuple[float, float]], props: dict) -> dict:
        # Coordinates must be closed (first==last)
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": props,
        }

    boundaries_geojson = {
        "type": "FeatureCollection",
        "features": [
            _polygon_feature(
                [
                    (base_lon - 3.5, base_lat - 3.5),
                    (base_lon + 3.5, base_lat - 3.5),
                    (base_lon + 3.5, base_lat + 3.5),
                    (base_lon - 3.5, base_lat + 3.5),
                ],
                {"name": "Project Boundary"},
            )
        ],
    }

    strata_geojson = {
        "type": "FeatureCollection",
        "features": [
            _polygon_feature(
                [
                    (base_lon - 2.0, base_lat - 2.0),
                    (base_lon, base_lat - 2.0),
                    (base_lon, base_lat),
                    (base_lon - 2.0, base_lat),
                ],
                {"name": "Upland"},
            ),
            _polygon_feature(
                [
                    (base_lon, base_lat - 2.0),
                    (base_lon + 2.0, base_lat - 2.0),
                    (base_lon + 2.0, base_lat),
                    (base_lon, base_lat),
                ],
                {"name": "Lowland"},
            ),
            _polygon_feature(
                [
                    (base_lon - 2.0, base_lat),
                    (base_lon, base_lat),
                    (base_lon, base_lat + 2.0),
                    (base_lon - 2.0, base_lat + 2.0),
                ],
                {"name": "Riparian"},
            ),
        ],
    }

    metadata = {
        "current_dataset_version": dataset_versions[-1],
        "last_update": datetime.now().date().isoformat(),
        "responsible_analyst": analysts[-1],
    }

    return {
        "baseline": baseline_df,
        "monitoring": monitoring_df,
        "species": species_df,
        "qc": qc_df,
        "plots_geojson": plots_geojson,
        "boundaries_geojson": boundaries_geojson,
        "strata_geojson": strata_geojson,
        "metadata": metadata,
    }


# ---------- Navigation & Filters UI ----------
def sidebar_advanced_filters(data: Dict[str, object]) -> dict:
    st.sidebar.markdown("**Advanced filters**")
    with st.sidebar.expander("Sampling plots", expanded=False):
        selected_plot_dataset = st.multiselect("Dataset", ["biomass", "SOC"], default=["biomass", "SOC"], key="adv_plot_dataset")
        selected_completion = st.multiselect("Completion status", COMPLETION_STATUSES, default=COMPLETION_STATUSES, key="adv_completion")
    return {
        "plot_dataset": selected_plot_dataset,
        "plot_completion": selected_completion,
    }


def quick_filter_bar(data: Dict[str, object]) -> dict:
    baseline_df: pd.DataFrame = data["baseline"]
    monitoring_df: pd.DataFrame = data["monitoring"]

    projects = sorted(set(baseline_df.get("project", pd.Series(["All"]))).union(monitoring_df.get("project", pd.Series(["All"]))))
    dataset_versions = sorted(set(baseline_df["dataset_version"]).union(monitoring_df["dataset_version"]))
    # These will be filtered by project selection below
    all_sites = sorted(set(baseline_df["site"]).union(monitoring_df["site"]))
    all_strata = sorted(set(baseline_df["stratum"]).union(monitoring_df["stratum"]))
    periods = sorted(monitoring_df["period"].dropna().unique().tolist())

    # Defaults
    default_project = projects[0] if projects else "All"
    default_version = dataset_versions[-1] if dataset_versions else None
    default_methods = METHODOLOGIES
    default_sites = all_sites
    default_strata = all_strata
    default_periods = periods

    # Session state for persistence
    if "flt_project" not in st.session_state:
        st.session_state.flt_project = default_project
    if "flt_version" not in st.session_state:
        st.session_state.flt_version = default_version
    if "flt_methods" not in st.session_state:
        st.session_state.flt_methods = default_methods
    if "flt_sites" not in st.session_state:
        st.session_state.flt_sites = default_sites
    if "flt_strata" not in st.session_state:
        st.session_state.flt_strata = default_strata
    if "flt_periods" not in st.session_state:
        st.session_state.flt_periods = default_periods

    st.markdown("<div class='filter-bar'>", unsafe_allow_html=True)
    row1 = st.columns([1.0, 1.0, 1.0, 1.0, 0.6])
    with row1[0]:
        st.session_state.flt_project = st.selectbox("Project", projects, index=0, key="flt_project_select")
    with row1[1]:
        st.session_state.flt_version = st.selectbox("Dataset version", dataset_versions, index=max(0, len(dataset_versions) - 1) if default_version else 0, key="flt_version_select")
    with row1[2]:
        st.session_state.flt_methods = st.multiselect("Methodology", METHODOLOGIES, default=st.session_state.flt_methods)
    # Filter sites/strata by project (if present)
    project_mask_base = (baseline_df["project"] == st.session_state.flt_project) if "project" in baseline_df.columns else pd.Series([True] * len(baseline_df))
    project_mask_mon = (monitoring_df["project"] == st.session_state.flt_project) if "project" in monitoring_df.columns else pd.Series([True] * len(monitoring_df))
    sites = sorted(set(baseline_df.loc[project_mask_base, "site"]).union(monitoring_df.loc[project_mask_mon, "site"]))
    strata = sorted(set(baseline_df.loc[project_mask_base, "stratum"]).union(monitoring_df.loc[project_mask_mon, "stratum"]))
    with row1[3]:
        st.session_state.flt_sites = st.multiselect("Site", sites, default=sites)
    with row1[4]:
        if st.button("Reset", use_container_width=True):
            st.session_state.flt_project = default_project
            st.session_state.flt_version = default_version
            st.session_state.flt_methods = default_methods
            st.session_state.flt_sites = sites
            st.session_state.flt_strata = strata
            st.session_state.flt_periods = default_periods

    row2 = st.columns([2, 3])
    with row2[0]:
        st.session_state.flt_periods = st.multiselect("Monitoring periods", periods, default=st.session_state.flt_periods)
    with row2[1]:
        # Compact summary chips
        sel_methods = ", ".join(st.session_state.flt_methods) if st.session_state.flt_methods else "—"
        sel_sites = ", ".join(st.session_state.flt_sites) if st.session_state.flt_sites else "—"
        sel_strata = ", ".join(strata) if strata else "—"
        st.markdown(f"Selected: <span class='chip'>Project: {st.session_state.flt_project}</span> <span class='chip'>Methods: {sel_methods}</span> <span class='chip'>Sites: {sel_sites}</span> <span class='chip'>Strata: {sel_strata}</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "project": st.session_state.flt_project,
        "dataset_version": st.session_state.flt_version,
        "methodologies": st.session_state.flt_methods,
        "sites": st.session_state.flt_sites,
        "strata": strata,
        "periods": st.session_state.flt_periods,
    }


# ---------- Utility Rendering ----------
def _render_empty_message(message: str) -> None:
    st.info(message)


def _kpi_card(title: str, value: str, help_text: Optional[str] = None, delta: Optional[str] = None) -> None:
    with st.container():
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)
        st.markdown("</div>", unsafe_allow_html=True)


def _apply_common_filters(df: pd.DataFrame, flt: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "project" in out.columns and "project" in flt:
        out = out[out["project"] == flt["project"]]
    if "dataset_version" in out.columns:
        out = out[out["dataset_version"] == flt["dataset_version"]]
    if "methodology" in out.columns:
        out = out[out["methodology"].isin(flt["methodologies"])]
    if "site" in out.columns:
        out = out[out["site"].isin(flt["sites"])]
    if "stratum" in out.columns:
        out = out[out["stratum"].isin(flt["strata"])]
    return out


# ---------- Pages ----------
def page_baseline(data: Dict[str, object], flt: dict) -> None:
    st.title("Baseline")
    st.caption("Baseline carbon stocks, emissions, leakage and CCB indicators by methodology, site and stratum.")

    baseline_df: pd.DataFrame = data["baseline"]
    base_all = _apply_common_filters(baseline_df, flt)
    if base_all.empty:
        _render_empty_message("No baseline data for the selected filters.")
        return

    tab1, tab2, tab3 = st.tabs(["VM0042", "VM0047", "CCB"])

    def render_vm(methodology: str) -> None:
        df = base_all[base_all["methodology"] == methodology]
        if df.empty:
            _render_empty_message(f"No baseline data for {methodology} under current filters.")
            return

        # Metric cards
        def metric_value(name: str) -> Tuple[float, float]:
            subset = df[df["metric"] == name]
            return float(subset["value"].sum()), float(subset["uncertainty_percent"].mean()) if not subset.empty else (0.0, float("nan"))  # type: ignore

        col1, col2, col3 = st.columns(3)
        v_agb, u_agb = metric_value("Aboveground biomass (AGB) (tCO2e)")
        v_bgb, u_bgb = metric_value("Belowground biomass (BGB) (tCO2e)")
        v_soc, u_soc = metric_value("Soil organic carbon (tCO2e)")
        v_foss, u_foss = metric_value("GHG Emissions — Fossil fuels (tCO2e)")
        v_burn, u_burn = metric_value("GHG Emissions — Biomass Burning (tCO2e)")
        v_nit, u_nit = metric_value("GHG Emissions — Nitrogen inputs to Soils (tCO2e)")
        v_leak, _ = metric_value(LEAKAGE_METRIC)

        with col1:
            _kpi_card("AGB VCUs", f"{v_agb:,.0f} tCO₂e", help_text=f"+ {u_agb:.1f}%")
        with col2:
            _kpi_card("BGB VCUs", f"{v_bgb:,.0f} tCO₂e", help_text=f"+ {u_bgb:.1f}%")
        with col3:
            _kpi_card("SOC VCUs", f"{v_soc:,.0f} tCO₂e", help_text=f"+ {u_soc:.1f}%")

        col4, col5, col6 = st.columns(3)
        with col4:
            _kpi_card("Leakage", f"{v_leak:,.0f} tCO₂e")
        with col5:
            _kpi_card("Emissions — Fossil fuels", f"{v_foss:,.0f} tCO₂e", help_text=f"+ {u_foss:.1f}%")
        with col6:
            _kpi_card("Emissions — Biomass burning", f"{v_burn:,.0f} tCO₂e", help_text=f"+ {u_burn:.1f}%")
        _kpi_card("Emissions — Nitrogen inputs", f"{v_nit:,.0f} tCO₂e", help_text=f"+ {u_nit:.1f}%")

        # Charts: Baseline components by grouping
        group_by = "site"
        comp_keep = ["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"] + EMISSIONS_METRICS + [LEAKAGE_METRIC]
        comp = df[df["metric"].isin(comp_keep)].groupby([group_by, "metric"], as_index=False)["value"].sum()
        fig = px.bar(comp, x=group_by, y="value", color="metric", barmode="group", labels={"value": "tCO₂e (VCUs)"})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Component")
        st.plotly_chart(fig, use_container_width=True)

        # Net GHG balance with positive/negative bars
        sign_map = {m: 1 for m in ["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"]}
        sign_map.update({m: -1 for m in EMISSIONS_METRICS + [LEAKAGE_METRIC]})
        comp["signed_value"] = comp.apply(lambda r: r["value"] * sign_map.get(r["metric"], 1), axis=1)
        # Add a net bar per site (avoid duplicate 'value' column names)
        net = comp.groupby(group_by, as_index=False)["signed_value"].sum().rename(columns={"signed_value": "value"}).assign(metric="Net")
        comp_signed = comp[[group_by, "metric", "signed_value"]].rename(columns={"signed_value": "value"})
        comp_net = pd.concat([comp_signed, net[[group_by, "metric", "value"]]], ignore_index=True)
        fign = px.bar(comp_net, x=group_by, y="value", color="metric", barmode="relative", labels={"value": "Net tCO₂e (VCUs)"})
        fign.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Component")
        st.plotly_chart(fign, use_container_width=True)

        # Tables
        st.markdown("#### Table 1 — Baseline carbon stocks")
        # Table filters
        sites_sel = st.multiselect("Filter sites", sorted(df["site"].unique().tolist()), default=sorted(df["site"].unique().tolist()), key=f"t1_sites_{methodology}")
        strata_sel = st.multiselect("Filter strata", sorted(df["stratum"].unique().tolist()), default=sorted(df["stratum"].unique().tolist()), key=f"t1_strata_{methodology}")
        pools_sel = st.multiselect("Filter pools", ["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"], default=["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"], key=f"t1_pools_{methodology}")
        t1 = df[(df["site"].isin(sites_sel)) & (df["stratum"].isin(strata_sel)) & (df["metric"].isin(pools_sel))][
            ["site", "stratum", "metric", "value", "uncertainty_percent", "sample_count"]
        ].rename(columns={"metric": "Carbon pool", "value": "Value (tCO₂e / VCUs)", "uncertainty_percent": "Uncertainty (+%)"})
        st.dataframe(t1.sort_values(["site", "stratum", "Carbon pool"]), use_container_width=True)

        st.markdown("#### Table 2 — Project emissions and leakage")
        sources_sel = st.multiselect("Filter sources", EMISSIONS_METRICS + [LEAKAGE_METRIC], default=EMISSIONS_METRICS + [LEAKAGE_METRIC], key=f"t2_sources_{methodology}")
        t2 = df[(df["site"].isin(sites_sel)) & (df["stratum"].isin(strata_sel)) & (df["metric"].isin(sources_sel))][
            ["site", "stratum", "metric", "value", "uncertainty_percent", "sample_count"]
        ].rename(columns={"metric": "Emission source", "value": "Value (tCO₂e / VCUs)", "uncertainty_percent": "Uncertainty (+%)"})
        st.dataframe(t2.sort_values(["site", "stratum", "Emission source"]), use_container_width=True)

    with tab1:
        render_vm("VM0042")

    with tab2:
        df47 = base_all[base_all["methodology"] == "VM0047"].copy()
        if df47.empty:
            _render_empty_message("No baseline data for VM0047 under current filters.")
        else:
            # Metric cards like VM0042 plus benchmark
            render_vm("VM0047")
            # Benchmark fraction card (mean over selected rows where available)
            bf = df47["benchmark_fraction"].dropna()
            if not bf.empty:
                _kpi_card("Performance benchmark fraction", f"{bf.mean():.2f}", help_text="Applied to biomass")

            # VM0047 net with benchmark deduction visual (approximate)
            group_by = st.radio("Group by (VM0047)", ["site", "stratum"], horizontal=True, key="group_by_vm0047")
            comp = df47[df47["metric"].isin(["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"] + EMISSIONS_METRICS + [LEAKAGE_METRIC])]
            comp = comp.groupby([group_by, "metric"], as_index=False).agg(value=("value", "sum"), benchmark_fraction=("benchmark_fraction", "mean"))
            # Split biomass into retained and deduction parts
            bio = comp[comp["metric"] == "Aboveground biomass (AGB) (tCO2e)"].copy()
            bio["biomass_retained"] = bio["value"] * bio["benchmark_fraction"].fillna(1.0)
            bio["biomass_deduction"] = bio["value"] - bio["biomass_retained"]
            # Assemble long format
            pieces = [
                bio[[group_by, "biomass_retained"]].rename(columns={"biomass_retained": "value"}).assign(component="Biomass (retained)"),
                bio[[group_by, "biomass_deduction"]].rename(columns={"biomass_deduction": "value"}).assign(component="Benchmark deduction"),
                comp[comp["metric"] == "Soil organic carbon (tCO2e)"].rename(columns={"metric": "component"})[[group_by, "component", "value"]],
                comp[comp["metric"].isin(EMISSIONS_METRICS + [LEAKAGE_METRIC])].rename(columns={"metric": "component"})[[group_by, "component", "value"]],
            ]
            long = pd.concat(pieces, ignore_index=True)
            # Sign: emissions, leakage, benchmark deduction negative
            neg = set(EMISSIONS_METRICS + [LEAKAGE_METRIC, "Benchmark deduction"])
            long["signed_value"] = long.apply(lambda r: -r["value"] if r["component"] in neg else r["value"], axis=1)
            figvm = px.bar(long, x=group_by, y="signed_value", color="component", barmode="relative", labels={"signed_value": "Net tCO₂e"})
            figvm.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Component")
            st.plotly_chart(figvm, use_container_width=True)

    with tab3:
        st.markdown("#### CCB: Climate, Community, Biodiversity")
        # Generate compact synthetic indicators constrained by selected sites
        sel_sites = flt.get("sites", [])
        rng = np.random.default_rng(123)
        def _make_rows(indicators: List[Tuple[str, str, float, float, str]]) -> pd.DataFrame:
            rows = []
            for site in sel_sites:
                for name, unit, mean, std, level in indicators:
                    val = float(np.clip(rng.normal(mean, std), 0, None))
                    unc = float(np.clip(rng.normal(10, 3), 2, 30))
                    rows.append({"Site": site, "Reporting level": level, "Indicator": name, "Value": round(val, 2), "Unit": unit, "Uncertainty (%)": round(unc, 1), "Sample size": int(rng.integers(20, 80))})
            return pd.DataFrame(rows)

        sub1, sub2, sub3 = st.tabs(["Climate", "Community", "Biodiversity"])
        with sub1:
            inds = [("Soil fertility index", "index", 0.7, 0.1, "Site"), ("Land productivity", "t ha⁻¹ yr⁻¹", 1.8, 0.3, "Site"), ("Resilience score", "0–1", 0.6, 0.1, "Site")]
            dfc = _make_rows(inds)
            # KPI mini cards
            c1, c2, c3 = st.columns(3)
            if not dfc.empty:
                _kpi_card("Soil fertility (mean)", f"{dfc['Value'].iloc[0]:.2f}")
                with c2: _kpi_card("Productivity (mean)", f"{dfc[dfc['Indicator']=='Land productivity']['Value'].mean():.2f}")
                with c3: _kpi_card("Resilience (mean)", f"{dfc[dfc['Indicator']=='Resilience score']['Value'].mean():.2f}")
            st.dataframe(dfc, use_container_width=True)
        with sub2:
            inds = [("Food-secure households", "%", 70, 8, "Village"), ("Crop yield", "t ha⁻¹", 2.0, 0.4, "Village"), ("Livelihood satisfaction", "%", 75, 7, "Village")]
            dcom = _make_rows(inds)
            st.dataframe(dcom, use_container_width=True)
        with sub3:
            inds = [("Species richness", "count", 25, 6, "Stratum"), ("Native vegetation cover", "%", 65, 10, "Stratum"), ("Habitat condition", "0–1", 0.6, 0.1, "Stratum")]
            dbio = _make_rows(inds)
            st.dataframe(dbio, use_container_width=True)


def page_overview(data: Dict[str, object], flt: dict) -> None:
    st.title("Overview")
    st.caption("Aggregated and time-series views across methodologies, sites and strata.")

    baseline_df: pd.DataFrame = data["baseline"]
    monitoring_df: pd.DataFrame = data["monitoring"]
    meta = data.get("metadata", {})

    mon = _apply_common_filters(monitoring_df, flt)
    if flt["periods"]:
        mon = mon[mon["period"].isin(flt["periods"])].copy()

    base = _apply_common_filters(baseline_df, flt)

    # If user selects only CCB, show CCB overview
    if flt.get("methodologies") == ["CCB"]:
        st.markdown("### CCB Overview")
        rng = np.random.default_rng(42)
        sites = flt.get("sites", [])
        def ccb_rows(indicators):
            rows = []
            for site in sites:
                for name, unit, mean, std in indicators:
                    val = float(np.clip(rng.normal(mean, std), 0, None))
                    rows.append({"Site": site, "Indicator": name, "Value": round(val, 2), "Unit": unit})
            return pd.DataFrame(rows)
        inds = [
            ("Species richness index", "index (0–1)", 0.6, 0.1),
            ("Native vegetation cover", "%", 65, 10),
            ("Habitat condition index", "index (0–1)", 0.6, 0.1),
        ]
        cdf = ccb_rows(inds)
        # Tooltips/explanations
        st.info("Indicator notes: Species richness and habitat condition are indices (0–1). Native vegetation cover is percentage area with native vegetation.")
        # KPIs
        c1, c2, c3 = st.columns(3)
        with c1: _kpi_card("Species richness (mean)", f"{cdf[cdf['Indicator']=='Species richness index']['Value'].mean():.2f}")
        with c2: _kpi_card("Native vegetation cover (mean)", f"{cdf[cdf['Indicator']=='Native vegetation cover']['Value'].mean():.1f}%")
        with c3: _kpi_card("Habitat condition (mean)", f"{cdf[cdf['Indicator']=='Habitat condition index']['Value'].mean():.2f}")
        st.dataframe(cdf, use_container_width=True)
        st.caption("Note: CCB indicators are displayed as indices or percentages with explanatory notes.")
        return

    # Summary cards (2 rows x 3 columns for symmetry)
    row1 = st.columns(3)
    row2 = st.columns(3)

    total_vcus = f"{mon['tco2e_this_period'].sum():,.0f} tCO₂e" if not mon.empty else "—"
    trees_af = f"{int(mon['agroforestry_trees_planted'].sum()):,}" if not mon.empty and 'agroforestry_trees_planted' in mon.columns else "—"
    trees_re = f"{int(mon['reforestation_trees_planted'].sum()):,}" if not mon.empty and 'reforestation_trees_planted' in mon.columns else "—"
    hectares_reforested = f"{mon['hectares_reforested'].sum():,.1f} ha" if not mon.empty and 'hectares_reforested' in mon.columns else "—"
    hectares_ag = f"{mon['hectares_ag_improved'].sum():,.1f} ha" if not mon.empty and 'hectares_ag_improved' in mon.columns else "—"
    hectares_af = f"{mon['hectares_agroforestry'].sum():,.1f} ha" if not mon.empty and 'hectares_agroforestry' in mon.columns else "—"

    with row1[0]:
        _kpi_card("Total Verified Carbon Units (VCUs)", total_vcus)
    with row1[1]:
        _kpi_card("Agroforestry trees planted", trees_af)
    with row1[2]:
        _kpi_card("Reforestation trees planted", trees_re)

    with row2[0]:
        _kpi_card("Hectares reforested", hectares_reforested)
    with row2[1]:
        _kpi_card("Area under IALM", hectares_ag)
    with row2[2]:
        _kpi_card("Area under agroforestry", hectares_af)

    st.markdown("### Cumulative sequestration vs baseline")
    if mon.empty:
        _render_empty_message("No monitoring data for the selected filters.")
    else:
        trend = mon.groupby(["date", "methodology"], as_index=False)[["tco2e_this_period", "uncertainty_percent"]].mean()
        # Lines only (bands removed per feedback)
        import plotly.graph_objects as go
        fig = go.Figure()
        for meth, d in trend.groupby("methodology"):
            d = d.sort_values("date")
            y = d["tco2e_this_period"].values
            x = d["date"].values
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=meth))
            # Baseline reference (net baseline: AGB+BGB+SOC - emissions - leakage)
            base_m = base[base["methodology"] == meth]
            pos = base_m[base_m["metric"].isin(["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"])]["value"].sum()
            neg = base_m[base_m["metric"].isin(EMISSIONS_METRICS + [LEAKAGE_METRIC])]["value"].sum()
            net_baseline = pos - neg
            if len(x):
                fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[net_baseline, net_baseline], mode="lines", name=f"{meth} baseline", line=dict(dash="dash")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="tCO₂e", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Contributions and highlights")
    pie_col1, pie_col2, pie_col3, pie_col4 = st.columns([1, 1, 1, 2])
    with pie_col1:
        by_meth = mon.groupby("methodology", as_index=False)["tco2e_this_period"].sum()
        st.plotly_chart(px.pie(by_meth, names="methodology", values="tco2e_this_period", hole=0.5, title="% by methodology"), use_container_width=True)
    with pie_col2:
        pools = base[base["metric"].isin(["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"])].groupby("metric", as_index=False)["value"].sum()
        if pools.empty:
            _render_empty_message("No pool data for current selection.")
        else:
            st.plotly_chart(px.pie(pools, names="metric", values="value", hole=0.5, title="% by carbon pool (AGB, BGB, SOC)"), use_container_width=True)
    with pie_col3:
        if mon.empty:
            _render_empty_message("No stratum data for current selection.")
        else:
            by_stratum = mon.groupby("stratum", as_index=False)["tco2e_this_period"].sum()
            st.plotly_chart(px.pie(by_stratum, names="stratum", values="tco2e_this_period", hole=0.5, title="% by stratum"), use_container_width=True)
    with pie_col4:
        # Top sites by cumulative (latest available)
        latest = mon.sort_values("date").groupby(["site"], as_index=False).last()[["site", "cumulative_tco2e"]]
        if latest.empty or "cumulative_tco2e" not in latest.columns:
            _render_empty_message("No cumulative data by site.")
        else:
            top_sites = latest.sort_values("cumulative_tco2e", ascending=False).head(5)
            st.markdown("#### Top sites by cumulative tCO₂e")
            st.dataframe(top_sites.rename(columns={"cumulative_tco2e": "Cumulative tCO₂e"}), use_container_width=True)

    st.markdown("### Trees per species")
    species_df = data.get("species", pd.DataFrame())
    if species_df.empty:
        _render_empty_message("No species data available.")
    else:
        # Filter species by current project/version/sites
        sdf = species_df.copy()
        if "project" in sdf.columns:
            sdf = sdf[sdf["project"] == flt["project"]]
        if flt.get("sites"):
            sdf = sdf[sdf["site"].isin(flt["sites"])]
        tabs_sp = st.tabs(["Agroforestry", "Reforestation"])
        with tabs_sp[0]:
            af = sdf[sdf["category"] == "Agroforestry"].groupby("species", as_index=False)["trees_planted"].sum().sort_values("trees_planted", ascending=False)
            st.plotly_chart(px.bar(af, x="species", y="trees_planted", title="Agroforestry: Trees per species"), use_container_width=True)
            st.dataframe(af, use_container_width=True)
        with tabs_sp[1]:
            re = sdf[sdf["category"] == "Reforestation"].groupby("species", as_index=False)["trees_planted"].sum().sort_values("trees_planted", ascending=False)
            st.plotly_chart(px.bar(re, x="species", y="trees_planted", title="Reforestation: Trees per species"), use_container_width=True)
            st.dataframe(re, use_container_width=True)

    st.markdown("### Verified Carbon Units tables")
    st.caption("Note: 1 VCU = 1 tCO₂e")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### Verified Carbon Units (VM0042)")
        vm42_base = base[base["methodology"] == "VM0042"]
        vm42_biomass = vm42_base[vm42_base["metric"] == "Biomass (tCO2e)"]["value"].sum()
        vm42_soc = vm42_base[vm42_base["metric"] == "Soil organic carbon (tCO2e)"]["value"].sum()
        vm42_total = vm42_biomass + vm42_soc
        st.table(pd.DataFrame({"Metric": ["Biomass", "Soil Organic Carbon", "Total"], "Value (tCO₂e)": [vm42_biomass, vm42_soc, vm42_total]}))
    with t2:
        st.markdown("#### Verified Carbon Units (VM0047)")
        vm47_base = base[base["methodology"] == "VM0047"]
        vm47_biomass = vm47_base[vm47_base["metric"] == "Biomass (tCO2e)"]["value"].sum()
        vm47_soc = vm47_base[vm47_base["metric"] == "Soil organic carbon (tCO2e)"]["value"].sum()
        vm47_total = vm47_biomass + vm47_soc
        st.table(pd.DataFrame({"Metric": ["Biomass", "Soil Organic Carbon", "Total"], "Value (tCO₂e)": [vm47_biomass, vm47_soc, vm47_total]}))

    st.markdown("### Dataset metadata")
    st.info(
        f"Version: {meta.get('current_dataset_version', flt['dataset_version'])}  •  "
        f"Last update: {meta.get('last_update', '—')}  •  "
        f"Analyst: {meta.get('responsible_analyst', '—')}"
    )


def page_monitoring_comparison(data: Dict[str, object], flt: dict) -> None:
    st.title("Monitoring Period Comparison")
    st.caption("Side-by-side baseline vs monitoring periods for sequestration, uncertainty and sample counts.")

    baseline_df: pd.DataFrame = data["baseline"]
    monitoring_df: pd.DataFrame = data["monitoring"]
    base = _apply_common_filters(baseline_df, flt)
    mon = _apply_common_filters(monitoring_df, flt)
    if flt["periods"]:
        mon = mon[mon["period"].isin(flt["periods"])].copy()

    # If CCB selected, show indicator trends
    if flt.get("methodologies") == ["CCB"]:
        st.markdown("### CCB indicators over time")
        periods_sorted = sorted(mon["period"].dropna().unique().tolist()) if not mon.empty else ["Time 1", "Time 2", "Time 3"]
        # Synthetic trends per indicator
        rng = np.random.default_rng(24)
        df_ccb = pd.DataFrame({
            "period": np.repeat(periods_sorted, 3),
            "indicator": ["Species richness index", "Native vegetation cover", "Habitat condition index"] * len(periods_sorted),
            "value": np.concatenate([
                np.clip(rng.normal(0.6, 0.05, len(periods_sorted)), 0, 1),
                np.clip(rng.normal(65, 5, len(periods_sorted)), 0, 100),
                np.clip(rng.normal(0.6, 0.05, len(periods_sorted)), 0, 1),
            ]),
            "unit": ["index (0–1)"] * len(periods_sorted) + ["%"] * len(periods_sorted) + ["index (0–1)"] * len(periods_sorted),
        })
        fig_ccb = px.line(df_ccb, x="period", y="value", color="indicator", markers=True)
        fig_ccb.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_ccb, use_container_width=True)
        st.caption("Notes: Species richness and habitat condition are index scores (0–1). Native vegetation cover is percentage cover.")
        return

    if mon.empty:
        _render_empty_message("No monitoring data for selected filters.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("#### Sequestration per period")
        fig = px.bar(
            mon,
            x="period",
            y="tco2e_this_period",
            color="methodology",
            barmode="group",
            hover_data={"run_id": True, "date": True, "tco2e_this_period": ":,.0f"},
            labels={"tco2e_this_period": "tCO₂e"},
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Cumulative removals")
        cum = mon.groupby(["period", "methodology"], as_index=False)["cumulative_tco2e"].max()
        fig2 = px.line(cum, x="period", y="cumulative_tco2e", color="methodology", markers=True)
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Uncertainty and sample counts")
    cols = st.columns(2)
    with cols[0]:
        fig3 = px.box(mon, x="period", y="uncertainty_percent", color="methodology", points="all")
        fig3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    with cols[1]:
        # Uncertainty by carbon pool (synthetic allocation from overall monitoring uncertainty)
        pools = ["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"]
        up = []
        for _, row in mon.iterrows():
            for p in pools:
                up.append({"period": row["period"], "methodology": row["methodology"], "pool": p, "uncertainty_percent": row["uncertainty_percent"]})
        up_df = pd.DataFrame(up)
        fig4 = px.box(up_df, x="period", y="uncertainty_percent", color="pool", points=False, title="Uncertainty by carbon pool")
        fig4.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)


def page_map(data: Dict[str, object], flt: dict) -> None:
    st.title("Map")
    st.caption("Verified boundaries, stratum polygons and sampling plots.")

    plots = data["plots_geojson"]
    boundaries = data["boundaries_geojson"]
    strata = data["strata_geojson"]

    # Filter plots by sidebar choices
    filtered_features: List[dict] = []
    for feat in plots.get("features", []):
        props = feat.get("properties", {})
        if props.get("project") != flt.get("project"):
            continue
        if props.get("methodology") not in flt["methodologies"]:
            continue
        if props.get("dataset") not in flt["plot_dataset"]:
            continue
        if props.get("completion_status") not in flt["plot_completion"]:
            continue
        if props.get("dataset_version") != flt["dataset_version"]:
            continue
        if props.get("site") not in flt["sites"]:
            continue
        if props.get("stratum") not in flt["strata"]:
            continue
        filtered_features.append(feat)

    plots_filtered = {"type": "FeatureCollection", "features": filtered_features}

    # Center map
    def _center_of_geojson(gj: dict) -> Tuple[float, float]:
        lons, lats = [], []
        for f in gj.get("features", []):
            geom = f.get("geometry", {})
            if geom.get("type") == "Point":
                lon, lat = geom.get("coordinates", [0, 0])
                lons.append(lon)
                lats.append(lat)
            elif geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[[]]])[0]
                for lon, lat in coords:
                    lons.append(lon)
                    lats.append(lat)
        if not lons or not lats:
            return (16.0, -12.5)
        return (float(np.mean(lons)), float(np.mean(lats)))

    center_lon, center_lat = _center_of_geojson(boundaries)
    if filtered_features:
        center_lon, center_lat = _center_of_geojson(plots_filtered)

    tooltip_html = """
    <div style='font-size:13px;'>
      <div><b>Site</b>: {site}</div>
      <div><b>Stratum</b>: {stratum}</div>
      <div><b>Methodology</b>: {methodology}</div>
      <div><b>Dataset</b>: {dataset}</div>
      <div><b>Component</b>: {component}</div>
      <div><b>Version</b>: {dataset_version}</div>
      <div><b>Completion</b>: {completion_status}</div>
      <div><b>Date</b>: {collection_date}</div>
      <div><b>Value</b>: {value}</div>
      <div><b>Uncertainty</b>: {uncertainty_percent}%</div>
    </div>
    """

    completion_to_color = {"Complete": [16, 185, 129], "Partial": [245, 158, 11], "Pending": [239, 68, 68]}

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v11",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6, bearing=0, pitch=0),
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                data=boundaries,
                stroked=True,
                filled=False,
                get_line_color=[52, 211, 153],
                line_width_min_pixels=2,
            ),
            pdk.Layer(
                "GeoJsonLayer",
                data=strata,
                stroked=True,
                filled=True,
                get_fill_color=[59, 130, 246, 40],
                get_line_color=[59, 130, 246],
                line_width_min_pixels=1,
            ),
            pdk.Layer(
                "GeoJsonLayer",
                data=plots_filtered,
                point_type="circle",
                get_fill_color="[properties.completion_status == 'Complete' ? 16 : (properties.completion_status == 'Partial' ? 245 : 239), properties.completion_status == 'Complete' ? 185 : (properties.completion_status == 'Partial' ? 158 : 68), properties.completion_status == 'Complete' ? 129 : (properties.completion_status == 'Partial' ? 11 : 68), 180]",
                get_radius=60,
                pickable=True,
            ),
        ],
        tooltip={"html": tooltip_html, "style": {"backgroundColor": "white", "color": "#111", "fontSize": "12px"}},
    )

    st.pydeck_chart(deck, use_container_width=True)

    st.markdown(
        """
        <div>
          <span class='legend-pill' style='background:#ecfdf5;'>Complete</span>
          <span class='legend-pill' style='background:#fff7ed;'>Partial</span>
          <span class='legend-pill' style='background:#fef2f2;'>Pending</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Right-side summary panel for visible features
    with st.expander("Map extent summary", expanded=True):
        feats = filtered_features
        if not feats:
            _render_empty_message("No sampling plots in current filters.")
        else:
            total = len(feats)
            completed = sum(1 for f in feats if f["properties"].get("completion_status") == "Complete")
            pct = 100 * completed / max(total, 1)
            vals = [f["properties"].get("value") for f in feats if isinstance(f["properties"].get("value"), (int, float))]
            uncs = [f["properties"].get("uncertainty_percent") for f in feats if isinstance(f["properties"].get("uncertainty_percent"), (int, float))]
            mean_val = np.mean(vals) if vals else np.nan
            mean_unc = np.mean(uncs) if uncs else np.nan
            meta_ver = list({f["properties"].get("dataset_version") for f in feats}) or ["—"]
            st.columns(4)[0].metric("Visible plots", f"{total:,}")
            st.columns(4)[1].metric("% completed", f"{pct:.1f}%")
            st.columns(4)[2].metric("Mean value", f"{mean_val:,.1f}" if not np.isnan(mean_val) else "—")
            st.columns(4)[3].metric("Mean uncertainty", f"{mean_unc:.1f}%" if not np.isnan(mean_unc) else "—")
            st.caption(f"Dataset versions: {', '.join(map(str, meta_ver))}")
            # Export visible features
            csv_rows = []
            for f in feats:
                row = {"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1]}
                row.update(f["properties"])
                csv_rows.append(row)
            df_vis = pd.DataFrame(csv_rows)
            st.download_button("Download visible (GeoJSON)", data=json.dumps({"type": "FeatureCollection", "features": feats}).encode("utf-8"), file_name="visible.geojson", mime="application/geo+json")
            st.download_button("Download visible (CSV)", data=_to_csv_bytes(df_vis), file_name="visible.csv", mime="text/csv")


def page_data_explorer(data: Dict[str, object], flt: dict) -> None:
    st.title("Data Explorer")
    st.caption("Tabular views with metadata, filtering and full-text search.")

    base = _apply_common_filters(data["baseline"], flt)
    mon = _apply_common_filters(data["monitoring"], flt)
    qc = data["qc"].copy()
    qc = qc[(qc["methodology"].isin(flt["methodologies"])) & (qc["dataset_version"] == flt["dataset_version"])]

    tabs = st.tabs(["Baseline", "Monitoring", "QC"])
    with tabs[0]:
        q = st.text_input("Search baseline", "")
        df = base.copy()
        if q:
            ql = q.lower()
            df = df[df.apply(lambda r: ql in str(r.values).lower(), axis=1)]
        st.dataframe(df, use_container_width=True)
    with tabs[1]:
        q = st.text_input("Search monitoring", "")
        df = mon.copy()
        if q:
            ql = q.lower()
            df = df[df.apply(lambda r: ql in str(r.values).lower(), axis=1)]
        st.dataframe(df, use_container_width=True)
    with tabs[2]:
        q = st.text_input("Search QC", "")
        df = qc.copy()
        if q:
            ql = q.lower()
            df = df[df.apply(lambda r: ql in str(r.values).lower(), axis=1)]
        st.dataframe(df, use_container_width=True)


def page_qc_status(data: Dict[str, object], flt: dict) -> None:
    st.title("QC Status")
    st.caption("Summary of QC outcomes with filters and alerts.")

    qc = data["qc"].copy()
    # Filters
    dataset = st.selectbox("Dataset", ["baseline", "monitoring", "All"], index=2)
    reviewer_opts = ["All"] + sorted(qc["reviewer"].dropna().unique().tolist())
    reviewer = st.selectbox("Reviewer", reviewer_opts, index=0)

    mask = (qc["methodology"].isin(flt["methodologies"])) & (qc["dataset_version"] == flt["dataset_version"])  # type: ignore
    if dataset != "All":
        mask &= qc["dataset"] == dataset
    if reviewer != "All":
        mask &= qc["reviewer"] == reviewer
    df = qc[mask].copy()

    if df.empty:
        _render_empty_message("No QC results for selected filters.")
        return

    total = int(df["total_records"].sum())
    failed = int(df["failed_records"].sum())
    rate = 100.0 * failed / max(total, 1)
    _kpi_card("Records tested", f"{total:,}")
    _kpi_card("Failed records", f"{failed:,}", delta=f"{rate:.1f}% fail rate")

    high_sev = df[df["severity"].str.lower() == "high"]
    if not high_sev.empty:
        st.markdown(
            f"<div class='alert-banner'><b>High-severity QC failures:</b> {len(high_sev)} run(s) require attention before approval/publication.</div>",
            unsafe_allow_html=True,
        )

    agg = df.groupby(["dataset", "methodology"], as_index=False)[["total_records", "failed_records"]].sum()
    agg["fail_rate_%"] = 100 * agg["failed_records"] / agg["total_records"].clip(lower=1)
    st.dataframe(agg, use_container_width=True)


def page_version_history(data: Dict[str, object], flt: dict) -> None:
    st.title("Version History / Audit")
    st.caption("Change logs of datasets, QC runs and parameter updates.")

    qc = data["qc"].copy()
    qc = qc[(qc["methodology"].isin(flt["methodologies"])) & (qc["dataset_version"] == flt["dataset_version"])].copy()
    if qc.empty:
        _render_empty_message("No change logs for selected filters.")
        return

    logs = qc.rename(columns={"date": "timestamp"})[
        ["timestamp", "dataset", "methodology", "dataset_version", "run_id", "reviewer", "total_records", "failed_records", "severity"]
    ].sort_values("timestamp")
    st.dataframe(logs, use_container_width=True)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        ds = st.multiselect("Filter dataset", sorted(qc["dataset"].unique()), default=sorted(qc["dataset"].unique()))
    with col2:
        per = st.multiselect("Filter severity", sorted(qc["severity"].unique()), default=sorted(qc["severity"].unique()))
    with col3:
        meth = st.multiselect("Filter methodology", METHODOLOGIES, default=flt["methodologies"])

    mask = logs["dataset"].isin(ds) & logs["severity"].isin(per) & logs["methodology"].isin(meth)
    st.dataframe(logs[mask], use_container_width=True)


def page_verification_dashboard(data: Dict[str, object], flt: dict) -> None:
    st.title("Verification Dashboard")
    st.caption("Read-only access to key evidence, QC summaries and sampling coverage.")

    monitoring_df: pd.DataFrame = data["monitoring"]
    mon = _apply_common_filters(monitoring_df, flt)
    if mon.empty:
        _render_empty_message("No monitoring data for selected filters.")
        return

    coverage = 100 * mon["plots_measured"].sum() / max(mon["plots_measured"].sum() + 1, 1)
    _kpi_card("Sampling coverage (proxy)", f"{min(coverage, 100):.1f}%", help_text="Proportion of plots measured across selections")

    st.markdown("### QC summaries")
    qc = data["qc"].copy()
    qc = qc[(qc["methodology"].isin(flt["methodologies"])) & (qc["dataset_version"] == flt["dataset_version"])].copy()
    if qc.empty:
        _render_empty_message("No QC summaries available.")
    else:
        st.dataframe(
            qc[["dataset", "run_id", "reviewer", "date", "total_records", "failed_records", "severity"]].sort_values("date", ascending=False),
            use_container_width=True,
        )

    st.markdown("### Supporting documents")
    st.info("Place evidence files in ./docs for download (PDFs, images, spreadsheets). This portal is read-only.")


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _zip_bytes(files: List[Tuple[str, bytes]]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files:
            zf.writestr(name, content)
    mem.seek(0)
    return mem.read()


def page_downloads(data: Dict[str, object], flt: dict) -> None:
    st.title("Downloads")
    st.caption("Standardised export packages for indicators, QC and spatial data.")

    base = _apply_common_filters(data["baseline"], flt)
    mon = _apply_common_filters(data["monitoring"], flt)
    qc = data["qc"].copy()
    qc = qc[(qc["methodology"].isin(flt["methodologies"])) & (qc["dataset_version"] == flt["dataset_version"])].copy()

    st.markdown("#### Raw datasets")
    colA, colB = st.columns(2)
    with colA:
        st.download_button("Download baseline (CSV)", data=_to_csv_bytes(base), file_name="baseline_filtered.csv", mime="text/csv", disabled=base.empty)
    with colB:
        st.download_button("Download monitoring (CSV)", data=_to_csv_bytes(mon), file_name="monitoring_filtered.csv", mime="text/csv", disabled=mon.empty)

    st.markdown("#### Previous versions")
    # Allow export for any dataset version, not only current filters
    all_versions = sorted(data["baseline"]["dataset_version"].dropna().unique().tolist())
    sel_ver = st.selectbox("Choose dataset version to export", all_versions, index=max(0, len(all_versions) - 1))
    base_ver = data["baseline"][data["baseline"]["dataset_version"] == sel_ver]
    mon_ver = data["monitoring"][data["monitoring"]["dataset_version"] == sel_ver]
    files_prev = [
        (f"baseline_{sel_ver}.csv", _to_csv_bytes(base_ver)),
        (f"monitoring_{sel_ver}.csv", _to_csv_bytes(mon_ver)),
    ]
    st.download_button("Download selected version (ZIP)", data=_zip_bytes(files_prev), file_name=f"datasets_{sel_ver}.zip", mime="application/zip")

    st.markdown("#### Indicator summary – KPIs by period, stratum, methodology and dataset")
    if mon.empty:
        _render_empty_message("No monitoring data to export.")
    else:
        kpi = mon.groupby(["period", "methodology", "site", "stratum"], as_index=False).agg(
            tco2e=("tco2e_this_period", "sum"),
            cumulative_tco2e=("cumulative_tco2e", "max"),
            uncertainty_percent=("uncertainty_percent", "mean"),
            trees_planted=("trees_planted", "sum"),
            species_count=("species_count", "max"),
            plots_measured=("plots_measured", "sum"),
        )
        st.download_button("Download indicator summary (CSV)", data=_to_csv_bytes(kpi), file_name="indicator_summary.csv", mime="text/csv")

    st.markdown("#### QC summary – Aggregated QC checks by dataset and run")
    if qc.empty:
        _render_empty_message("No QC data to export.")
    else:
        st.download_button("Download QC summary (CSV)", data=_to_csv_bytes(qc), file_name="qc_summary.csv", mime="text/csv")

    st.markdown("#### Flagged records – All QC fails with rules and messages")
    if qc.empty:
        _render_empty_message("No QC fails in current selection.")
    else:
        fails = qc[qc["failed_records"] > 0].copy()
        st.download_button("Download flagged runs (CSV)", data=_to_csv_bytes(fails), file_name="qc_flagged_runs.csv", mime="text/csv")

    st.markdown("#### Spatial data – Boundaries and sampling points (GeoJSON)")
    # Filter plots as per map filters
    plots = data["plots_geojson"]
    filtered_features = []
    for feat in plots.get("features", []):
        p = feat.get("properties", {})
        if p.get("dataset_version") != flt["dataset_version"]:
            continue
        if p.get("methodology") not in flt["methodologies"]:
            continue
        if p.get("site") not in flt["sites"] or p.get("stratum") not in flt["strata"]:
            continue
        filtered_features.append(feat)
    plots_filtered = {"type": "FeatureCollection", "features": filtered_features}

    files = [
        ("boundaries.geojson", json.dumps(data["boundaries_geojson"]).encode("utf-8")),
        ("strata.geojson", json.dumps(data["strata_geojson"]).encode("utf-8")),
        ("sampling_points.geojson", json.dumps(plots_filtered).encode("utf-8")),
    ]
    st.download_button(
        "Download spatial data (ZIP)",
        data=_zip_bytes(files),
        file_name="spatial_data.zip",
        mime="application/zip",
    )


# ---------- Navigation ----------
PAGES = {
    "Overview": page_overview,
    "Baseline": page_baseline,
    "Monitoring": page_monitoring_comparison,
    "Map": page_map,
    "Downloads": page_downloads,
    "Administration": lambda d, f: (st.title("Administration"), st.info("Read-only portal. Admin features are disabled in this environment.")),
}


def main() -> None:
    st.sidebar.title("MRV Portal")
    st.sidebar.caption("Read-only visualisation of verified MRV data")

    # Move navigation to the top of the sidebar for visibility
    page = st.sidebar.radio("Navigation", list(PAGES.keys()), index=0)

    data = load_data()
    # Sidebar hosts only advanced options
    advanced = sidebar_advanced_filters(data)
    # Top sticky quick filter bar for primary filters
    filters_top = quick_filter_bar(data)
    # Merge filters
    filters = {**filters_top, **advanced}

    st.markdown(
        """
        > This portal provides secure, read-only visualisation of verified MRV datasets, quality-control results and audit trails. 
        > Use the Quick Filter Bar at the top to refine version, methodology, site and stratum.
        > Advanced filters for sampling plots are available in the sidebar.
        """
    )

    # Render the chosen page
    try:
        PAGES[page](data, filters)
    except Exception as exc:  # Controlled error surfacing for clarity
        st.error("An unexpected error occurred while rendering the page. Please adjust filters or try again.")
        st.exception(exc)


if __name__ == "__main__":
    main()


