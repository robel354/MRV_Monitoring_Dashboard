 
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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import shapefile  # pyshp
from pyproj import CRS, Transformer
from typing import Any
import os
import tempfile


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


# ---------- Mapbox token (for basemap) ----------
# Use Streamlit secrets first, then environment variables. Warn if missing.
MAPBOX_ENABLED = False
try:
    token = ""
    if "MAPBOX_TOKEN" in st.secrets:
        token = st.secrets["MAPBOX_TOKEN"] or ""
    if not token:
        token = os.environ.get("MAPBOX_API_KEY", "") or os.environ.get("MAPBOX_TOKEN", "")
    if token:
        pdk.settings.mapbox_api_key = token  # enables basemap
        MAPBOX_ENABLED = True
    else:
        st.warning("Map basemap token is not configured. Add MAPBOX_TOKEN to .streamlit/secrets.toml for full map rendering.")
except Exception:
    pass

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


def _transform_coords(coords, transformer):
    # Recursively transform coordinates (supports Polygon/MultiPolygon)
    if isinstance(coords[0], (float, int)):
        x, y = coords[0], coords[1]
        lon, lat = transformer.transform(x, y)
        return [lon, lat]
    return [_transform_coords(c, transformer) for c in coords]


def _maybe_read_shapefile_folder(folder_path: str) -> Optional[dict]:
    try:
        import os
        if not os.path.isdir(folder_path):
            return None
        shp_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".shp")]
        if not shp_files:
            return None
        shp_path = os.path.join(folder_path, shp_files[0])
        r = shapefile.Reader(shp_path)
        fields = [f[0] for f in r.fields if f[0] != "DeletionFlag"]
        # Try read projection; if not WGS84, transform to EPSG:4326
        transformer = None
        prj_path = shp_path[:-4] + ".prj"
        try:
            if os.path.exists(prj_path):
                with open(prj_path, "r") as f:
                    src = CRS.from_wkt(f.read())
                if not src.equals(CRS.from_epsg(4326)):
                    transformer = Transformer.from_crs(src, CRS.from_epsg(4326), always_xy=True)
        except Exception:
            transformer = None
        feats = []
        for sr in r.iterShapeRecords():
            geom = sr.shape.__geo_interface__
            if transformer is not None and "coordinates" in geom:
                geom["coordinates"] = _transform_coords(geom["coordinates"], transformer)
            # Sanitize attribute types for JSON
            props = {}
            for i in range(len(fields)):
                k = fields[i]
                v = sr.record[i]
                try:
                    if hasattr(v, "item"):
                        v = v.item()
                    elif isinstance(v, bytes):
                        v = v.decode("utf-8", "ignore")
                    elif isinstance(v, (set, tuple)):
                        v = list(v)
                except Exception:
                    v = str(v)
                # Coerce non-JSON-safe objects to strings
                if not isinstance(v, (str, int, float, bool, type(None), list, dict)):
                    v = str(v)
                props[k] = v
            props["popup"] = "<br>".join([f"<b>{k}</b>: {props[k]}" for k in list(props.keys())[:12]])
            feats.append({"type": "Feature", "geometry": geom, "properties": props})
        return {"type": "FeatureCollection", "features": feats}
    except Exception:
        return None


def _json_safe(value):
    """Recursively convert common non-JSON-safe python/numpy/shapefile types to JSON-safe primitives."""
    try:
        import numpy as _np  # local import to avoid polluting globals
        if isinstance(value, _np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "ignore")
        except Exception:
            return str(value)
    if isinstance(value, (tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    # Basic primitives pass through
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Fallback to string
    return str(value)

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

        # Optionally load local shapefiles if present (prefer relative paths for production compatibility)
        custom_polygons: List[Tuple[str, dict]] = []
        cwd = os.getcwd()
        candidate_paths = [
            os.path.join(cwd, "Luxiha_Dashboard"),
            r"C:\Users\RobelBerhanu\Desktop\MRV_Visuals_Angola\Luxiha_Dashboard",
        ]
        for path in candidate_paths:
            gj = _maybe_read_shapefile_folder(path)
            if gj:
                name = os.path.basename(path.rstrip("\\/"))
                custom_polygons.append((name, gj))

        return {
            "baseline": baseline_df,
            "monitoring": monitoring_df,
            "qc": qc_df,
            "plots_geojson": plots_geojson or {"type": "FeatureCollection", "features": []},
            "boundaries_geojson": boundaries_geojson or {"type": "FeatureCollection", "features": []},
            "strata_geojson": strata_geojson or {"type": "FeatureCollection", "features": []},
            "metadata": metadata,
            "custom_polygons": custom_polygons,
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

    # Monitoring records across baseline T0 and subsequent monitoring rounds T1, T2, T3 with explicit years
    baseline_year = 2025
    monitoring_years = [2030, 2032, 2034]  # demo spacing; real years can be mapped from data
    periods = [f"{baseline_year} (T0)"] + [f"{y} (T{i})" for i, y in enumerate(monitoring_years, start=1)]
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
                            # Derive a representative date from the period label's year (mid-year for plotting consistency)
                            "date": datetime(int(period.split()[0]), 7, 1).date().isoformat(),
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

    # Load local shapefiles (relative-first) so polygons show without upload
    custom_polygons: List[Tuple[str, dict]] = []
    cwd = os.getcwd()
    candidate_paths = [
        os.path.join(cwd, "Luxiha_Dashboard"),
        r"C:\Users\RobelBerhanu\Desktop\MRV_Visuals_Angola\Luxiha_Dashboard",
    ]
    for path in candidate_paths:
        gj = _maybe_read_shapefile_folder(path)
        if gj:
            name = os.path.basename(path.rstrip("\\/"))
            custom_polygons.append((name, gj))

    return {
        "baseline": baseline_df,
        "monitoring": monitoring_df,
        "species": species_df,
        "qc": qc_df,
        "plots_geojson": plots_geojson,
        "boundaries_geojson": boundaries_geojson,
        "strata_geojson": strata_geojson,
        "metadata": metadata,
        "custom_polygons": custom_polygons,
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

        if methodology == "VM0042":
            col1, col2 = st.columns(2)
        else:
            col1, col2, col3 = st.columns(3)
        v_agb, u_agb = metric_value("Aboveground biomass (AGB) (tCO2e)")
        v_bgb, u_bgb = metric_value("Belowground biomass (BGB) (tCO2e)")
        v_soc, u_soc = metric_value("Soil organic carbon (tCO2e)")
        v_foss, u_foss = metric_value("GHG Emissions — Fossil fuels (tCO2e)")
        v_burn, u_burn = metric_value("GHG Emissions — Biomass Burning (tCO2e)")
        v_nit, u_nit = metric_value("GHG Emissions — Nitrogen inputs to Soils (tCO2e)")
        v_leak, _ = metric_value(LEAKAGE_METRIC)

        with col1:
            _kpi_card("AGB", f"{v_agb:,.0f} tCO₂e", help_text=f"+ {u_agb:.1f}%")
        with col2:
            _kpi_card("BGB", f"{v_bgb:,.0f} tCO₂e", help_text=f"+ {u_bgb:.1f}%")
        if methodology != "VM0042":
            with col3:
                _kpi_card("SOC", f"{v_soc:,.0f} tCO₂e", help_text=f"+ {u_soc:.1f}%")

        # Charts: Baseline components by grouping
        group_by = "site"
        comp_keep = ["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)"]
        comp = df[df["metric"].isin(comp_keep)].groupby([group_by, "metric"], as_index=False)["value"].sum()
        fig = px.bar(comp, x=group_by, y="value", color="metric", barmode="group", labels={"value": "tCO₂e"})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Component")
        st.plotly_chart(fig, use_container_width=True)

        # Tables (column header filters via AgGrid; keep scrollbars by fixed height)
        st.markdown("#### Table 1 — Baseline carbon stocks")
        # Filter rows to match requested display: show only AGB/BGB
        t1_src = df.copy()
        t1_src = t1_src[t1_src["metric"].isin(["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)"])]
        t1_df = t1_src[["site", "stratum", "metric", "value", "uncertainty_percent", "sample_count"]].rename(
            columns={"metric": "Carbon pool", "value": "Value (tCO₂e)", "uncertainty_percent": "Uncertainty (+%)"}
        )
        t1_df["Value (tCO₂e)"] = t1_df["Value (tCO₂e)"].map(lambda x: float(f"{x:.2f}"))
        t1_df["Uncertainty (+%)"] = t1_df["Uncertainty (+%)"].map(lambda x: float(f"{x:.2f}"))
        g1 = GridOptionsBuilder.from_dataframe(t1_df)
        g1.configure_default_column(filter=True, sortable=True, resizable=True, floatingFilter=True)
        g1.configure_column("site", filter="agTextColumnFilter")
        g1.configure_column("stratum", filter="agTextColumnFilter")
        g1.configure_column("Carbon pool", filter="agTextColumnFilter")
        g1.configure_column("Value (tCO₂e)", filter="agNumberColumnFilter")
        g1.configure_column("Uncertainty (+%)", filter="agNumberColumnFilter")
        g1.configure_grid_options(animateRows=True)
        AgGrid(t1_df, gridOptions=g1.build(), theme="balham", update_mode=GridUpdateMode.NO_UPDATE, fit_columns_on_grid_load=True, height=420)

        # Table 2 — Project emissions and leakage (restored)
        st.markdown("#### Table 2 — Project emissions and leakage")
        t2_df = df[df["metric"].isin(EMISSIONS_METRICS + [LEAKAGE_METRIC])][
            ["site", "stratum", "metric", "value", "uncertainty_percent", "sample_count"]
        ].rename(
            columns={
                "metric": "Emission source",
                "value": "Value (tCO₂e)",
                "uncertainty_percent": "Uncertainty (+%)",
            }
        )
        t2_df["Value (tCO₂e)"] = t2_df["Value (tCO₂e)"].map(lambda x: float(f"{x:.2f}"))
        t2_df["Uncertainty (+%)"] = t2_df["Uncertainty (+%)"].map(lambda x: float(f"{x:.2f}"))
        g2 = GridOptionsBuilder.from_dataframe(t2_df)
        g2.configure_default_column(filter=True, sortable=True, resizable=True, floatingFilter=True)
        g2.configure_column("site", filter="agTextColumnFilter")
        g2.configure_column("stratum", filter="agTextColumnFilter")
        g2.configure_column("Emission source", filter="agTextColumnFilter")
        g2.configure_column("Value (tCO₂e)", filter="agNumberColumnFilter")
        g2.configure_column("Uncertainty (+%)", filter="agNumberColumnFilter")
        g2.configure_grid_options(animateRows=True)
        AgGrid(
            t2_df.sort_values(["site", "stratum", "Emission source"]),
            gridOptions=g2.build(),
            theme="balham",
            update_mode=GridUpdateMode.NO_UPDATE,
            fit_columns_on_grid_load=True,
            height=420,
        )

    with tab1:
        render_vm("VM0042")

    with tab2:
        df47 = base_all[base_all["methodology"] == "VM0047"].copy()
        if df47.empty:
            _render_empty_message("No baseline data for VM0047 under current filters.")
        else:
            # Show SOC only (split into Reforestation vs IALM & Agroforestry)
            soc_only = df47[df47["metric"] == "Soil organic carbon (tCO2e)"].copy()
            v_soc = float(soc_only["value"].sum())
            u_soc = float(soc_only["uncertainty_percent"].mean()) if not soc_only.empty else float("nan")
            col = st.columns(1)[0]
            with col:
                _kpi_card("SOC", f"{v_soc:,.0f} tCO₂e", help_text=f"+ {u_soc:.1f}%")

            group_by = "site"
            if soc_only.empty:
                comp = pd.DataFrame(columns=[group_by, "metric", "value"])
            else:
                site_soc = soc_only.groupby(group_by, as_index=False)["value"].sum()
                soc_re = site_soc.copy()
                soc_re["value"] = site_soc["value"] * 0.5
                soc_re["metric"] = "Soil organic carbon — Reforestation (tCO₂e)"
                soc_ialm = site_soc.copy()
                soc_ialm["value"] = site_soc["value"] - soc_re["value"]
                soc_ialm["metric"] = "Soil organic carbon — IALM & Agroforestry (tCO₂e)"
                comp = pd.concat([soc_re[[group_by, "metric", "value"]], soc_ialm[[group_by, "metric", "value"]]], ignore_index=True)
            fig = px.bar(comp, x=group_by, y="value", color="metric", barmode="group", labels={"value": "tCO₂e"})
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Component")
            st.plotly_chart(fig, use_container_width=True)

            # Table 1 — SOC only
            st.markdown("#### Table 1 — Baseline carbon stocks (SOC)")
            t1_df = soc_only[["site", "stratum", "metric", "value", "uncertainty_percent", "sample_count"]].rename(
                columns={"metric": "Carbon pool", "value": "Value (tCO₂e)", "uncertainty_percent": "Uncertainty (+%)"}
            )
            t1_df["Value (tCO₂e)"] = t1_df["Value (tCO₂e)"].map(lambda x: float(f"{x:.2f}"))
            t1_df["Uncertainty (+%)"] = t1_df["Uncertainty (+%)"].map(lambda x: float(f"{x:.2f}"))
            g1 = GridOptionsBuilder.from_dataframe(t1_df)
            g1.configure_default_column(filter=True, sortable=True, resizable=True, floatingFilter=True)
            g1.configure_column("site", filter="agTextColumnFilter")
            g1.configure_column("stratum", filter="agTextColumnFilter")
            g1.configure_column("Carbon pool", filter="agTextColumnFilter")
            g1.configure_column("Value (tCO₂e)", filter="agNumberColumnFilter")
            g1.configure_column("Uncertainty (+%)", filter="agNumberColumnFilter")
            g1.configure_grid_options(animateRows=True)
            AgGrid(t1_df, gridOptions=g1.build(), theme="balham", update_mode=GridUpdateMode.NO_UPDATE, fit_columns_on_grid_load=True, height=420)

            # Table 2 — Project emissions and leakage (unchanged)
            st.markdown("#### Table 2 — Project emissions and leakage")
            t2_df = df47[df47["metric"].isin(EMISSIONS_METRICS + [LEAKAGE_METRIC])][
                ["site", "stratum", "metric", "value", "uncertainty_percent", "sample_count"]
            ].rename(
                columns={
                    "metric": "Emission source",
                    "value": "Value (tCO₂e)",
                    "uncertainty_percent": "Uncertainty (+%)",
                }
            )
            t2_df["Value (tCO₂e)"] = t2_df["Value (tCO₂e)"].map(lambda x: float(f"{x:.2f}"))
            t2_df["Uncertainty (+%)"] = t2_df["Uncertainty (+%)"].map(lambda x: float(f"{x:.2f}"))
            g2 = GridOptionsBuilder.from_dataframe(t2_df)
            g2.configure_default_column(filter=True, sortable=True, resizable=True, floatingFilter=True)
            g2.configure_column("site", filter="agTextColumnFilter")
            g2.configure_column("stratum", filter="agTextColumnFilter")
            g2.configure_column("Emission source", filter="agTextColumnFilter")
            g2.configure_column("Value (tCO₂e)", filter="agNumberColumnFilter")
            g2.configure_column("Uncertainty (+%)", filter="agNumberColumnFilter")
            g2.configure_grid_options(animateRows=True)
            AgGrid(
                t2_df.sort_values(["site", "stratum", "Emission source"]),
                gridOptions=g2.build(),
                theme="balham",
                update_mode=GridUpdateMode.NO_UPDATE,
                fit_columns_on_grid_load=True,
                height=420,
            )

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
            # Only Soil fertility for Climate; remove productivity and resilience
            inds = [("Soil fertility index", "index", 0.7, 0.1, "Site")]
            dfc = _make_rows(inds)
            # Show nutrient status aligned in three columns
            st.markdown("##### Nutrient status")
            ns1, ns2, ns3 = st.columns(3)
            with ns1:
                st.metric("Nitrogen", "Low")
            with ns2:
                st.metric("Phosphorus", "Low")
            with ns3:
                st.metric("Potassium", "Low")
            # Table for Soil fertility measurements
            st.dataframe(dfc, use_container_width=True)
        with sub2:
            # Build Community tables as requested (synthetic demo values)
            sites = flt.get("sites", [])
            rng = np.random.default_rng(321)

            def _rows_for(metrics: List[Tuple[str, str]]) -> pd.DataFrame:
                rows = []
                for site in sites:
                    for metric, unit in metrics:
                        if unit == "%":
                            val = float(np.clip(rng.normal(65, 10), 0, 100))
                        elif unit == "kg per capita":
                            val = float(np.clip(rng.normal(220, 60), 10, None))
                        elif unit == "count":
                            val = int(np.clip(rng.normal(40, 15), 0, None))
                        elif unit == "ha":
                            val = float(np.clip(rng.normal(120, 40), 0, None))
                        elif unit == "score":
                            val = float(np.clip(rng.normal(4.5, 1.1), 0, 12))
                        else:  # generic mean count
                            val = float(np.clip(rng.normal(3.0, 0.8), 0, None))
                        rows.append({"Site": site, "Metric": metric, "Value": round(val, 2), "Unit": unit})
                return pd.DataFrame(rows)

            # 1) Sustainable agroforestry and climate‑resilient land use
            st.markdown("#### Sustainable agroforestry and climate‑resilient land use")
            landuse_metrics = [
                ("Mean number of crop types per household", "count"),
                ("Crop specific yield", "kg per capita"),
                ("% households reporting low crop productivity due to reduced soil health", "%"),
                ("% households using any existing land management practice", "%"),
                ("Mapped area under existing land-use practices", "ha"),
            ]
            df_landuse = _rows_for(landuse_metrics)
            st.dataframe(df_landuse, use_container_width=True)

            # 2) Food security
            st.markdown("#### Food security")
            food_metrics = [
                ("Average household dietary diversity score per village", "score"),
                ("% households reporting year-round food sufficiency", "%"),
                ("% households where ≥25% of food comes from own production", "%"),
            ]
            df_food = _rows_for(food_metrics)
            st.dataframe(df_food, use_container_width=True)

            # 3) Training
            st.markdown("#### Training")
            training_metrics = [
                ("Participants with prior training in agriculture or NRM", "count"),
                ("Women enrolled in training activities", "count"),
                ("% women enrolled in training activities", "%"),
                ("Women with knowledge/skills in agriculture or NRM", "count"),
                ("% women with knowledge/skills in agriculture or NRM", "%"),
                ("Mean number of ag/NRM techniques applied by women", "count"),
                ("Women holding leadership/supervisory roles", "count"),
                ("Youth enrolled in training activities", "count"),
                ("% youth with knowledge/skills in CSA or NRM", "%"),
                ("Mean number of CSA techniques applied by youth", "count"),
                ("Youth holding leadership/supervisory roles", "count"),
            ]
            df_train = _rows_for(training_metrics)
            st.dataframe(df_train, use_container_width=True)

            # 4) Employment
            st.markdown("#### Employment")
            employment_metrics = [
                ("% of project employees previously unemployed", "%"),
            ]
            df_emp = _rows_for(employment_metrics)
            st.dataframe(df_emp, use_container_width=True)
        with sub3:
            # Biodiversity: remove habitat condition; split into diversity (woody/herbaceous) and vegetation cover by lifeform
            inds = [
                ("Diversity (woody)", "index (0–1)", 0.60, 0.10, "Stratum"),
                ("Diversity (herbaceous)", "index (0–1)", 0.55, 0.10, "Stratum"),
                ("Vegetation cover – forbs", "%", 28, 8, "Stratum"),
                ("Vegetation cover – shrubs", "%", 32, 8, "Stratum"),
                ("Vegetation cover – trees", "%", 40, 10, "Stratum"),
            ]
            dbio = _make_rows(inds)
            st.dataframe(dbio, use_container_width=True)


def page_vcus(data: Dict[str, object], flt: dict) -> None:
    """
    VCUs summary by methodology and by project area instance (site).
    Values shown in tCO2e; Biomass = AGB + BGB; SOC as is; Total = Biomass + SOC.
    """
    st.title("VCUs")
    st.caption("Verified Carbon Units (tCO₂e) by methodology and project area instance.")

    baseline_df: pd.DataFrame = data["baseline"]
    base = _apply_common_filters(baseline_df, flt)
    if base.empty:
        _render_empty_message("No baseline data for the selected filters.")
        return

    def build_vcu_table(df: pd.DataFrame) -> pd.DataFrame:
        def sum_metric(method: str, metric: str) -> float:
            subset = df[(df["methodology"] == method) & (df["metric"] == metric)]
            return float(subset["value"].sum())

        agb_metric = "Aboveground biomass (AGB) (tCO2e)"
        bgb_metric = "Belowground biomass (BGB) (tCO2e)"
        soc_metric = "Soil organic carbon (tCO2e)"

        # Enforce ownership: AGB/BGB -> VM0047; SOC -> VM0042
        v7_agb = sum_metric("VM0047", agb_metric)
        v7_bgb = sum_metric("VM0047", bgb_metric)
        v4_soc = sum_metric("VM0042", soc_metric)

        # Split SOC (placeholder 50/50 unless dataset provides explicit split)
        soc_re = v4_soc * 0.5
        soc_ialm = v4_soc - soc_re

        rows = [
            ["Aboveground biomass (AGB)", 0.00, round(v7_agb, 2), round(v7_agb, 2)],
            ["Belowground biomass (BGB)", 0.00, round(v7_bgb, 2), round(v7_bgb, 2)],
            ["Soil organic carbon — Reforestation (tCO₂e)", round(soc_re, 2), 0.00, round(soc_re, 2)],
            ["Soil organic carbon — IALM & Agroforestry (tCO₂e)", round(soc_ialm, 2), 0.00, round(soc_ialm, 2)],
        ]
        total_v4 = v4_soc
        total_v7 = v7_agb + v7_bgb
        rows.append(["Total", round(total_v4, 2), round(total_v7, 2), round(total_v4 + total_v7, 2)])

        return pd.DataFrame(
            rows,
            columns=[
                "Metric",
                "VM0042 (tCO₂e)",
                "VM0047 (tCO₂e)",
                "Combined (tCO₂e)",
            ],
        )

    st.markdown("#### All selected sites")
    # Two-decimal formatting for the summary table
    tbl_all = build_vcu_table(base)
    for c in ["VM0042 (tCO₂e)", "VM0047 (tCO₂e)", "Combined (tCO₂e)"]:
        tbl_all[c] = tbl_all[c].map(lambda x: f"{float(x):,.2f}")
    st.table(tbl_all)

    # (moved chart to end of section after per-site tables)

    # Per project area instance (site) tabs
    sites = list(dict.fromkeys(base["site"].dropna().tolist()))  # preserve order
    if not sites:
        return
    tabs = st.tabs(sites)
    for t, site in zip(tabs, sites):
        with t:
            df_site = base[base["site"] == site]
            st.markdown(f"##### {site} (tCO₂e)")
            tbl_site = build_vcu_table(df_site)
            for c in ["VM0042 (tCO₂e)", "VM0047 (tCO₂e)", "Combined (tCO₂e)"]:
                tbl_site[c] = tbl_site[c].map(lambda x: f"{float(x):,.2f}")
            st.table(tbl_site)

    # Cross-site comparison chart (enforcing ownership in aggregation)
    agb_rows = (
        base[(base["methodology"] == "VM0047") & (base["metric"] == "Aboveground biomass (AGB) (tCO2e)")]
        .groupby(["site"], as_index=False)["value"]
        .sum()
        .assign(pool="AGB")
    )
    bgb_rows = (
        base[(base["methodology"] == "VM0047") & (base["metric"] == "Belowground biomass (BGB) (tCO2e)")]
        .groupby(["site"], as_index=False)["value"]
        .sum()
        .assign(pool="BGB")
    )
    soc_rows = (
        base[(base["methodology"] == "VM0042") & (base["metric"] == "Soil organic carbon (tCO2e)")]
        .groupby(["site"], as_index=False)["value"]
        .sum()
        .assign(pool="SOC")
    )
    by_site = pd.concat([agb_rows, bgb_rows, soc_rows], ignore_index=True)
    by_site["value"] = by_site["value"].round(2)
    st.markdown("#### VCUs by site and carbon pool (combined across VM0042 & VM0047)")
    fig_comp = px.bar(
        by_site,
        x="site",
        y="value",
        color="pool",
        barmode="group",
        labels={"value": "tCO₂e"},
        text_auto=".2f",
    )
    fig_comp.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Carbon pool")
    st.plotly_chart(fig_comp, use_container_width=True)


def page_deductions(data: Dict[str, object], flt: dict) -> None:
    """
    Deductions summary by methodology (VM0042, VM0047) and combined.
    Rows: Leakage, Performance Benchmark (0–1), GHG emissions (biomass burning, nitrogen inputs, fossil fuels),
    NPR % (dummy 19), Uncertainty % (dummy 10), Total Deductions (tCO2e) – sums tCO2e rows only.
    """
    st.title("Deductions")
    st.caption("Leakage and emissions (tCO₂e), performance benchmark (0–1), NPR %, and Uncertainty %.")

    baseline_df: pd.DataFrame = data["baseline"]
    base = _apply_common_filters(baseline_df, flt)
    if base.empty:
        _render_empty_message("No baseline data for the selected filters.")
        return

    def sum_metric(df: pd.DataFrame, method: str, metric: str) -> float:
        subset = df[(df["methodology"] == method) & (df["metric"] == metric)]
        return float(subset["value"].sum())

    # Metric names
    leak_metric = LEAKAGE_METRIC
    foss_metric = "GHG Emissions — Fossil fuels (tCO2e)"
    burn_metric = "GHG Emissions — Biomass Burning (tCO2e)"
    nit_metric = "GHG Emissions — Nitrogen inputs to Soils (tCO2e)"
    agb_metric = "Aboveground biomass (AGB) (tCO2e)"

    def build_deductions_table(df: pd.DataFrame) -> pd.DataFrame:
        v4_leak = sum_metric(df, "VM0042", leak_metric)
        v7_leak = sum_metric(df, "VM0047", leak_metric)

        v4_foss = sum_metric(df, "VM0042", foss_metric)
        v7_foss = sum_metric(df, "VM0047", foss_metric)
        v4_burn = sum_metric(df, "VM0042", burn_metric)
        v7_burn = sum_metric(df, "VM0047", burn_metric)
        v4_nit = sum_metric(df, "VM0042", nit_metric)
        v7_nit = sum_metric(df, "VM0047", nit_metric)

        # Performance benchmark fraction (VM0047 only, computed on AGB rows)
        v7_pb = float(
            df[(df["methodology"] == "VM0047") & (df["metric"] == agb_metric)]["benchmark_fraction"].dropna().mean()
        )
        v4_pb = np.nan
        comb_pb = v7_pb

        npr = 19.0
        uncert = 10.0

        rows = [
            ["Leakage (tCO₂e)", round(v4_leak, 2), round(v7_leak, 2), round(v4_leak + v7_leak, 2)],
            ["Performance Benchmark (0–1)", v4_pb, round(v7_pb, 2), round(comb_pb, 2)],
            ["GHG Emissions — Biomass Burning (tCO₂e)", round(v4_burn, 2), round(v7_burn, 2), round(v4_burn + v7_burn, 2)],
            ["GHG Emissions — Nitrogen Inputs to Soil (tCO₂e)", round(v4_nit, 2), round(v7_nit, 2), round(v4_nit + v7_nit, 2)],
            ["GHG Emissions — Fossil Fuels (tCO₂e)", round(v4_foss, 2), round(v7_foss, 2), round(v4_foss + v7_foss, 2)],
            ["NPR %", npr, npr, npr],
            ["Uncertainty %", uncert, uncert, uncert],
        ]
        # Total deductions = tCO2e items only (exclude PB, NPR, Uncertainty)
        t_v4 = v4_leak + v4_burn + v4_nit + v4_foss
        t_v7 = v7_leak + v7_burn + v7_nit + v7_foss
        rows.append(["Total Deductions (tCO₂e)", round(t_v4, 2), round(t_v7, 2), round(t_v4 + t_v7, 2)])

        return pd.DataFrame(
            rows,
            columns=["Metric", "VM0042", "VM0047", "Combined"],
        )

    st.markdown("#### All selected sites")
    st.table(build_deductions_table(base))
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
    # Exclude CCB from overview line chart visuals
    mon_no_ccb = mon[mon["methodology"] != "CCB"].copy()
    if mon_no_ccb.empty:
        _render_empty_message("No monitoring data for the selected filters.")
    else:
        trend = mon_no_ccb.groupby(["date", "methodology"], as_index=False)[["tco2e_this_period", "uncertainty_percent"]].mean()
        # Lines only (bands removed per feedback)
        import plotly.graph_objects as go
        fig = go.Figure()
        for meth, d in trend.groupby("methodology"):
            d = d.sort_values("date")
            y = d["tco2e_this_period"].values
            x = d["date"].values
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=meth))
            # Baseline reference (net baseline: AGB+BGB+SOC - emissions - leakage)
            base_m = base[(base["methodology"] == meth)]
            pos = base_m[base_m["metric"].isin(["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"])]["value"].sum()
            neg = base_m[base_m["metric"].isin(EMISSIONS_METRICS + [LEAKAGE_METRIC])]["value"].sum()
            net_baseline = pos - neg
            if len(x):
                # Create a slightly varying baseline over time (non-flat) to reflect realistic fluctuations
                if len(x) >= 2:
                    t = np.linspace(0, 2 * np.pi, len(x))
                    variation = 0.03 * np.sin(t)  # ±3% gentle oscillation
                    y_base = net_baseline * (1 + variation)
                else:
                    y_base = [net_baseline]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_base,
                        mode="lines",
                        name=f"{meth} baseline",
                        line=dict(dash="dash"),
                    )
                )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="tCO₂e", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Contributions and highlights")
    pie_col1, pie_col2, pie_col3, pie_col4 = st.columns([1, 1, 1, 2])
    with pie_col1:
        by_meth = mon_no_ccb.groupby("methodology", as_index=False)["tco2e_this_period"].sum()
        st.plotly_chart(px.pie(by_meth, names="methodology", values="tco2e_this_period", hole=0.5, title="% by methodology"), use_container_width=True)
    with pie_col2:
        pools = base[base["metric"].isin(["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"])].groupby("metric", as_index=False)["value"].sum()
        if pools.empty:
            _render_empty_message("No pool data for current selection.")
        else:
            st.plotly_chart(px.pie(pools, names="metric", values="value", hole=0.5, title="% by carbon pool (AGB, BGB, SOC)"), use_container_width=True)
    with pie_col3:
        if mon_no_ccb.empty:
            _render_empty_message("No stratum data for current selection.")
        else:
            by_stratum = mon_no_ccb.groupby("stratum", as_index=False)["tco2e_this_period"].sum()
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
        # SOC belongs under VM0042. Split SOC row into Reforestation SOC and IALM & Agroforestry SOC.
        vm42_soc_total = float(vm42_base[vm42_base["metric"] == "Soil organic carbon (tCO2e)"]["value"].sum())
        # If you have explicit split fields, replace the 50/50 placeholders below with the correct logic.
        soc_re = round(vm42_soc_total * 0.5, 2)
        soc_ialm_af = round(vm42_soc_total - soc_re, 2)
        vm42_total = soc_re + soc_ialm_af
        df_vm42 = pd.DataFrame({
            "Metric": [
                "Soil Organic Carbon — Reforestation (tCO₂e)",
                "Soil Organic Carbon — IALM & Agroforestry (tCO₂e)",
                "Total",
            ],
            "Value (tCO₂e)": [soc_re, soc_ialm_af, vm42_total],
        })
        df_vm42["Value (tCO₂e)"] = df_vm42["Value (tCO₂e)"].map(lambda x: f"{float(x):,.2f}")
        st.table(df_vm42)
    with t2:
        st.markdown("#### Verified Carbon Units (VM0047)")
        vm47_base = base[base["methodology"] == "VM0047"]
        # AGB and BGB belong under VM0047. SOC excluded here.
        vm47_agb = float(vm47_base[vm47_base["metric"] == "Aboveground biomass (AGB) (tCO2e)"]["value"].sum())
        vm47_bgb = float(vm47_base[vm47_base["metric"] == "Belowground biomass (BGB) (tCO2e)"]["value"].sum())
        vm47_total = vm47_agb + vm47_bgb
        df_vm47 = pd.DataFrame({
            "Metric": ["Aboveground biomass (AGB)", "Belowground biomass (BGB)", "Total"],
            "Value (tCO₂e)": [vm47_agb, vm47_bgb, vm47_total],
        })
        df_vm47["Value (tCO₂e)"] = df_vm47["Value (tCO₂e)"].map(lambda x: f"{float(x):,.2f}")
        st.table(df_vm47)

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

    # Tabs: carbon (VM0042 + VM0047 together) and dedicated CCB
    tab_carbon, tab_ccb = st.tabs(["Carbon (VM0042 & VM0047)", "CCB"])

    with tab_carbon:
        mon_carbon = mon[mon["methodology"].isin(["VM0042", "VM0047"])].copy()
        if mon_carbon.empty:
            _render_empty_message("No monitoring data for VM0042/VM0047 under current filters.")
        else:
            # Consistent, high-contrast colors for methods
            method_color_map = {
                "VM0042": "#2563EB",  # blue-600
                "VM0047": "#10B981",  # emerald-500
            }
            left, right = st.columns(2)
            with left:
                st.markdown("#### Sequestration per period")
                fig = px.bar(
                    mon_carbon,
                    x="period",
                    y="tco2e_this_period",
                    color="methodology",
                    color_discrete_map=method_color_map,
                    barmode="group",
                    hover_data={"run_id": True, "date": True, "tco2e_this_period": ":,.0f"},
                    labels={"tco2e_this_period": "tCO₂e"},
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.markdown("#### Cumulative removals")
                cum = mon_carbon.groupby(["period", "methodology"], as_index=False)["cumulative_tco2e"].max()
                fig2 = px.line(
                    cum,
                    x="period",
                    y="cumulative_tco2e",
                    color="methodology",
                    color_discrete_map=method_color_map,
                    markers=True,
                )
                fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Uncertainty and sample counts")
            cols = st.columns(2)
            with cols[0]:
                fig3 = px.box(
                    mon_carbon,
                    x="period",
                    y="uncertainty_percent",
                    color="methodology",
                    color_discrete_map=method_color_map,
                    points="all",
                )
                fig3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig3, use_container_width=True)
            with cols[1]:
                # Uncertainty by carbon pool (synthetic allocation from overall monitoring uncertainty)
                pools = ["Aboveground biomass (AGB) (tCO2e)", "Belowground biomass (BGB) (tCO2e)", "Soil organic carbon (tCO2e)"]
                up = []
                for _, row in mon_carbon.iterrows():
                    for p in pools:
                        up.append({"period": row["period"], "pool": p, "uncertainty_percent": row["uncertainty_percent"]})
                up_df = pd.DataFrame(up)
                fig4 = px.box(
                    up_df,
                    x="period",
                    y="uncertainty_percent",
                    color="pool",
                    points=False,
                )
                # No title to avoid clipping; compact margins
                fig4.update_layout(height=360, margin=dict(l=10, r=20, t=10, b=10))
                fig4.update_yaxes(automargin=True)
                fig4.update_xaxes(automargin=True)
                st.plotly_chart(fig4, use_container_width=True)

    with tab_ccb:
        mon_ccb = mon[mon["methodology"] == "CCB"].copy()
        if mon_ccb.empty:
            _render_empty_message("No CCB monitoring data for selected filters.")
            return
        periods_sorted = sorted(mon_ccb["period"].dropna().unique().tolist())
        rng = np.random.default_rng(42)

        sub1, sub2, sub3 = st.tabs(["Climate", "Community", "Biodiversity"])

        # Climate: nutrient status over time (Low/Medium/High)
        with sub1:
            st.markdown("#### Climate – Nutrient status per monitoring period")
            nutrient_levels = ["Low", "Medium", "High"]
            level_to_score = {"Low": 1, "Medium": 2, "High": 3}
            rows = []
            for i, p in enumerate(periods_sorted):
                # Deterministic cycling of levels per period for demo
                n_level = nutrient_levels[i % 3]
                p_level = nutrient_levels[(i + 1) % 3]
                k_level = nutrient_levels[(i + 2) % 3]
                rows += [
                    {"period": p, "nutrient": "Nitrogen", "level": n_level, "level_score": level_to_score[n_level]},
                    {"period": p, "nutrient": "Phosphorus", "level": p_level, "level_score": level_to_score[p_level]},
                    {"period": p, "nutrient": "Potassium", "level": k_level, "level_score": level_to_score[k_level]},
                ]
            df_climate = pd.DataFrame(rows)
            fig_c = px.bar(df_climate, x="period", y="level_score", color="nutrient", barmode="group")
            fig_c.update_yaxes(tickmode="array", tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"], title="Level")
            # Extra top margin to prevent any header clipping
            fig_c.update_layout(height=380, margin=dict(l=16, r=16, t=40, b=16), legend_title_text="Nutrient")
            st.plotly_chart(fig_c, use_container_width=True)
            st.dataframe(df_climate.drop(columns=["level_score"]).pivot(index="nutrient", columns="period", values="level").reset_index(), use_container_width=True)

        # Community: grouped sections with bars per metric across periods
        with sub2:
            st.markdown("#### Community – Indicators per monitoring period")

            def build_section(section_metrics: List[Tuple[str, str]]) -> pd.DataFrame:
                rows = []
                for p in periods_sorted:
                    for name, unit in section_metrics:
                        if unit == "%":
                            val = float(np.clip(rng.normal(65, 10), 0, 100))
                        elif unit == "kg per capita":
                            val = float(np.clip(rng.normal(220, 60), 10, None))
                        elif unit == "count":
                            val = float(np.clip(rng.normal(40, 15), 0, None))
                        elif unit == "score":
                            val = float(np.clip(rng.normal(4.5, 1.1), 0, 12))
                        elif unit == "ha":
                            val = float(np.clip(rng.normal(120, 40), 0, None))
                        else:
                            val = float(np.clip(rng.normal(3.0, 0.8), 0, None))
                        rows.append({"period": p, "metric": name, "value": round(val, 2), "unit": unit})
                return pd.DataFrame(rows)

            # Sustainable agroforestry and climate-resilient land use
            landuse_metrics = [
                ("Mean number of crop types per household", "count"),
                ("Crop specific yield (kg per capita)", "kg per capita"),
                ("% households reporting low crop productivity due to reduced soil health", "%"),
                ("% households using any existing land management practice", "%"),
                ("Mapped area under existing land-use practices", "ha"),
            ]
            df_landuse = build_section(landuse_metrics)
            st.markdown("##### Sustainable agroforestry and climate-resilient land use")
            st.plotly_chart(px.bar(df_landuse, x="period", y="value", color="metric", barmode="group"), use_container_width=True)
            st.dataframe(df_landuse, use_container_width=True)

            # Food security
            food_metrics = [
                ("Average household dietary diversity score per village", "score"),
                ("% households reporting year-round food sufficiency", "%"),
                ("% households where ≥25% of food comes from own production", "%"),
            ]
            df_food = build_section(food_metrics)
            st.markdown("##### Food security")
            st.plotly_chart(px.bar(df_food, x="period", y="value", color="metric", barmode="group"), use_container_width=True)
            st.dataframe(df_food, use_container_width=True)

            # Training
            training_metrics = [
                ("Participants with prior training in agriculture or NRM", "count"),
                ("Women enrolled in training activities", "count"),
                ("% women enrolled in training activities", "%"),
                ("Women with knowledge/skills in agriculture or NRM", "count"),
                ("% women with knowledge/skills in agriculture or NRM", "%"),
                ("Mean number of ag/NRM techniques applied by women", "count"),
                ("Women holding leadership/supervisory roles", "count"),
                ("Youth enrolled in training activities", "count"),
                ("% youth with knowledge/skills in CSA or NRM", "%"),
                ("Mean number of CSA techniques applied by youth", "count"),
                ("Youth holding leadership/supervisory roles", "count"),
            ]
            df_train = build_section(training_metrics)
            st.markdown("##### Training")
            st.plotly_chart(px.bar(df_train, x="period", y="value", color="metric", barmode="group"), use_container_width=True)
            st.dataframe(df_train, use_container_width=True)

            # Employment
            employment_metrics = [("% of project employees previously unemployed", "%")]
            df_emp = build_section(employment_metrics)
            st.markdown("##### Employment")
            st.plotly_chart(px.bar(df_emp, x="period", y="value", color="metric", barmode="group"), use_container_width=True)
            st.dataframe(df_emp, use_container_width=True)

        # Biodiversity: diversity and cover over time
        with sub3:
            st.markdown("#### Biodiversity – Indicators per monitoring period")
            bio_metrics = [
                ("Diversity (woody)", "index (0–1)"),
                ("Diversity (herbaceous)", "index (0–1)"),
                ("Vegetation cover – forbs", "%"),
                ("Vegetation cover – shrubs", "%"),
                ("Vegetation cover – trees", "%"),
            ]
            rows = []
            for p in periods_sorted:
                rows.append({"period": p, "metric": "Diversity (woody)", "value": round(float(np.clip(rng.normal(0.60, 0.06), 0, 1)), 2), "unit": "index (0–1)"})
                rows.append({"period": p, "metric": "Diversity (herbaceous)", "value": round(float(np.clip(rng.normal(0.55, 0.06), 0, 1)), 2), "unit": "index (0–1)"})
                rows.append({"period": p, "metric": "Vegetation cover – forbs", "value": round(float(np.clip(rng.normal(28, 8), 0, 100)), 2), "unit": "%"})
                rows.append({"period": p, "metric": "Vegetation cover – shrubs", "value": round(float(np.clip(rng.normal(32, 8), 0, 100)), 2), "unit": "%"})
                rows.append({"period": p, "metric": "Vegetation cover – trees", "value": round(float(np.clip(rng.normal(40, 10), 0, 100)), 2), "unit": "%"})
            df_bio = pd.DataFrame(rows)
            st.plotly_chart(px.bar(df_bio, x="period", y="value", color="metric", barmode="group"), use_container_width=True)
            st.dataframe(df_bio, use_container_width=True)


def page_map(data: Dict[str, object], flt: dict) -> None:
    st.title("Map")
    st.caption("Verified boundaries, stratum polygons and sampling plots.")

    plots = data["plots_geojson"]
    boundaries = data["boundaries_geojson"]
    strata = data["strata_geojson"]
    custom_polys = list(data.get("custom_polygons", []))  # List[Tuple[name, geojson]]
    # Remove Kamwenyetulo polygons as requested
    custom_polys = [t for t in custom_polys if "kamwenyetulo" not in (t[0] or "").lower()]

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

    # Add 'popup' field to each plot feature for a generic tooltip
    enriched = []
    for f in filtered_features:
        props = f.get("properties", {}).copy()
        popup = "<br>".join(
            [
                f"<b>Site</b>: {props.get('site', '')}",
                f"<b>Stratum</b>: {props.get('stratum', '')}",
                f"<b>Methodology</b>: {props.get('methodology', '')}",
                f"<b>Dataset</b>: {props.get('dataset', '')}",
                f"<b>Version</b>: {props.get('dataset_version', '')}",
                f"<b>Value</b>: {props.get('value', '')}",
                f"<b>Uncertainty</b>: {props.get('uncertainty_percent', '')}%",
            ]
        )
        props["popup"] = popup
        enriched.append({"type": "Feature", "geometry": f["geometry"], "properties": props})
    plots_filtered = {"type": "FeatureCollection", "features": enriched}

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

    # Methodology selector for custom polygons
    def _detect_site_key(props: Dict[str, Any]) -> Optional[str]:
        cand = ["site", "Site", "SITE", "site_name", "SITE_NAME", "name", "Name"]
        for k in cand:
            if k in props:
                return k
        return None

    # Dedupe methodology names to avoid repeated options
    methods_available = sorted(list({name for name, _ in custom_polys})) if custom_polys else []
    selected_method = None
    selected_site = None
    filtered_custom = custom_polys
    if methods_available:
        with st.container():
            mcol1, mcol2 = st.columns([1, 1])
            with mcol1:
                selected_method = st.selectbox("Polygon methodology layer", ["All"] + methods_available, index=0)
            if selected_method and selected_method != "All":
                filtered_custom = [t for t in custom_polys if t[0] == selected_method]
            # Site drop-down (aggregated across chosen layers)
            all_sites = []
            for _, gj in filtered_custom:
                for feat in gj.get("features", []):
                    key = _detect_site_key(feat.get("properties", {})) or ""
                    if key:
                        all_sites.append(str(feat["properties"].get(key)))
            all_sites = sorted(list({s for s in all_sites if s and s.strip()}))
            with mcol2:
                selected_site = st.selectbox("Site (attributes)", ["All"] + all_sites, index=0) if all_sites else None

    # Filter features by selected site for popup
    final_polys: List[Tuple[str, dict]] = []
    for name, gj in filtered_custom:
        if selected_site and selected_site != "All":
            feats = []
            for f in gj.get("features", []):
                key = _detect_site_key(f.get("properties", {}))
                if key and str(f["properties"].get(key)) == selected_site:
                    feats.append(f)
            gj2 = {"type": "FeatureCollection", "features": feats}
            final_polys.append((name, gj2))
        else:
            final_polys.append((name, gj))

    # If we have polygons, recentre the map to them (they may be outside plots/boundaries)
    initial_zoom = 6
    try:
        if final_polys:
            merged = {"type": "FeatureCollection", "features": []}
            for _, gj in final_polys:
                merged["features"].extend(gj.get("features", []))
            cx, cy = _center_of_geojson(merged)
            center_lon, center_lat = cx, cy
            # Estimate a sensible zoom from polygon span
            lons, lats = [], []
            for f in merged.get("features", []):
                geom = f.get("geometry", {})
                if geom.get("type") == "Polygon":
                    coords = geom.get("coordinates", [[[]]])[0]
                    for lon, lat in coords:
                        lons.append(lon); lats.append(lat)
                elif geom.get("type") == "MultiPolygon":
                    for ring in geom.get("coordinates", []):
                        for lon, lat in ring[0]:
                            lons.append(lon); lats.append(lat)
            if lons and lats:
                lon_span = max(lons) - min(lons)
                lat_span = max(lats) - min(lats)
                span = max(lon_span, lat_span)
                # crude mapping span (degrees) → zoom
                if span < 0.15: initial_zoom = 12
                elif span < 0.3: initial_zoom = 11
                elif span < 0.6: initial_zoom = 10
                elif span < 1.2: initial_zoom = 9
                elif span < 2.5: initial_zoom = 8
                elif span < 5.0: initial_zoom = 7
                else: initial_zoom = 6
    except Exception:
        pass

    # Build custom polygon layers if available
    custom_layers = []
    # Distinct, high-contrast colors
    custom_colors = [
        [59, 130, 246, 140],   # blue-500
        [16, 185, 129, 140],   # emerald-500
        [245, 158, 11, 140],   # amber-500
        [219, 39, 119, 140],   # pink-600
    ]
    def _infer_method_from(layer_name: str, props: Dict[str, Any]) -> Optional[str]:
        # Try various common keys
        for key in ["methodology", "Methodology", "method", "Method", "METHOD"]:
            if key in props and str(props[key]).strip():
                return str(props[key]).strip()
        nm = (layer_name or "").lower()
        if "vm0042" in nm or "vm42" in nm or "0042" in nm:
            return "VM0042"
        if "vm0047" in nm or "vm47" in nm or "0047" in nm:
            return "VM0047"
        return None
    for idx, (name, gj) in enumerate(final_polys):
        if not gj:
            continue
        gj = _json_safe(gj)
        # Build rich popup from all attributes; compute consistent methodology color
        gj_features = []
        color_map = {"VM0042": [59, 130, 246, 140], "VM0047": [16, 185, 129, 140]}
        for f in gj.get("features", []):
            props = dict(f.get("properties", {}))
            method_value = _infer_method_from(name, props)
            if method_value and "methodology" not in props:
                props["methodology"] = method_value
            # Assign consistent fill color by methodology; fallback to pink if unknown
            col = color_map.get(method_value or str(props.get("methodology", "")).strip() or "", [219, 39, 119, 140])
            props["fill_r"], props["fill_g"], props["fill_b"], props["fill_a"] = col
            if "popup" not in props:
                props["popup"] = "<br>".join([f"<b>{k}</b>: {v}" for k, v in props.items()])
            gj_features.append({"type": "Feature", "geometry": f.get("geometry"), "properties": props})
        gj = {"type": "FeatureCollection", "features": gj_features}
        custom_layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=gj,
                stroked=True,
                filled=True,
                opacity=0.45,
                get_fill_color="[properties.fill_r, properties.fill_g, properties.fill_b, properties.fill_a]",
                get_line_color=[17, 24, 39],
                line_width_min_pixels=3.5,
                pickable=True,
                auto_highlight=True,
            )
        )

    # Ensure all GeoJSONs are JSON-serializable for pydeck
    boundaries_safe = _json_safe(boundaries)
    strata_safe = _json_safe(strata)
    plots_safe = _json_safe(plots_filtered)

    # Guard against invalid GeoJSON objects (deck.gl requires a 'type' field)
    def _ensure_geojson(gj: Any) -> dict:
        if not isinstance(gj, dict) or "type" not in gj:
            return {"type": "FeatureCollection", "features": []}
        if gj.get("type") == "FeatureCollection" and "features" not in gj:
            return {"type": "FeatureCollection", "features": []}
        return gj

    boundaries_safe = _ensure_geojson(boundaries_safe)
    strata_safe = _ensure_geojson(strata_safe)
    plots_safe = _ensure_geojson(plots_safe)

    # Basemap selector: Satellite (Esri) or Satellite (Mapbox) when token available
    basemap_options = ["Satellite (Esri)"] + (["Satellite (Mapbox)"] if MAPBOX_ENABLED else [])
    basemap_choice = st.selectbox("Basemap", basemap_options, index=0)

    # Build layers list and map style according to selection
    layers_list = []
    deck_map_style = None
    if basemap_choice == "Satellite (Mapbox)" and MAPBOX_ENABLED:
        # Use Mapbox Satellite style (requires MAPBOX_TOKEN in secrets/env)
        deck_map_style = "mapbox://styles/mapbox/satellite-v9"
    else:
        # Tokenless satellite fallback (Esri World Imagery)
        layers_list.append(
            pdk.Layer(
                "TileLayer",
                data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            )
        )
    # Only show points and any custom polygons (hide default boundaries/strata overlays)
    layers_list.extend([
        pdk.Layer(
            "GeoJsonLayer",
            data=plots_safe,
            point_type="circle",
            get_fill_color="[properties.completion_status == 'Complete' ? 16 : (properties.completion_status == 'Partial' ? 245 : 239), properties.completion_status == 'Complete' ? 185 : (properties.completion_status == 'Partial' ? 158 : 68), properties.completion_status == 'Complete' ? 129 : (properties.completion_status == 'Partial' ? 11 : 68), 180]",
            get_radius=60,
            pickable=True,
        ),
    ])
    layers_list.extend(custom_layers)

    deck = pdk.Deck(
        # Basemap controlled above; disable built-in provider styles unless Mapbox selected
        map_style=deck_map_style,
        map_provider=None if deck_map_style is None else "mapbox",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=initial_zoom, bearing=0, pitch=0),
        layers=layers_list,
        tooltip={"html": "{popup}", "style": {"backgroundColor": "white", "color": "#111", "fontSize": "12px"}},
    )

    st.pydeck_chart(deck, use_container_width=True)

    # Methodology legend (consistent colors; styled card similar to example)
    # inst_count = sum(len(gj.get("features", [])) for _, gj in final_polys) if final_polys else 0
    inst_count = 1
    legend_title = selected_site if selected_site and selected_site != "All" else "Luxiha"
    legend_html = f"""
    <div style="position:fixed; top:90px; right:24px; z-index:1000; pointer-events:none;">
      <div style="background:#ffffffdd;border:1px solid #ddd;border-radius:6px;padding:12px 14px;box-shadow:0 1px 2px rgba(0,0,0,0.05); pointer-events:auto;">
        <div style="font-weight:700;font-size:16px;margin-bottom:6px;">{legend_title}</div>
        <div style="font-size:12px;color:#555;margin-bottom:8px;">Instance: {inst_count}</div>
        <div style="display:flex;align-items:center;gap:12px;">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="display:inline-block;width:14px;height:14px;background:#3B82F6;border:1px solid #93C5FD;"></span>
            <span style="font-size:13px;color:#111;">VM0042</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="display:inline-block;width:14px;height:14px;background:#10B981;border:1px solid #6EE7B7;"></span>
            <span style="font-size:13px;color:#111;">VM0047</span>
          </div>
        </div>
      </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    # Quick debug note on custom polygons loaded
    if final_polys:
        st.caption(f"Loaded custom polygon layers: {', '.join([n for n,_ in final_polys])}")
        # Attributes preview for currently selected site (if any)
        if selected_site and selected_site != "All":
            rows = []
            for _, gj in final_polys:
                for f in gj.get("features", []):
                    rows.append(f.get("properties", {}))
            if rows:
                st.markdown("##### Selected site attributes")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

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

    # Removed map extent summary section per request


# Removed unused pages: Data Explorer, QC Status, Version History/Audit, Verification Dashboard


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
    "VCUs": page_vcus,
    "Deductions": page_deductions,
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


