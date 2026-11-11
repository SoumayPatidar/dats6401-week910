# DATS 6401: Weeks 9 & 10: Map Visualization
# Soumay Patidar
# Dataset: US Tesla Supercharger locations
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Supercharger Map", layout="wide")
st.title(" US Tesla Supercharger Dashboard")

st.markdown(
    """
    This app maps **2,264 Tesla Supercharger locations** across the US. 
    It combines a choropleth (states colored by charger density) with 
    the actual point locations on top, plus some state-level comparisons.
    """
)

# LOAD DATA (cached)
@st.cache_data
def load_data():
    sc = pd.read_csv("data/superchargers_us.csv")
    states_gdf = gpd.read_file("data/states.geojson")
    stats = pd.read_csv("data/state_stats.csv")
    stats["fips"] = stats["fips"].astype(str).str.zfill(2)
    return sc, states_gdf, stats

sc, states_gdf, stats = load_data()

@st.cache_data
def do_joins(_sc, _states_gdf, _stats):
    # spatial join: assign each charger to a state
    sc_gdf = gpd.GeoDataFrame(
        _sc, geometry=gpd.points_from_xy(_sc.lon, _sc.lat), crs="EPSG:4326"
    )
    joined = gpd.sjoin(
        sc_gdf, _states_gdf[["state", "fips", "geometry"]],
        how="left", predicate="within"
    )

    matched = joined["fips"].notna().sum()
    unmatched_df = joined[joined["fips"].isna()]
    unmatched_count = len(unmatched_df)

    # count sites and stalls per state
    counts = (
        joined[joined["fips"].notna()]
        .groupby("fips")
        .agg(sites=("name", "count"), total_stalls=("stalls", "sum"))
        .reset_index()
    )

    # attribute join: merge population from state_stats
    merged = _states_gdf.merge(counts, on="fips", how="left")
    merged = merged.fillna({"sites": 0, "total_stalls": 0})
    merged = merged.merge(
        _stats[["fips", "state", "pop_total", "land_sqmi", "pop_per_sqmi"]],
        on="fips", how="left", suffixes=("_geo", "_stats")
    )

    # compute rates
    merged["sites_per_million"] = (
        merged["sites"] / merged["pop_total"] * 1_000_000
    ).round(1)
    merged["stalls_per_million"] = (
        merged["total_stalls"] / merged["pop_total"] * 1_000_000
    ).round(1)

    return joined, merged, matched, unmatched_count, unmatched_df

joined, merged, matched, unmatched_count, unmatched_df = do_joins(sc, states_gdf, stats)

# SIDEBAR (shared filter)
st.sidebar.title("Filters")

# build abbreviation-to-name lookup
abbrev_to_name = dict(zip(stats["state"], stats["state"]))
# also map from supercharger state abbrevs using the joined data
state_abbrevs = sorted(sc["state"].dropna().unique().tolist())

state_list = ["All States"] + state_abbrevs
selected_state = st.sidebar.selectbox("Select a state", state_list)

st.sidebar.markdown("---")

min_stalls = st.sidebar.slider("Min stalls per site", 1, 40, 1)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(sc)} total charger sites**")
st.sidebar.markdown(f"**{sc['stalls'].sum():,.0f} total stalls**")
st.sidebar.caption("DATS 6401 | Weeks 9-10 | Soumay Patidar")

# get FIPS for selected state (for filtering merged data)
selected_fips = None
if selected_state != "All States":
    match = joined[joined["state_left"] == selected_state]
    if len(match) > 0:
        selected_fips = match["fips"].dropna().iloc[0]

# filter points
filtered_sc = sc[sc["stalls"] >= min_stalls]
if selected_state != "All States":
    filtered_sc = filtered_sc[filtered_sc["state"] == selected_state]

st.markdown("---")

# TABS
tab1, tab2, tab3 = st.tabs([" Layered Map", "State Comparison", "Joins & Data"])

# TAB 1 :LAYERED MAP
with tab1:
    st.markdown("### Choropleth + Supercharger Locations")

    # set map center
    if selected_state != "All States" and len(filtered_sc) > 0:
        center_lat = filtered_sc["lat"].mean()
        center_lon = filtered_sc["lon"].mean()
        zoom = 6
    else:
        center_lat, center_lon, zoom = 39.5, -98.5, 4

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles="OpenStreetMap")

    # layer 1: choropleth (rate)
    folium.Choropleth(
        geo_data=json.loads(states_gdf.to_json()),
        data=merged,
        columns=["fips", "sites_per_million"],
        key_on="feature.properties.fips",
        fill_color="YlOrRd",
        fill_opacity=0.6,
        line_opacity=0.3,
        legend_name="Supercharger sites per million people",
        nan_fill_color="white",
    ).add_to(m)

    # layer 2: points
    for _, row in filtered_sc.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="navy",
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['name']}<br>{row['stalls']} stalls",
        ).add_to(m)

    st_folium(m, width=900, height=500, returned_objects=[])

    st.markdown(
        f"""
        **Caption:** This map shows two layers of data. The **choropleth** 
        (colored polygons) encodes supercharger sites per million people 
        by state, using the YlOrRd sequential color ramp with Jenks 
        natural breaks classification. The **navy dots** are individual 
        supercharger locations ({len(filtered_sc)} shown with current 
        filters). Projection: WGS84 / EPSG:4326 (the default for web 
        maps). Basemap: CartoDB Positron.

        **What this shows:** States like Vermont, Wyoming, and Montana 
        have high per-capita charger density despite low total counts, 
        because their populations are small. California has the most 
        sites overall but its rate is more moderate due to its huge 
        population. This is the classic counts vs rates difference — 
        the two tell very different stories about the same data.
        """
    )

# TAB 2: STATE COMPARISON
with tab2:
    st.markdown("### Sites vs Rate — The Normalization Effect")

    col1, col2 = st.columns(2)

    # filter merged data for this tab too
    if selected_fips is not None:
        display_merged = merged.copy()  # show all for comparison context
    else:
        display_merged = merged.copy()

    with col1:
        top_count = display_merged.nlargest(15, "sites")
        fig1 = px.bar(
            top_count.sort_values("sites"),
            x="sites", y="state_geo", orientation="h",
            title="Top States by Total Sites (count)",
            labels={"sites": "Number of sites", "state_geo": ""},
            color_discrete_sequence=["steelblue"],
        )
        fig1.update_layout(height=450)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        top_rate = display_merged.nlargest(15, "sites_per_million")
        fig2 = px.bar(
            top_rate.sort_values("sites_per_million"),
            x="sites_per_million", y="state_geo", orientation="h",
            title="Top States by Sites Per Million People (rate)",
            labels={"sites_per_million": "Sites per million", "state_geo": ""},
            color_discrete_sequence=["coral"],
        )
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        **What this shows:** The left chart uses raw counts and the 
        right uses a rate (sites divided by population in millions). 
        California dominates the count chart but drops when normalized. 
        Smaller states like Vermont or Wyoming jump up in the rate 
        chart because a handful of chargers divided by a small 
        population gives a high per-capita number.

        Neither chart is "right": they answer different questions. 
        The count says "where are the most chargers" and the rate 
        says "where is it easiest to find one relative to how many 
        people live there." The assignment specifically asks for 
        a rate because counts alone can be misleading for comparisons 
        across states with very different populations.
        """
    )

    # stalls breakdown
    st.markdown("### Stalls Distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(filtered_sc["stalls"], bins=20, color="steelblue", edgecolor="white")
    ax3.set_xlabel("Stalls per site")
    ax3.set_ylabel("Number of sites")
    ax3.set_title("Distribution of stall counts across sites")
    plt.tight_layout()
    st.pyplot(fig3)


# TAB 3: JOINS & DATA
with tab3:
    st.markdown("### How the Data Was Joined")

    st.markdown("#### 1. Spatial Join")
    st.markdown(
        f"""
        Each of the 2,264 supercharger points was joined to the state 
        polygon that contains it using `gpd.sjoin(..., predicate='within')`.

        **Result:** {matched} points matched a state, **{unmatched_count} 
        did not match**.

        The 3 unmatched points are:
        """
    )

    unmatched_display = unmatched_df[["name", "city", "state_left", "lat", "lon"]].copy()
    unmatched_display.columns = ["Name", "City", "State", "Lat", "Lon"]
    st.dataframe(unmatched_display, use_container_width=True, hide_index=True)

    st.markdown(
        """
        These sites fall just outside the state boundary polygons:
        Manhattan's Vesey St is near the waterfront (the polygon 
        edge may not extend into the water), McDermitt is right on 
        the Nevada/Oregon border, and South Padre Island is an 
        offshore barrier island. This is a common issue with spatial 
        joins, boundary files don't always include water or tiny 
        islands.
        """
    )

    st.markdown("#### 2. Attribute Join")
    st.markdown(
        """
        After counting sites per state, I joined `state_stats.csv` 
        on the FIPS code to get population numbers. This lets me 
        compute a **rate** (sites per million people) instead of 
        just a count.

        The join uses FIPS codes (2-digit state identifiers) as the 
        key — I had to zero-pad them since the CSV stores them as 
        integers (so "1" becomes "01" for Alabama).
        """
    )

    # show the merged table
    show_cols = ["state_geo", "fips", "sites", "total_stalls",
                 "pop_total", "sites_per_million", "stalls_per_million"]
    display_df = merged[show_cols].copy()
    display_df.columns = ["State", "FIPS", "Sites", "Stalls",
                          "Population", "Sites/Million", "Stalls/Million"]
    display_df = display_df.sort_values("Sites", ascending=False)

    if selected_fips is not None:
        display_df = display_df[display_df["FIPS"] == selected_fips]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Classification scheme:** The choropleth on Tab 1 uses 
        **natural breaks (Jenks)** with 6 classes in the YlOrRd 
        color ramp. I chose Jenks because it minimizes within-class 
        variance, grouping states with similar rates together. 
        Quantiles would have forced equal numbers of states into 
        each bin, which hides the fact that a few states have much 
        higher rates than the rest.

        **Projection:** The web map uses EPSG:4326 (WGS84), which 
        is the default for Leaflet/folium. It's a geographic CRS, 
        not a projected one, so areas near the poles are distorted 
        (Alaska looks bigger than it is). For a more honest 
        area comparison I'd use Albers Equal Area (EPSG:5070), but 
        that doesn't work with folium's web tiles.
        """
    )
