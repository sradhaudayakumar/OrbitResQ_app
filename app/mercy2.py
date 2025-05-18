import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
import requests
from geopy.distance import geodesic
import openrouteservice
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import json
from shapely.geometry import shape, Polygon
import time
import os
from PIL import Image, ImageFile
import numpy as np
from shapely.geometry import mapping

# === CONFIG ===
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Paths relative to the project root
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "hospitals.csv")
SHAKEMAP_PATH = os.path.join(DATA_DIR, "intensity_contours.json")
MAG_CSV_PATH = os.path.join(DATA_DIR, "magnitude.csv")
ELDERLY_CSV_PATH = os.path.join(DATA_DIR, "elderly_population.csv")
GEOJSON_PATH = os.path.join(DATA_DIR, "damaged_roads.geojson")

# API Key - Should be set as environment variable in Streamlit Cloud
ORS_API_KEY = st.secrets.get("ORS_API_KEY", "your_default_key_here")

# Sentinel images configuration
SENTINEL_IMAGES = {
    "thermal.png": {
        "label": "Sentinel-3 Thermal Anomaly Image",
        "description": "Thermal infrared image showing heat anomalies across Hatay, Turkey after the February 2023 earthquake. Blue indicates cooler zones, while white/bright shows potential heat concentration.",
        "coordinates": "Approx. Lat: 36.2°N, Lon: 36.1°E"
    },
    "sar_vv.png": {
        "label": "Sentinel-1 SAR VV Image",
        "description": "Synthetic Aperture Radar (VV polarization) to detect surface displacements or structural changes. Useful in identifying collapsed zones with possible trapped victims.",
        "coordinates": "Approx. Lat: 36.2°N, Lon: 36.1°E"
    },
    "sar_vh.png": {
        "label": "Sentinel-1 SAR VH Image",
        "description": "SAR cross-polarized (VH) data enhances feature detection like disturbed debris zones, aiding post-earthquake search/rescue missions.",
        "coordinates": "Approx. Lat: 36.2°N, Lon: 36.1°E"
    }
}

MAX_DISTANCE_KM = 300

# === UTILITY FUNCTIONS ===
@st.cache_data
def get_ip_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5)
        data = res.json()
        loc = data.get("loc", None)
        country = data.get("country", "Unknown")
        if loc:
            lat, lon = map(float, loc.split(","))
            return lat, lon, country
    except:
        pass
    return None, None, None

@st.cache_data
def get_coordinates_from_address(address):
    try:
        geo = Nominatim(user_agent="OrbitResQ-triage")
        location = geo.geocode(address, timeout=10)
        if location:
            lat, lon = location.latitude, location.longitude
            city = location.raw.get("display_name", "Unknown").split(",")[0].strip()
            country = location.raw.get("display_name", "Unknown").split(",")[-1].strip()
            return lat, lon, city, country
    except:
        pass
    return None, None, None, None

@st.cache_resource
def get_ors_client():
    try:
        if not ORS_API_KEY or ORS_API_KEY == "your_default_key_here":
            st.warning("No valid ORS API key configured - routing features will be limited")
            return None
        return openrouteservice.Client(key=ORS_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize ORS client: {e}")
        return None

@st.cache_data
def get_matrix(_client, locations):
    if _client is None:
        return None
    try:
        return _client.distance_matrix(
            locations=locations,
            profile="driving-car",
            metrics=["distance", "duration"],
            sources=[0],
            destinations=list(range(1, len(locations))))
    except Exception as e:
        st.warning(f"Matrix request failed: {e}")
        return None

@st.cache_data
def get_route(_client, coords, avoid_geojson=None):
    if _client is None:
        return None
    params = {
        "coordinates": coords,
        "profile": "driving-car",
        "format": "geojson"
    }
    if avoid_geojson:
        params["options"] = {"avoid_polygons": avoid_geojson["features"][0]["geometry"]}
    try:
        return _client.directions(**params)
    except Exception as e:
        st.warning(f"Route request failed: {e}")
        return None

# === TRIAGE DASHBOARD FUNCTIONS ===
# ... [keep all existing triage dashboard functions unchanged] ...

# === HOSPITAL FINDER FUNCTIONS ===
def run_hospital_finder(user_location):
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.rename(columns={
            "Latitude": "lat",
            "Longitude": "lon",
            "Hospital": "name",
            "Beds": "beds",
            "Province": "province"
        })
        df = df[df[['lat', 'lon']].notnull().all(axis=1)]
    except Exception as e:
        st.error(f"Error loading hospital data: {e}")
        return

    df["geo_distance"] = df.apply(lambda row: geodesic(user_location, (row["lat"], row["lon"])).km, axis=1)
    df = df[df["geo_distance"] <= MAX_DISTANCE_KM].reset_index(drop=True)
    if df.empty:
        st.warning("No hospitals within 300km.")
        return

    client = get_ors_client()
    locations = [[user_location[1], user_location[0]]] + df[['lon', 'lat']].values.tolist()

    # Updated matrix calculation with fallback
    matrix = get_matrix(client, locations)
    if matrix and 'distances' in matrix and 'durations' in matrix:
        df["distance_km"] = [d / 1000 for d in matrix["distances"][0]]
        df["duration_min"] = [t / 60 for t in matrix["durations"][0]]
        routing_available = True
    else:
        st.warning("Routing service unavailable - using straight-line distances")
        df["distance_km"] = df["geo_distance"]
        df["duration_min"] = df["distance_km"] * 2  # Approx 30 km/h speed
        routing_available = False

    try:
        gdf = gpd.read_file(GEOJSON_PATH).to_crs(epsg=4326)
        gdf['geometry'] = gdf['geometry'].buffer(0.0005)
        avoid_geometry = gdf.unary_union
        avoid_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": mapping(avoid_geometry),
                "properties": {}
            }]
        }
    except Exception as e:
        st.warning(f"Failed to load or buffer avoid_polygons: {e}")
        avoid_geojson = None

    accessible_hospitals = []
    skipped_hospitals = []

    for _, row in df.sort_values(by="duration_min").iterrows():
        coords = [(user_location[1], user_location[0]), (row["lon"], row["lat"])]
        if routing_available:
            route = get_route(client, coords, avoid_geojson=avoid_geojson)
            if route:
                accessible_hospitals.append((row, route))
            else:
                skipped_hospitals.append(row["name"])
        else:
            accessible_hospitals.append((row, None))
        
        if len(accessible_hospitals) >= 5:
            break

    if not accessible_hospitals:
        st.warning("No accessible hospitals found.")
        return

    st.subheader("Top Accessible Hospitals")
    rows = [r[0] for r in accessible_hospitals]
    df_display = pd.DataFrame(rows)[["name", "distance_km", "duration_min"]].round(2)
    if "beds" in df.columns:
        df_display["Beds Available"] = [r[0]["beds"] for r in accessible_hospitals]
    st.table(df_display)

    m = folium.Map(location=user_location, zoom_start=11)
    folium.Marker(user_location, popup="📍 You", icon=folium.Icon(color="blue")).add_to(m)

    for (row, route) in accessible_hospitals:
        if route:
            folium.GeoJson(route, name=row["name"]).add_to(m)
        folium.Marker((row["lat"], row["lon"]), popup=row["name"], icon=folium.Icon(color="red")).add_to(m)

    selected_name = st.selectbox(
        "🧭 Choose hospital to highlight route & get directions",
        df_display["name"].tolist(),
        key=f"hospital_selector_{hash(str(user_location))}"
    )

    selected_hospital_data = next((r for r in accessible_hospitals if r[0]["name"] == selected_name), None)

    try:
        folium.GeoJson(gdf.to_json(), name="Damaged Roads", style_function=lambda x: {"color": "black", "weight": 2}).add_to(m)
    except Exception as e:
        st.warning(f"Failed to overlay damaged roads: {e}")

    if skipped_hospitals and routing_available:
        with st.sidebar:
            st.markdown("""
            <div style="
                background-color:#ffe6e6;
                border-left: 6px solid #cc0000;
                padding: 1em;
                margin-top: 1em;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            ">
                <h4 style="color:#cc0000;">❌ Notice Board: Inaccessible Hospitals</h4>
                <p style="color:#cc0000;">The following hospital(s) could not be reached due to blocked or damaged roads:</p>
                <ul style="color:#333;">
                    %s
                </ul>
            </div>
            """ % "".join(f"<li>{name}</li>" for name in skipped_hospitals), unsafe_allow_html=True)

    if selected_hospital_data:
        selected_row, selected_route = selected_hospital_data
        coords = [(user_location[1], user_location[0]), (selected_row["lon"], selected_row["lat"])]

        if selected_route:
            folium.GeoJson(selected_route, name=f"Route to {selected_row['name']}",
                           style_function=lambda x: {'color': 'green'}).add_to(m)
            folium.Marker((selected_row["lat"], selected_row["lon"]), popup=selected_row["name"],
                          icon=folium.Icon(color="green", icon="plus")).add_to(m)

            if routing_available and client:
                try:
                    route_detail = client.directions(coords, profile='driving-car', format='json')
                    steps = route_detail['routes'][0]['segments'][0]['steps']

                    directions_html = "".join(
                        f"<li><strong>{i+1}.</strong> {step['instruction']} <span style='color:#555;'>({step['distance']:.0f} m, {step['duration']:.0f} sec)</span></li>"
                        for i, step in enumerate(steps)
                    )

                    with st.sidebar:
                        st.markdown(f"""
                        <div style="
                            background-color:#e6ffe6;
                            border-left: 6px solid #33cc33;
                            padding: 1em;
                            margin-top: 1em;
                            border-radius: 8px;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        ">
                            <h4 style="color:#2e8b57;">🧭 Directions Board</h4>
                            <p style="color:#2e8b57;">Step-by-step directions to <strong>{selected_row['name']}</strong>:</p>
                            <ul style="color:#333; padding-left: 1em;">
                                {directions_html}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not fetch detailed directions: {e}")

    st.markdown(f"""
    🏥 Hospital: {selected_row.get('name', 'Unknown')}  
    🚗 Distance: {selected_row['distance_km']:.2f} km  
    ⏱ Time: {selected_row['duration_min']:.1f} min
    """)

    st_folium(m, width=700, height=500)

# === MAIN APP === 
# ... [keep the main() function unchanged] ...

if __name__ == "__main__":
    main()
