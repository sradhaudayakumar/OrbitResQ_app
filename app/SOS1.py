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
import matplotlib.pyplot as plt
from skyfield.api import Loader, EarthSatellite, Topos, load, utc  # Added utc import
from datetime import datetime
import random

# === CONFIG ===
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Paths for Triage Dashboard
CSV_PATH = r"C:\Users\udaya\OneDrive\Desktop\AIresque\Hackathon Clinical data - Sheet1.csv"
ORS_API_KEY = st.secrets["ORS_API_KEY"]  # Safe for Streamlit Cloud
SHAKEMAP_PATH = r"C:\Users\udaya\OneDrive\Desktop\AIresque\intensity_contours_turkey.json"
MAG_CSV_PATH = r"C:\Users\udaya\OneDrive\Desktop\AIresque\magnitude.csv"
ELDERLY_CSV_PATH = r"C:\Users\udaya\OneDrive\Desktop\AIresque\elderly_population_estimate.csv"
SENTINEL_IMAGES = {
    r"C:\Users\udaya\OneDrive\Desktop\AIresque\S3A_SL_2_LST____2023_S3A_ThermalPreview.png": {
        "label": "Sentinel-3 Thermal Anomaly Image",
        "description": "Thermal infrared image showing heat anomalies across Hatay, Turkey after the February 2023 earthquake. Blue indicates cooler zones, while white/bright shows potential heat concentration.",
        "coordinates": "Approx. Lat: 36.2°N, Lon: 36.1°E"
    },
    r"C:\Users\udaya\OneDrive\Desktop\AIresque\Sentinel-1 SAR VV Image S1A_IW_GRDH_1SDV_202_S1A_SAR_VV.png": {
        "label": "Sentinel-1 SAR VV Image",
        "description": "Synthetic Aperture Radar (VV polarization) to detect surface displacements or structural changes. Useful in identifying collapsed zones with possible trapped victims.",
        "coordinates": "Approx. Lat: 36.2°N, Lon: 36.1°E"
    },
    r"C:\Users\udaya\OneDrive\Desktop\AIresque\Sentinel-1 SAR VH Image S1A_IW_GRDH_1SDV_202_S1A_SAR_VH.png": {
        "label": "Sentinel-1 SAR VH Image",
        "description": "SAR cross-polarized (VH) data enhances feature detection like disturbed debris zones, aiding post-earthquake search/rescue missions.",
        "coordinates": "Approx. Lat: 36.2°N, Lon: 36.1°E"
    }
}

# Paths for Hospital Finder
GEOJSON_PATH = r"C:/Users/udaya/OneDrive/Desktop/AIresque/merged_damaged_roads.geojson"
MAX_DISTANCE_KM = 300

# === UTILITY FUNCTIONS ===
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

@st.cache_resource(show_spinner=False)
def get_ors_client():
    return openrouteservice.Client(key=ORS_API_KEY)

@st.cache_data(show_spinner=False)
def get_matrix(_client, locations):
    return _client.distance_matrix(
        locations=locations,
        profile="driving-car",
        metrics=["distance", "duration"],
        sources=[0],
        destinations=list(range(1, len(locations)))
    )
        
@st.cache_data(show_spinner=False)
def get_route(_client, coords, avoid_geojson=None):
    params = {
        "coordinates": coords,
        "profile": "driving-car",
        "format": "geojson"
    }
    if avoid_geojson:
        params["options"] = {"avoid_polygons": avoid_geojson["features"][0]["geometry"]}
    return _client.directions(**params)

# === GALILEO SAR FUNCTIONS ===
def fetch_galileo_tles():
    """Fetch current Galileo satellite TLEs from Celestrak"""
    try:
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle"
        response = requests.get(url)
        response.raise_for_status()
        
        tle_data = response.text.strip().split('\n')
        satellites = []
        
        for i in range(0, len(tle_data), 3):
            if i+2 >= len(tle_data):
                break
            name = tle_data[i].strip()
            line1 = tle_data[i+1].strip()
            line2 = tle_data[i+2].strip()
            satellites.append((name, line1, line2))
        
        return satellites
    except Exception as e:
        st.warning(f"Could not fetch Galileo TLEs from Celestrak: {e}")
        return []

def load_galileo_satellites():
    """Load Galileo satellites from Celestrak into Skyfield objects"""
    tles = fetch_galileo_tles()
    satellites = []
    for name, line1, line2 in tles:
        satellites.append(EarthSatellite(line1, line2, name))
    return satellites

def get_visible_satellites(lat, lon, time=None):
    from skyfield.api import wgs84

    load = Loader('~/.skyfield-data')
    ts = load.timescale()
    t = ts.from_datetime(time.replace(tzinfo=utc) if time else datetime.utcnow().replace(tzinfo=utc))

    observer = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
    satellites = load_galileo_satellites()

    visible = []

    for sat in satellites:
        sat_at = sat.at(t)
        difference = sat_at - observer.at(t)
        alt, az, distance = difference.altaz()

        if alt.degrees > 10:
            visible.append({
                'name': sat.name,
                'altitude': alt.degrees,
                'azimuth': az.degrees,
                'distance': distance.km
            })

    return visible





def display_galileo_sar_simulation(lat, lon):
    """Display Galileo SAR response with satellite data"""
    st.header("🛰️ Galileo MEOSAR Distress Signal ")
    st.markdown("""
    This Galileo Search and Rescue service (MEOSAR) receives a distress signal 
    and relays confirmation using the unique Return Link Service (RLS).
    """)
    
    if st.button("🆘 SOS Distress Signal", 
                help="Press to send a distress signal to Galileo satellites",
                type="primary"):
        
        with st.spinner("Acquiring current Galileo satellite positions..."):
            # Get current time with UTC timezone
            from datetime import timezone
            current_time = datetime.now(timezone.utc)

            try:
                visible_sats = get_visible_satellites(lat, lon, current_time)
                if not visible_sats:
                    st.warning("No Galileo satellites visible at this location/time")
                    return
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(1, 6):
                    progress_bar.progress(i * 20)
                    status_text.text(f"Searching for Galileo satellites... [{i}/5]")
                    time.sleep(0.5)
                
                visible_sats.sort(key=lambda x: x['altitude'], reverse=True)
                receiving_sat = visible_sats[0]
                
                base_time = max(3, min(15, int(20 - (receiving_sat['altitude'] / 5))))
                confirmation_time = random.randint(base_time - 2, base_time + 2)
                
                st.success(f"""
                **✅ Distress signal received by {receiving_sat['name'].split(' (')[0]}**
                
                - **Confirmation received in**: {confirmation_time} seconds (via RLS)
                - **Satellite elevation**: {receiving_sat['altitude']:.1f}°
                - **Azimuth**: {receiving_sat['azimuth']:.1f}°
                - **Distance**: {receiving_sat['distance']:.1f} km
                - **Local Time**: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}
                """)
                
                st.markdown(f"""
                **🔔 Galileo Return Link Service (RLS) Activated**  
                The victim's beacon has received confirmation that their distress signal was received by:
                
                - **MEOLUT**: {random.choice(['Larnaca, Cyprus', 'Maspalomas, Spain', 'Toulouse, France'])}
                - **Rescue Coordination Center**: {random.choice(['Ankara MRCC', 'Rome RCC', 'Piraeus JRCC'])}
                - **Confirmation ID**: RL-{random.randint(100000, 999999)}
                """)
                
                st.subheader("Visible Galileo Satellites at Time of Distress")
                sat_df = pd.DataFrame(visible_sats)
                sat_df['Type'] = 'Galileo'
                st.dataframe(
                    sat_df[['name', 'altitude', 'azimuth', 'distance']]
                    .rename(columns={
                        'name': 'Satellite',
                        'altitude': 'Elevation (°)',
                        'azimuth': 'Azimuth (°)',
                        'distance': 'Distance (km)'
                    })
                    .round(1)
                    .style.set_properties(**{'text-align': 'left'})
                )
                
                st.subheader("Satellite Visibility Skyplot")
                try:
                    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                    for sat in visible_sats:
                        ax.plot(np.radians(sat['azimuth']), 90 - sat['altitude'], 'o', 
                                label=sat['name'].split(' (')[0])
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    ax.set_ylim(0, 90)
                    ax.set_yticks(range(0, 90, 15))
                    ax.set_yticklabels([f'{x}°' for x in range(90, 0, -15)])
                    ax.set_title("Satellite Positions (N=0°, E=90°)")
                    ax.legend(bbox_to_anchor=(1.1, 1.05))
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"Could not generate skyplot: {e}")
                
            except Exception as e:
                st.error(f"Error in SAR : {str(e)}")

# [Rest of your existing functions remain unchanged...]
# [Triage Dashboard functions, Hospital Finder functions, etc.]
# === TRIAGE DASHBOARD FUNCTIONS ===
def load_elderly_data(path):
    try:
        df = pd.read_csv(path)
        df["Elderly_Estimated"] = (df["Elderly_Percent"] / 100 * df["Total_Population"]).astype(int)
        df["Risk_Level"] = df["Elderly_Percent"].apply(lambda x: "High" if x > 15 else ("Medium" if x >= 10 else "Low"))
        return df
    except Exception as e:
        st.error(f"Error loading elderly data: {e}")
        return pd.DataFrame()

def display_phase_3_victim_detection():
    st.header("🔦 Phase 3: Trapped Victim Detection via Sentinel Imagery")
    st.caption("Thermal and SAR image data from Sentinel satellites identifying possible heat or motion anomalies.")
    for path, meta in SENTINEL_IMAGES.items():
        st.subheader(meta["label"])
        if os.path.exists(path):
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                st.image(img, caption=f"{meta['label']}\n{os.path.basename(path)}", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render image due to size or format: {e}")
            st.markdown(f"Description: {meta['description']}")
            st.markdown(f"Coordinates: {meta['coordinates']}")
        else:
            st.warning(f"Missing image: {os.path.basename(path)}")

def classify_earthquake_zones(shake_json_path):
    try:
        with open(shake_json_path, "r") as f:
            data = json.load(f)

        geo = Nominatim(user_agent="OrbitResQ-zone-locator")
        zone_data = []

        for feature in data["features"]:
            mmi = feature["properties"].get("value", 0)
            geometry = shape(feature["geometry"])
            if not isinstance(geometry, Polygon):
                continue
            centroid = geometry.centroid
            if mmi >= 7:
                zone = "Red"
                priority = "High"
            elif mmi >= 5:
                zone = "Orange"
                priority = "Medium"
            else:
                zone = "Green"
                priority = "Low"

            lat, lon = round(centroid.y, 4), round(centroid.x, 4)
            try:
                location = geo.reverse((lat, lon), timeout=10)
                region = location.address if location else "Unknown"
            except:
                region = "Unknown"
            time.sleep(1)

            zone_data.append({
                "Zone": zone,
                "MMI": mmi,
                "Priority": priority,
                "Latitude": lat,
                "Longitude": lon,
                "Region": region
            })

        return pd.DataFrame(zone_data)
    except Exception as e:
        st.error(f"Failed to classify earthquake zones: {e}")
        return pd.DataFrame()

def classify_magnitude_events(csv_path):
    try:
        df = pd.read_csv(csv_path)

        def classify(row):
            if row["mag"] >= 7.0:
                return "Red", "Major earthquake (MMI ≥ 7)", "#FF0000", 1
            elif row["mag"] >= 5.0:
                return "Orange", "Moderate earthquake (MMI 5–6.9)", "#FFA500", 2
            else:
                return "Green", "Minor earthquake (MMI < 5)", "#00CC66", 3

        df[["Zone", "Description", "Color", "Priority"]] = df.apply(lambda row: pd.Series(classify(row)), axis=1)
        df = df.sort_values("Priority")
        return df
    except Exception as e:
        st.warning(f"Error loading magnitude data: {e}")
        return pd.DataFrame()

def run_triage_dashboard():
    st.title("🚑 OrbitResQ Triage Dashboard")

    if "triage_stage" not in st.session_state:
        st.session_state.triage_stage = "start"

    if st.session_state.triage_stage == "start":
        if st.button("👩‍🚒 First Responder"):
            st.session_state.triage_stage = "ask_count"

    elif st.session_state.triage_stage == "ask_count":
        count = st.number_input("How many patients are you reporting?", min_value=1, step=1)
        if st.button("Next"):
            st.session_state.triage_count = count
            st.session_state.triage_stage = "assign_status"

    elif st.session_state.triage_stage == "assign_status":
        st.subheader("🩺 Assign Severity to Each Patient")
        st.session_state.statuses = []
        for i in range(int(st.session_state.triage_count)):
            status = st.selectbox(
                f"Patient {i+1} Status:",
                ["Red - Critical", "Yellow - Mild", "Green - Stable"],
                key=f"status_{i}"
            )
            st.session_state.statuses.append(status)
        if st.button("Next: Enter Location"):
            st.session_state.triage_stage = "location"

    elif st.session_state.triage_stage == "location":
        st.subheader("📍 Enter Patient Location")
        location_mode = st.radio("Choose location method", ["Use My Current Location", "Type Address"])
        if location_mode == "Use My Current Location":
            lat, lon, country = get_ip_location()
            if lat and lon:
                st.success(f"Detected location: {country} ({lat:.4f}, {lon:.4f})")
                st.session_state.user_location = (lat, lon)
                if st.button("Show Nearest Hospitals"):
                    st.session_state.triage_stage = "show_hospitals"
            else:
                st.error("Could not detect location from IP")
        else:
            address = st.text_input("Enter address")
            if address:
                lat, lon, city, country = get_coordinates_from_address(address)
                if lat and lon:
                    st.success(f"Address located: {city}, {country} ({lat:.4f}, {lon:.4f})")
                    st.session_state.user_location = (lat, lon)
                    if st.button("Show Nearest Hospitals"):
                        st.session_state.triage_stage = "show_hospitals"
                else:
                    st.error("Could not find coordinates for the given address")

    elif st.session_state.triage_stage == "show_hospitals":
        try:
            df = pd.read_csv(CSV_PATH)
            df = df.rename(columns={
                "Latitude": "lat",
                "Longitude": "lon",
                "Hospital": "name",
                "Beds": "beds",
                "Province": "province",
                "Clinical capacity": "capacity"
            })
            df["category"] = df["capacity"].apply(lambda x: "Red" if x >= 3 else ("Yellow" if x == 2 else "Green"))
        except Exception as e:
            st.error(f"Error loading hospital data: {e}")
            return

        status_mapping = {
            "Red - Critical": "Red",
            "Yellow - Mild": "Yellow",
            "Green - Stable": "Green"
        }

        selected_categories = list({status_mapping[s] for s in st.session_state.statuses})
        lat, lon = st.session_state.user_location
        client = openrouteservice.Client(key=ORS_API_KEY)

        for category in selected_categories:
            subset = df[df["category"] == category].copy()
            if subset.empty:
                st.warning(f"No hospitals available for category: {category}")
                continue
            subset["distance_km"] = subset.apply(lambda row: geodesic((lat, lon), (row["lat"], row["lon"])).km, axis=1)
            subset = subset.sort_values("distance_km").head(3)
            durations = []
            for _, row in subset.iterrows():
                try:
                    route = client.directions(coordinates=[(lon, lat), (row["lon"], row["lat"])] ,profile='driving-car')
                    duration = route['routes'][0]['summary']['duration'] / 60
                except:
                    duration = "Unavailable"
                durations.append(duration)
            subset["duration_min"] = durations

            st.subheader(f"🏥 Top 3 Nearest {category} Hospitals")
            st.dataframe(subset[["name", "province", "beds", "distance_km", "duration_min"]].round(2))

    st.header("🌍 Phase 1: Earthquake Zone Classification")
    zone_df = classify_earthquake_zones(SHAKEMAP_PATH)
    if not zone_df.empty:
        st.dataframe(zone_df)

    st.header("👵 Phase 2: Vulnerability Classification by Elderly Population")
    elderly_df = load_elderly_data(ELDERLY_CSV_PATH)
    if not elderly_df.empty:
        st.dataframe(elderly_df)

    st.header("🧯 Earthquake Zone + Event Classification Table")
    mag_df = classify_magnitude_events(MAG_CSV_PATH)
    if not mag_df.empty:
        display_df = mag_df[["time", "place", "latitude", "longitude", "mag", "depth", "Zone", "Description", "Color"]]
        styled = display_df.style.apply(
            lambda row: [f"background-color: {row['Color']}; color: white" if col == 'Description' else "" for col in display_df.columns],
            axis=1
        )
        st.dataframe(styled)
    else:
        st.info("No earthquake magnitude data available.")

    display_phase_3_victim_detection()

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


    try:
        matrix = get_matrix(client, locations)
        df["distance_km"] = [d / 1000 for d in matrix["distances"][0]]
        df["duration_min"] = [t / 60 for t in matrix["durations"][0]]
    except Exception as e:
        st.error(f"Routing matrix failed: {e}")
        return

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
        try:
            route = get_route(client, coords, avoid_geojson=avoid_geojson)
            accessible_hospitals.append((row, route))
        except:
            skipped_hospitals.append(row["name"])
        if len(accessible_hospitals) >= 5:
            break

    if not accessible_hospitals:
        st.warning("No accessible hospitals found due to damaged roads.")
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

    # Moved inaccessible hospitals notice BEFORE directions
    if skipped_hospitals:
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

        folium.GeoJson(selected_route, name=f"Route to {selected_row['name']}",
                       style_function=lambda x: {'color': 'green'}).add_to(m)
        folium.Marker((selected_row["lat"], selected_row["lon"]), popup=selected_row["name"],
                      icon=folium.Icon(color="green", icon="plus")).add_to(m)

        try:
            client = get_ors_client()
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
            st.warning(f"Could not fetch directions: {e}")

    st.markdown(f"""
    🏥 Hospital: {selected_row.get('name', 'Unknown')}  
    🚗 Distance: {selected_row['distance_km']:.2f} km  
    ⏱ Time: {selected_row['duration_min']:.1f} min
    """)

    st_folium(m, width=700, height=500)
# === MAIN APP ===
def main():
    st.set_page_config(page_title="OrbitResQ Dashboard", layout="wide")
    
    tab1, tab2 = st.tabs(["🚑 Triage Dashboard", "🏥 Hospital Finder"])
    
    with tab1:
        run_triage_dashboard()
    
    with tab2:
        st.title("🟥 OrbitResQ – Find Hospitals Nearby")
        st.markdown("### 📍 Choose Your Location")
        location_mode = st.radio("Select method", ["Use My Current Location", "Enter Address"], key="location_mode")
        lat, lon = None, None

        if location_mode == "Use My Current Location":
            lat, lon, _ = get_ip_location()
            if lat and lon:
                st.success(f"Detected Location: {lat:.4f}, {lon:.4f}")
                display_galileo_sar_simulation(lat, lon)
            else:
                st.warning("Could not detect location from IP")
        else:
            address = st.text_input("Enter address", key="address_input")
            if address:
                lat, lon, city, country = get_coordinates_from_address(address)
                if lat and lon:
                    st.success(f"Located: {city}, {country} ({lat:.4f}, {lon:.4f})")
                    display_galileo_sar_simulation(lat, lon)
                else:
                    st.warning("Address not found.")

        if lat and lon:
            st.subheader("🗺 Closest Hospitals (Avoiding Damaged Roads)")
            run_hospital_finder((lat, lon))

if __name__ == "__main__":
    main()
