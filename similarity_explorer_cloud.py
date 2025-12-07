"""
AlphaEarth Embedding Similarity Explorer (Cloud Version)
Interactive Streamlit app for exploring cosine similarity in embedding space.

This version loads tile index from remote parquet via DuckDB and streams
GeoTIFFs directly from S3 - suitable for Streamlit Cloud deployment.

Run with: streamlit run similarity_explorer_cloud.py
"""

import streamlit as st
import numpy as np
import duckdb
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from streamlit_image_coordinates import streamlit_image_coordinates
import pydeck as pdk
import plotly.graph_objects as go
import io

st.set_page_config(
    page_title="AlphaEarth Similarity Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
PARQUET_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"

# --- Data Loading Functions ---

@st.cache_resource
def get_duckdb_connection():
    """Create a DuckDB connection with httpfs extension for remote parquet."""
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    return conn

@st.cache_data(ttl=3600)
def load_tile_index():
    """Load tile index from remote parquet using DuckDB."""
    conn = get_duckdb_connection()
    df = conn.execute(f"""
        SELECT 
            fid,
            path,
            year,
            utm_zone,
            wgs84_west,
            wgs84_south,
            wgs84_east,
            wgs84_north,
            location
        FROM '{PARQUET_URL}'
        ORDER BY year DESC, utm_zone
    """).fetchdf()
    return df

@st.cache_data(ttl=3600)
def get_available_years():
    """Get list of available years."""
    conn = get_duckdb_connection()
    result = conn.execute(f"""
        SELECT DISTINCT year 
        FROM '{PARQUET_URL}' 
        ORDER BY year DESC
    """).fetchall()
    return [r[0] for r in result]

@st.cache_data(ttl=3600)
def get_utm_zones_for_year(year: int):
    """Get available UTM zones for a specific year."""
    conn = get_duckdb_connection()
    result = conn.execute(f"""
        SELECT DISTINCT utm_zone 
        FROM '{PARQUET_URL}' 
        WHERE year = {year}
        ORDER BY utm_zone
    """).fetchall()
    return [r[0] for r in result]

@st.cache_data(ttl=3600)
def get_tiles_for_selection(year: int, utm_zone: str):
    """Get tiles for a specific year and UTM zone."""
    conn = get_duckdb_connection()
    df = conn.execute(f"""
        SELECT 
            fid,
            path,
            location,
            wgs84_west,
            wgs84_south,
            wgs84_east,
            wgs84_north
        FROM '{PARQUET_URL}'
        WHERE year = {year} AND utm_zone = '{utm_zone}'
        ORDER BY wgs84_south DESC, wgs84_west
    """).fetchdf()
    return df

@st.cache_data(ttl=3600)
def get_all_tiles_for_year(year: int):
    """Get all tiles for a specific year (for map view)."""
    conn = get_duckdb_connection()
    df = conn.execute(f"""
        SELECT 
            fid,
            path,
            utm_zone,
            location,
            wgs84_west,
            wgs84_south,
            wgs84_east,
            wgs84_north
        FROM '{PARQUET_URL}'
        WHERE year = {year}
    """).fetchdf()
    return df

def create_tile_polygons(df):
    """Convert tile bounds to polygon coordinates for pydeck."""
    polygons = []
    for _, row in df.iterrows():
        west, south, east, north = row['wgs84_west'], row['wgs84_south'], row['wgs84_east'], row['wgs84_north']
        polygons.append({
            'polygon': [
                [west, north],
                [east, north],
                [east, south],
                [west, south],
                [west, north]
            ],
            'fid': row['fid'],
            'utm_zone': row['utm_zone'],
            'location': row['location'] if row['location'] else f"Tile {row['fid']}",
            'path': row['path'],
            'center_lat': (north + south) / 2,
            'center_lon': (west + east) / 2
        })
    return polygons

@st.cache_data
def dequantize(values, nodata_value=-128):
    """Convert int8 AlphaEarth embeddings [-127, 127] to float [-1, 1]."""
    nodata_mask = values == nodata_value
    values_float = values.astype(np.float32)
    result = ((values_float / 127.5) ** 2) * np.sign(values_float)
    result[nodata_mask] = np.nan
    return result

@st.cache_data
def normalize_display(arr):
    """Normalize array to [0, 1] for display."""
    min_val, max_val = np.nanmin(arr), np.nanmax(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def s3_path_to_https(s3_path: str) -> str:
    """Convert s3:// path to HTTPS URL for Source Cooperative data."""
    # Source Cooperative data is accessible via HTTPS
    # s3://us-west-2.opendata.source.coop/... -> https://data.source.coop/...
    if s3_path.startswith("s3://us-west-2.opendata.source.coop/"):
        return s3_path.replace("s3://us-west-2.opendata.source.coop/", "https://data.source.coop/")
    # Fallback to vsis3 for other S3 paths
    return s3_path.replace("s3://", "/vsis3/")

@st.cache_data(show_spinner="Loading embeddings from cloud...")
def load_embeddings_from_s3(s3_path: str, downsample: int = 4):
    """
    Load Cloud Optimized GeoTIFF from S3 with efficient streaming.
    
    Uses HTTP range requests to only download the data needed:
    - With downsampling, reads from overviews if available (much faster)
    - Uses GDAL's vsicurl for efficient HTTP access
    - Caches results to avoid re-downloading
    """
    # Convert to HTTPS URL for better compatibility and no AWS credentials needed
    url = s3_path_to_https(s3_path)
    
    # Use /vsicurl/ for HTTPS URLs (more reliable than /vsis3/ for public data)
    if url.startswith("https://"):
        vsi_path = f"/vsicurl/{url}"
    else:
        vsi_path = url  # Already a /vsis3/ path
    
    # Configure GDAL for optimal COG streaming
    env = rasterio.Env(
        AWS_NO_SIGN_REQUEST='YES',
        GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff,.tif',
        GDAL_HTTP_MULTIPLEX='YES',  # Enable HTTP/2 multiplexing
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES',  # Merge range requests
        VSI_CACHE='TRUE',  # Enable VSI caching
        VSI_CACHE_SIZE='5000000',  # 5MB cache
    )
    
    with env:
        with rasterio.open(vsi_path) as src:
            original_transform = src.transform
            original_crs = src.crs
            
            if downsample > 1:
                # Calculate output shape
                out_height = src.height // downsample
                out_width = src.width // downsample
                
                # Read with downsampling - GDAL will use overviews if available
                data = src.read(
                    out_shape=(src.count, out_height, out_width),
                    resampling=rasterio.enums.Resampling.nearest
                )
                new_transform = Affine(
                    original_transform.a * downsample,
                    original_transform.b,
                    original_transform.c,
                    original_transform.d,
                    original_transform.e * downsample,
                    original_transform.f
                )
            else:
                data = src.read()
                new_transform = original_transform
    
    # Transpose to (height, width, 64) and flip vertically
    embeddings = np.flipud(np.transpose(data, (1, 2, 0)))
    return dequantize(embeddings), new_transform, original_crs

@st.cache_data
def create_false_color(embeddings, r_dim=0, g_dim=31, b_dim=63):
    """Create RGB false color composite from 3 embedding dimensions."""
    rgb = np.stack([
        normalize_display(embeddings[:, :, r_dim]),
        normalize_display(embeddings[:, :, g_dim]),
        normalize_display(embeddings[:, :, b_dim])
    ], axis=-1)
    return rgb

def compute_similarity_map(embeddings, ref_y, ref_x):
    """Compute cosine similarity from a reference pixel to all other pixels."""
    h, w = embeddings.shape[:2]
    ref_embedding = embeddings[ref_y, ref_x, :]
    
    if np.isnan(ref_embedding).any():
        return None
    
    flat_emb = embeddings.reshape(-1, 64)
    ref_norm = np.linalg.norm(ref_embedding)
    flat_norms = np.linalg.norm(flat_emb, axis=1)
    
    valid_norms = flat_norms > 0
    similarity = np.zeros(h * w)
    similarity[valid_norms] = np.dot(flat_emb[valid_norms], ref_embedding) / (flat_norms[valid_norms] * ref_norm)
    
    return similarity.reshape(h, w)

# --- Main App ---

st.title("🌍 AlphaEarth Embedding Similarity Explorer")
st.markdown("**Cloud Edition** - Data streamed from [Source Cooperative](https://source.coop/tge-labs/aef)")

# Load available years (needed for both tabs)
with st.spinner("Loading tile index..."):
    try:
        years = get_available_years()
    except Exception as e:
        st.error(f"Error loading tile index: {e}")
        st.stop()

# Create tabs for Map View and Explorer
tab_map, tab_explorer = st.tabs(["🗺️ Map View", "🔍 Similarity Explorer"])

# ============== MAP VIEW TAB ==============
with tab_map:
    st.markdown("### Browse and select a tile to load")
    
    # Filters at top
    col_year, col_zone, col_downsample = st.columns([1, 1, 1])
    
    with col_year:
        map_year = st.selectbox("Year", years, index=0, key="map_year")
    
    with col_zone:
        # Load tiles to get UTM zones
        with st.spinner("Loading tiles..."):
            map_tiles_df = get_all_tiles_for_year(map_year)
        utm_zones = sorted(map_tiles_df['utm_zone'].unique())
        selected_zone = st.selectbox("UTM Zone", ["All Zones"] + utm_zones, key="utm_filter")
    
    with col_downsample:
        map_downsample = st.slider(
            "Resolution",
            min_value=4, max_value=16, value=8,
            help="Downsample factor (higher = faster load)",
            key="map_downsample"
        )
    
    # Apply zone filter
    if selected_zone != "All Zones":
        filtered_df = map_tiles_df[map_tiles_df['utm_zone'] == selected_zone]
    else:
        filtered_df = map_tiles_df
    
    # Create polygons for filtered tiles only (for better performance)
    tile_polygons = create_tile_polygons(filtered_df)
    
    # Layout: Map on left, selection on right
    col_map, col_select = st.columns([2, 1])
    
    with col_map:
        if tile_polygons:
            avg_lat = np.mean([p['center_lat'] for p in tile_polygons])
            avg_lon = np.mean([p['center_lon'] for p in tile_polygons])
            
            polygon_layer = pdk.Layer(
                "PolygonLayer",
                tile_polygons,
                get_polygon="polygon",
                get_fill_color=[100, 180, 200, 100],
                get_line_color=[50, 50, 50, 200],
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True,
                highlight_color=[255, 200, 0, 200],
            )
            
            view_state = pdk.ViewState(
                latitude=avg_lat,
                longitude=avg_lon,
                zoom=4 if selected_zone != "All Zones" else 2,
                pitch=0,
            )
            
            deck = pdk.Deck(
                layers=[polygon_layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>{location}</b><br/>Zone: {utm_zone}<br/>FID: {fid}<br/><i>Select from dropdown →</i>",
                    "style": {"backgroundColor": "#333", "color": "white", "fontSize": "12px"}
                },
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
            )
            
            st.pydeck_chart(deck, key="tile_map", use_container_width=True)
            st.caption(f"Showing **{len(tile_polygons):,}** tiles. Hover for info.")
        else:
            st.warning("No tiles to display.")
    
    with col_select:
        st.markdown("#### Select Tile")
        
        # Show current selection if any
        if 'map_selected_tile' in st.session_state and st.session_state.map_selected_tile:
            tile_info = st.session_state.map_selected_tile
            st.success(f"✅ **{tile_info.get('location', 'Tile')}**")
        
        # Searchable dropdown
        if len(filtered_df) > 0:
            tile_options = filtered_df.apply(
                lambda row: f"{row['location'] if row['location'] else 'Tile ' + str(row['fid'])}",
                axis=1
            ).tolist()
            
            selected_tile_idx = st.selectbox(
                "Choose tile",
                range(len(tile_options)),
                format_func=lambda i: tile_options[i],
                key="tile_dropdown",
                index=None,
                placeholder="Type to search..."
            )
            
            if selected_tile_idx is not None:
                row = filtered_df.iloc[selected_tile_idx]
                
                # Show tile info
                st.markdown(f"""
                **FID:** {row['fid']}  
                **Zone:** {row['utm_zone']}  
                **Bounds:**  
                {row['wgs84_west']:.2f}° to {row['wgs84_east']:.2f}° E  
                {row['wgs84_south']:.2f}° to {row['wgs84_north']:.2f}° N
                """)
                
                if st.button("📥 Load This Tile", type="primary", use_container_width=True):
                    st.session_state.load_tile = True
                    st.session_state.current_path = row['path']
                    st.session_state.current_downsample = map_downsample
                    st.session_state.selected_year = map_year
                    st.session_state.map_selected_tile = {
                        'fid': row['fid'],
                        'path': row['path'],
                        'utm_zone': row['utm_zone'],
                        'location': row['location'] if row['location'] else f"Tile {row['fid']}"
                    }
                    st.rerun()
        else:
            st.warning("No tiles match filters.")

# ============== SIMILARITY EXPLORER TAB ==============
with tab_explorer:
    # Check if we have a tile loaded
    if 'load_tile' not in st.session_state:
        st.info("👈 Go to the **Map View** tab and click on a tile to load it.")
        st.markdown("""
        ### How to use:
        1. Go to the **Map View** tab
        2. Select a **year** from the dropdown
        3. **Click on any tile** on the map to load it
        4. The similarity explorer will appear here automatically
        """)
        st.stop()
    
    # Sidebar shows current tile info
    st.sidebar.header("📍 Current Tile")
    if 'map_selected_tile' in st.session_state and st.session_state.map_selected_tile:
        tile_info = st.session_state.map_selected_tile
        st.sidebar.markdown(f"**Location:** {tile_info.get('location', 'Unknown')}")
        st.sidebar.markdown(f"**UTM Zone:** {tile_info.get('utm_zone', 'N/A')}")
        st.sidebar.markdown(f"**FID:** {tile_info.get('fid', 'N/A')}")
    
    # Downsample factor for reloading
    st.sidebar.header("⚙️ Settings")
    downsample = st.sidebar.slider(
        "Downsample Factor", 
        min_value=2, max_value=16, value=st.session_state.get('current_downsample', 8),
        help="Higher = faster loading but lower resolution",
        key="explorer_downsample"
    )
    
    # Update downsample if changed
    if downsample != st.session_state.get('current_downsample', 8):
        if st.sidebar.button("🔄 Reload with new resolution"):
            st.session_state.current_downsample = downsample
            st.rerun()

    # Load the embeddings
    with st.spinner(f"Streaming tile from S3..."):
        try:
            embeddings, geo_transform, geo_crs = load_embeddings_from_s3(
                st.session_state.current_path, 
                st.session_state.current_downsample
            )
            valid_mask = ~np.isnan(embeddings).any(axis=2)
            rgb_image = create_false_color(embeddings)
            h, w = embeddings.shape[:2]
            st.sidebar.success(f"✅ Loaded: {h} x {w} pixels")
        except Exception as e:
            st.error(f"Error loading tile: {e}")
            st.info("Try increasing the downsample factor or selecting a different tile.")
            st.stop()

    # Initialize session state for coordinates
    if 'ref_y' not in st.session_state or st.session_state.get('last_path') != st.session_state.current_path:
        st.session_state.ref_y = h // 2
        st.session_state.ref_x = w // 2
        st.session_state.last_path = st.session_state.current_path

    # Sidebar coordinate display
    st.sidebar.header("📍 Reference Point")
    st.sidebar.markdown(f"**Current:** ({st.session_state.ref_y}, {st.session_state.ref_x})")

    # Manual coordinate input
    with st.sidebar.expander("Manual Coordinate Entry"):
        col1, col2 = st.columns(2)
        manual_y = col1.number_input("Y", min_value=0, max_value=h-1, value=st.session_state.ref_y, key="manual_y")
        manual_x = col2.number_input("X", min_value=0, max_value=w-1, value=st.session_state.ref_x, key="manual_x")
        if st.button("Apply"):
            st.session_state.ref_y = manual_y
            st.session_state.ref_x = manual_x
            st.rerun()

    # Threshold slider
    threshold = st.sidebar.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

    # Get reference coordinates
    ref_y = st.session_state.ref_y
    ref_x = st.session_state.ref_x

    # Check if reference point is valid
    if not valid_mask[ref_y, ref_x]:
        st.warning(f"⚠️ Point ({ref_y}, {ref_x}) is invalid (NoData). Please click a different location.")
        st.subheader("Click on the image to select a valid point:")
        rgb_uint8 = (rgb_image * 255).astype(np.uint8)
        coords = streamlit_image_coordinates(rgb_uint8, key="selector_invalid")
        if coords is not None:
            new_x = coords["x"]
            new_y = coords["y"]
            if 0 <= new_y < h and 0 <= new_x < w:
                st.session_state.ref_y = new_y
                st.session_state.ref_x = new_x
                st.rerun()
        st.stop()

    # Compute similarity
    with st.spinner("Computing similarity map..."):
        similarity_map = compute_similarity_map(embeddings, ref_y, ref_x)

    if similarity_map is None:
        st.error("Could not compute similarity for this point.")
        st.stop()

    # Create the clickable image with marker
    st.subheader("👆 Click on the image to select a new reference point")

    # Prepare RGB image with marker
    rgb_with_marker = rgb_image.copy()
    marker_size = max(5, h // 100)
    y_start, y_end = max(0, ref_y - marker_size), min(h, ref_y + marker_size)
    x_start, x_end = max(0, ref_x - marker_size), min(w, ref_x + marker_size)
    rgb_with_marker[y_start:y_end, ref_x, :] = [1, 0, 0]
    rgb_with_marker[ref_y, x_start:x_end, :] = [1, 0, 0]

    rgb_uint8 = (rgb_with_marker * 255).astype(np.uint8)

    # Clickable image
    coords = streamlit_image_coordinates(rgb_uint8, key="main_selector")
    if coords is not None:
        new_x = coords["x"]
        new_y = coords["y"]
        if 0 <= new_y < h and 0 <= new_x < w:
            if new_y != st.session_state.ref_y or new_x != st.session_state.ref_x:
                st.session_state.ref_y = new_y
                st.session_state.ref_x = new_x
                st.rerun()

    st.caption(f"Current reference point: **({ref_y}, {ref_x})**")

    # Results section
    st.markdown("---")
    st.subheader("📊 Similarity Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Cosine Similarity Map**")
        sim_normalized = np.clip(similarity_map, 0, 1)
        sim_colored = cm.RdYlBu_r(sim_normalized)[:, :, :3]
        st.image(sim_colored, caption="Red = High Similarity, Blue = Low", use_container_width=True)

    with col2:
        st.markdown(f"**Similar Areas (≥ {threshold:.2f})**")
        mask = (similarity_map >= threshold) & valid_mask
        n_similar = np.sum(mask)
        pct_similar = 100 * n_similar / np.sum(valid_mask)
        
        binary_display = np.zeros((h, w, 3), dtype=np.float32)
        binary_display[mask] = [1, 1, 1]
        
        st.image(binary_display, caption=f"{n_similar:,} pixels ({pct_similar:.1f}%)", use_container_width=True)

    # Statistics
    st.markdown("---")
    st.subheader("📈 Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reference Point", f"({ref_y}, {ref_x})")
    col2.metric("Threshold", f"{threshold:.2f}")
    col3.metric("Similar Pixels", f"{n_similar:,}")
    col4.metric("Coverage", f"{pct_similar:.1f}%")

    # Histogram
    st.subheader("Similarity Distribution")
    valid_similarities = similarity_map[valid_mask]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(valid_similarities, bins=100, color='steelblue', edgecolor='none', alpha=0.7)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Pixel Count')
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # Export section (simplified for cloud - no GeoTIFF export)
    st.markdown("---")
    st.subheader("💾 Export Results")

    # Get tile info for filenames
    export_fid = st.session_state.get('map_selected_tile', {}).get('fid', 'unknown')
    export_year = st.session_state.get('selected_year', 'unknown')

    # Export as PNG
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download Similarity Map (PNG)"):
            buf = io.BytesIO()
            plt.imsave(buf, sim_colored, format='png')
            buf.seek(0)
            st.download_button(
                label="⬇️ Download PNG",
                data=buf,
                file_name=f"similarity_map_{export_year}_{export_fid}.png",
                mime="image/png"
            )

    with col2:
        if st.button("📥 Download Binary Mask (PNG)"):
            buf = io.BytesIO()
            plt.imsave(buf, binary_display, format='png')
            buf.seek(0)
            st.download_button(
                label="⬇️ Download PNG",
                data=buf,
                file_name=f"similarity_mask_{export_year}_{export_fid}_thresh{threshold:.2f}.png",
                mime="image/png"
            )

    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("""
    - **Click** directly on the image to select a reference point
    - Adjust the **threshold** slider to change what counts as "similar"
    - Use higher **downsample factor** for faster loading on slow connections
    - Go back to **Map View** to load a different tile
    """)

# Footer (outside tabs)
st.markdown("---")
st.markdown("*Data source: [TGE Labs AlphaEarth Foundation](https://source.coop/tge-labs/aef) via Source Cooperative*")
