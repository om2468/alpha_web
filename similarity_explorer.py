"""
AlphaEarth Embedding Similarity Explorer
Interactive Streamlit app for exploring cosine similarity in embedding space.

Click on the image to select a reference point!

Run with: streamlit run similarity_explorer.py
"""

import streamlit as st
import numpy as np
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from streamlit_image_coordinates import streamlit_image_coordinates
import io

st.set_page_config(
    page_title="AlphaEarth Similarity Explorer",
    page_icon="🌍",
    layout="wide"
)

# --- Data Loading Functions ---

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

@st.cache_data
def load_embeddings_with_meta(filepath, downsample=4):
    """Load GeoTIFF with downsampling for memory efficiency. Returns embeddings and geo metadata."""
    with rasterio.open(filepath) as src:
        original_transform = src.transform
        original_crs = src.crs
        if downsample > 1:
            data = src.read(
                out_shape=(src.count, src.height // downsample, src.width // downsample),
                resampling=rasterio.enums.Resampling.nearest
            )
            # Adjust transform for downsampling
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

def load_embeddings(filepath, downsample=4):
    """Load GeoTIFF with downsampling (legacy wrapper)."""
    emb, _, _ = load_embeddings_with_meta(filepath, downsample)
    return emb

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
    
    # Flatten embeddings for batch computation
    flat_emb = embeddings.reshape(-1, 64)
    
    # Compute cosine similarity
    ref_norm = np.linalg.norm(ref_embedding)
    flat_norms = np.linalg.norm(flat_emb, axis=1)
    
    # Avoid division by zero
    valid_norms = flat_norms > 0
    similarity = np.zeros(h * w)
    similarity[valid_norms] = np.dot(flat_emb[valid_norms], ref_embedding) / (flat_norms[valid_norms] * ref_norm)
    
    return similarity.reshape(h, w)

# --- Main App ---

st.title("🌍 AlphaEarth Embedding Similarity Explorer")
st.markdown("**👆 Click on the image below to select a reference point and find similar areas!**")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# File paths
TIFF_2018 = "input_data/xbckr5wa0l2omaauu-0000008192-0000000000.tiff"
TIFF_2024 = "input_data/xks7764uu0jo1h8jh-0000008192-0000000000.tiff"

# Year selection
year = st.sidebar.radio("Select Year", ["2024", "2018"])
tiff_path = TIFF_2024 if year == "2024" else TIFF_2018

# Downsample factor
downsample = st.sidebar.slider("Downsample Factor", min_value=2, max_value=8, value=4, 
                                help="Higher = faster but lower resolution")

# Load data
with st.spinner(f"Loading {year} embeddings..."):
    try:
        embeddings, geo_transform, geo_crs = load_embeddings_with_meta(tiff_path, downsample)
        valid_mask = ~np.isnan(embeddings).any(axis=2)
        rgb_image = create_false_color(embeddings)
        h, w = embeddings.shape[:2]
        st.sidebar.success(f"✅ Loaded: {h} x {w} pixels")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure the TIFF files are in the `input_data/` folder.")
        st.stop()

# Initialize session state for coordinates
if 'ref_y' not in st.session_state:
    st.session_state.ref_y = h // 2
if 'ref_x' not in st.session_state:
    st.session_state.ref_x = w // 2

# Sidebar coordinate display and manual override
st.sidebar.header("📍 Reference Point")
st.sidebar.markdown(f"**Current:** ({st.session_state.ref_y}, {st.session_state.ref_x})")

# Manual coordinate input (optional)
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
    # Still show the clickable image
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
# Draw red cross
rgb_with_marker[y_start:y_end, ref_x, :] = [1, 0, 0]
rgb_with_marker[ref_y, x_start:x_end, :] = [1, 0, 0]

# Convert to uint8 for streamlit_image_coordinates
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

# Left: Similarity map
with col1:
    st.markdown("**Cosine Similarity Map**")
    # Colormap: RdYlBu_r (red = high similarity)
    sim_normalized = np.clip(similarity_map, 0, 1)
    sim_colored = cm.RdYlBu_r(sim_normalized)[:, :, :3]
    st.image(sim_colored, caption="Red = High Similarity, Blue = Low", width="stretch")

# Right: Matching pixels only (black and white)
with col2:
    st.markdown(f"**Similar Areas (≥ {threshold:.2f})**")
    mask = (similarity_map >= threshold) & valid_mask
    n_similar = np.sum(mask)
    pct_similar = 100 * n_similar / np.sum(valid_mask)
    
    # Show binary mask - white for matches, black for non-matches
    binary_display = np.zeros((h, w, 3), dtype=np.float32)
    binary_display[mask] = [1, 1, 1]  # White for similar pixels
    
    st.image(binary_display, caption=f"{n_similar:,} pixels ({pct_similar:.1f}%)", width="stretch")

# Export section
st.markdown("---")
st.subheader("💾 Export Results")

export_col1, export_col2 = st.columns(2)

with export_col1:
    # Export as GeoTIFF (binary mask)
    if st.button("📥 Export Mask as GeoTIFF", help="Download binary mask (1=similar, 0=not similar)"):
        # Create binary mask - need to flip back for GeoTIFF format
        export_mask = np.flipud(mask.astype(np.uint8))
        
        # Write to in-memory buffer
        buffer = io.BytesIO()
        with rasterio.open(
            buffer,
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=1,
            dtype=np.uint8,
            crs=geo_crs,
            transform=geo_transform,
            compress='lzw'
        ) as dst:
            dst.write(export_mask, 1)
            dst.update_tags(1, description=f'Similarity mask (threshold={threshold})')
        
        buffer.seek(0)
        st.download_button(
            label="⬇️ Download GeoTIFF",
            data=buffer,
            file_name=f"similarity_mask_{year}_thresh{threshold:.2f}.tif",
            mime="image/tiff"
        )

with export_col2:
    # Export similarity values as GeoTIFF
    if st.button("📥 Export Similarity Map as GeoTIFF", help="Download continuous similarity values"):
        # Flip back for GeoTIFF format
        export_sim = np.flipud(similarity_map.astype(np.float32))
        
        buffer = io.BytesIO()
        with rasterio.open(
            buffer,
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=1,
            dtype=np.float32,
            crs=geo_crs,
            transform=geo_transform,
            compress='lzw'
        ) as dst:
            dst.write(export_sim, 1)
            dst.update_tags(1, description=f'Cosine similarity from point ({ref_y}, {ref_x})')
        
        buffer.seek(0)
        st.download_button(
            label="⬇️ Download GeoTIFF",
            data=buffer,
            file_name=f"similarity_map_{year}_ref{ref_y}_{ref_x}.tif",
            mime="image/tiff"
        )

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

st.markdown("---")
st.markdown("**💡 Tips:**")
st.markdown("""
- **Click** directly on the image to select a reference point
- Adjust the **threshold** slider to change what counts as "similar"
- Try different **years** to compare 2018 vs 2024
- Use the **downsample factor** for faster loading on large images
""")
