# AlphaEarth Similarity Explorer

A cloud-based web application for exploring [AlphaEarth](https://www.alphaearthdata.com/) satellite imagery embeddings using cosine similarity. Stream massive embedding datasets directly from the cloud without downloading anything.

🌐 **[Try it live on Streamlit Cloud](https://alpha-web.streamlit.app)**

## What is AlphaEarth?

AlphaEarth provides **pixel-level embedding vectors** derived from satellite imagery. Each pixel is represented as a **64-dimensional vector** that encodes semantic information about that location—a "fingerprint" of the land cover.

These embeddings are:
- **Pre-computed** from Sentinel-2 satellite imagery
- **Stored as Cloud Optimized GeoTIFFs (COGs)** with 64 bands
- **Quantized to int8** for efficient storage
- **Georeferenced** with full coordinate system metadata

## Features

- 🗺️ **Interactive tile browser** - Select from 30,000+ tiles globally
- 🔍 **Cosine similarity explorer** - Click any pixel to find similar areas
- 📊 **Real-time statistics** - Histograms, thresholds, coverage metrics
- 💾 **GeoTIFF export** - Download georeferenced similarity maps and masks
- ☁️ **100% cloud-based** - No downloads, no local storage needed

## The Magic of Cloud Optimized GeoTIFFs (COGs)

This app demonstrates how **COGs enable efficient access to massive raster datasets** without downloading entire files.

### How COGs Work

A Cloud Optimized GeoTIFF contains **pre-computed resolution pyramids** (overviews) baked into a single file:

```
┌─────────────────────────────────────┐
│           COG File Structure        │
├─────────────────────────────────────┤
│  Header (tile index, metadata)      │  ← ~1KB
├─────────────────────────────────────┤
│  Full Resolution (10000 x 10000)    │  ← Level 0
├─────────────────────────────────────┤
│  Overview 1 (5000 x 5000)           │  ← 2x downsample
├─────────────────────────────────────┤
│  Overview 2 (2500 x 2500)           │  ← 4x downsample
├─────────────────────────────────────┤
│  Overview 3 (1250 x 1250)           │  ← 8x downsample
├─────────────────────────────────────┤
│  Overview 4 (625 x 625)             │  ← 16x downsample
└─────────────────────────────────────┘
```

### HTTP Range Requests

When you select a downsample factor (e.g., 8x), the app:

1. **Reads the header** (~1KB) to find where the 8x overview lives
2. **Fetches only those bytes** via HTTP Range Request
3. **Returns pre-downsampled data** - no computation needed!

```
GET /file.tif
Range: bytes=0-1023           # Header only

GET /file.tif  
Range: bytes=450000-460000    # Just the overview tiles needed
```

**Result**: A 500MB GeoTIFF loads in seconds by downloading only ~1-2MB.

### Why This Matters for Embeddings

AlphaEarth embeddings are **64-band GeoTIFFs** - each pixel has 64 values. At full resolution, these files are huge. But with COGs:

- **Downsampled embeddings preserve semantic meaning** - a 8x8 average of forest pixels still represents "forest"
- **Interactive exploration becomes possible** - load any tile in seconds
- **No preprocessing required** - the pyramids are pre-built

This technique works for any high-dimensional raster data: hyperspectral imagery, model outputs, feature maps, etc.

## How It Works

### Embedding Space

The 64-dimensional embedding space captures semantic meaning:
- **Similar land cover types** (forests, water, urban) cluster together
- **Cosine similarity** measures how semantically similar two pixels are
- Values range from -1 (opposite) to +1 (identical)

### De-quantization

AlphaEarth embeddings are stored as int8 and converted back to floats:

```python
def dequantize(value):
    return ((value / 127.5) ** 2) * sign(value)
```

## Quick Start

### Run Locally

```bash
# Clone the repo
git clone https://github.com/om2468/alpha_web.git
cd alpha_web

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run similarity_explorer_cloud.py
```

### Deploy to Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `similarity_explorer_cloud.py`

## Tech Stack

- **Streamlit** - Web interface
- **Rasterio + GDAL** - COG streaming via `/vsicurl/`
- **DuckDB** - Query remote Parquet tile index
- **Plotly** - Interactive maps
- **NumPy** - Similarity computation

## Data Source

Embeddings from [TGE Labs AlphaEarth Foundation](https://source.coop/tge-labs/aef) via Source Cooperative.

## License

MIT
