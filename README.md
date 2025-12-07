# AlphaEarth Embedding Analysis

A Python toolkit for exploring and analyzing [AlphaEarth](https://www.alphaearthdata.com/) satellite imagery embeddings. This project replicates and extends the techniques demonstrated in [Element84's blog post on exploring AlphaEarth embeddings](https://element84.com/machine-learning/exploring-alphaearth-embeddings/).

## What is AlphaEarth?

AlphaEarth provides **pixel-level embedding vectors** derived from satellite imagery. Each pixel in an image is represented as a **64-dimensional vector** that encodes semantic information about that location—essentially a "fingerprint" of what the land looks like and represents.

These embeddings are:
- **Pre-computed** from satellite imagery (Sentinel-2)
- **Stored as GeoTIFFs** with 64 bands (one per dimension)
- **Quantized to int8** values [-127, 127] for efficient storage
- **Georeferenced** so they align with real-world coordinates

## How It Works

### Embedding Space

The 64-dimensional embedding space captures semantic meaning:
- **Similar land cover types** (forests, water, urban areas) cluster together in this space
- **Cosine similarity** between vectors measures how semantically similar two pixels are
- Values range from -1 (opposite) to +1 (identical)

### De-quantization

AlphaEarth embeddings are stored as int8 values and must be converted back to floats:

```python
def dequantize(value):
    return ((value / 127.5) ** 2) * sign(value)
```

This non-linear transformation recovers the original embedding values from the compressed storage format.

## Project Components

### 1. Jupyter Notebook (`notebooks/alphaearth_analysis.ipynb`)

A comprehensive analysis notebook demonstrating 8 techniques:

| Section | Technique | Description |
|---------|-----------|-------------|
| 1 | Setup | Load data, handle NoData values, memory optimization |
| 2 | False Color Composite | Visualize 3 embedding dimensions as RGB |
| 3 | Individual Dimensions | Explore what each of the 64 dimensions represents |
| 4 | PCA | Reduce 64D → 3D for visualization |
| 5 | K-Means Clustering | Unsupervised land cover classification |
| 6 | Cosine Similarity | Find similar areas to a reference point |
| 7 | Change Detection | Compare embeddings between 2018 and 2024 |
| 8 | Interactive Exploration | Combine techniques for analysis |

### 2. Streamlit App (`similarity_explorer.py`)

An interactive web tool for similarity search:

- **Click-to-select**: Click anywhere on the image to select a reference point
- **Real-time similarity**: See which pixels are most similar to your selection
- **Adjustable threshold**: Fine-tune what counts as "similar"
- **GeoTIFF export**: Download results for use in GIS software

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python 3.x** | Core programming language |
| **NumPy** | Array operations and linear algebra |
| **Rasterio** | Reading/writing GeoTIFF files |
| **Matplotlib** | Visualization and plotting |
| **scikit-learn** | PCA, K-Means, and other ML algorithms |
| **Streamlit** | Interactive web application |
| **streamlit-image-coordinates** | Click-to-select functionality |
| **Jupyter** | Interactive notebook environment |

## Data Format

### Input GeoTIFFs

- **Dimensions**: 8192 × 8192 pixels (original resolution)
- **Bands**: 64 (one per embedding dimension)
- **Data type**: int8 [-127, 127]
- **NoData value**: -128
- **CRS**: EPSG:32630 (UTM Zone 30N)
- **Location**: Somerset, UK

### Memory Optimization

Full-resolution data requires ~4GB per year. The tools include **downsampling** (default 4×) to work efficiently on machines with limited RAM (tested on 16GB MacBook M2).

## Usage

### Jupyter Notebook

```bash
cd notebooks
jupyter notebook alphaearth_analysis.ipynb
```

### Streamlit App

```bash
streamlit run similarity_explorer.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
alpha/
├── README.md                 # This file
├── similarity_explorer.py    # Streamlit interactive app
├── notebooks/
│   └── alphaearth_analysis.ipynb  # Analysis notebook
├── input_data/
│   ├── xbckr5wa0l2omaauu-*.tiff   # 2018 embeddings
│   └── xks7764uu0jo1h8jh-*.tiff   # 2024 embeddings
└── .venv/                    # Python virtual environment
```

## Key Concepts

### Cosine Similarity

Measures the angle between two vectors, ignoring magnitude:

$$\text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| \cdot |\mathbf{b}|}$$

- **1.0** = identical direction (same land cover type)
- **0.0** = orthogonal (unrelated)
- **-1.0** = opposite direction

### PCA (Principal Component Analysis)

Reduces 64 dimensions to 3 for RGB visualization while preserving maximum variance. Helps reveal structure in the embedding space.

### K-Means Clustering

Groups pixels into K clusters based on embedding similarity. Effectively performs unsupervised land cover classification.

### Change Detection

Compares embeddings from different years:
- **Euclidean distance** between 2018 and 2024 vectors
- High distance = significant change (development, deforestation, etc.)
- Low distance = stable land cover

## References

- [Element84 Blog: Exploring AlphaEarth Embeddings](https://element84.com/machine-learning/exploring-alphaearth-embeddings/)
- [AlphaEarth Data](https://www.alphaearthdata.com/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

This project is for educational and research purposes, demonstrating techniques for working with AlphaEarth embedding data.
