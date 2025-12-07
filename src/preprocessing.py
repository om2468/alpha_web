"""
Preprocessing utilities for AlphaEarth embeddings.
"""

import numpy as np
import rasterio


def load_embeddings(filepath: str) -> tuple[np.ndarray, dict]:
    """
    Load AlphaEarth embeddings from a GeoTIFF file.
    
    Args:
        filepath: Path to the GeoTIFF file with 64-band embeddings
        
    Returns:
        embeddings: Array of shape (height, width, 64)
        profile: Rasterio profile with georeferencing info
    """
    with rasterio.open(filepath) as src:
        # Read all bands: shape (64, height, width)
        data = src.read()
        profile = src.profile.copy()
    
    # Transpose to (height, width, 64) for easier processing
    embeddings = np.transpose(data, (1, 2, 0))
    return embeddings, profile


def dequantize(values: np.ndarray, nodata_value: int = -128) -> np.ndarray:
    """
    De-quantize int8 AlphaEarth embeddings to float [-1, 1].
    
    The AlphaEarth embeddings are stored as signed 8-bit integers (-127 to 127).
    This function converts them back to their original float representation.
    
    Formula: de_quantized = ((value / 127.5) ** 2) * sign(value)
    
    Args:
        values: Int8 array of embeddings
        nodata_value: Value representing NoData (default -128)
        
    Returns:
        Float32 array with values in [-1, 1], NoData as NaN
    """
    # Create mask for NoData
    nodata_mask = values == nodata_value
    
    # Convert to float and apply de-quantization formula
    values_float = values.astype(np.float32)
    result = ((values_float / 127.5) ** 2) * np.sign(values_float)
    
    # Set NoData to NaN
    result[nodata_mask] = np.nan
    
    return result


def z_normalize(arr: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Z-normalize an array and clip to specified standard deviations.
    
    Useful for visualizing individual embedding dimensions with better contrast.
    
    Args:
        arr: Input array
        sigma: Number of standard deviations to clip to
        
    Returns:
        Z-normalized and clipped array
    """
    # Calculate mean and std ignoring NaN
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    
    if std == 0:
        return np.zeros_like(arr)
    
    # Z-normalize
    z = (arr - mean) / std
    
    # Clip to sigma standard deviations
    return np.clip(z, -sigma, sigma)


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] range for display.
    
    Args:
        arr: Input array
        
    Returns:
        Array normalized to [0, 1]
    """
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    
    return (arr - min_val) / (max_val - min_val)
