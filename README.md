# Edge Map Extraction of High Resolution Facial Images

A computer vision project implementing an advanced edge detection algorithm specifically designed for high-resolution facial images, combining directional gradients, weak/strong edge detection, and Zhang-Suen thinning for optimal facial contour extraction.

## Overview

This project addresses the challenge of extracting precise edge maps from high-resolution facial images while maintaining a balance between preserving fine details and highlighting facial contours. The algorithm outperforms traditional methods like Canny and HED by effectively capturing both low-level and high-level facial features.

## Algorithm Pipeline

The edge detection process follows a six-step methodology:

1. **Gaussian Smoothing** - Noise reduction using optimal σ value (σ=2)
2. **Directional Gradients** - Horizontal and vertical gradient computation
3. **Weak Edge Detection** - Curved boundary detection with adaptive thresholding
4. **Strong Edge Detection** - Composite gradient variation analysis
5. **Edge Merging** - Combining weak and strong edge maps
6. **Zhang-Suen Thinning** - Skeletal edge refinement

## Key Features

- **High-Resolution Optimization**: Specifically designed for facial images with exceptional resolution
- **Dual-Threshold Detection**: Separate handling of weak and strong edges for comprehensive coverage
- **Adaptive Smoothing**: Optimal σ parameter selection for noise reduction without detail loss
- **Facial-Specific Design**: Tailored for facial features like eyes, nose, mouth, and overall face contour
- **Balanced Detail Preservation**: Maintains fine details while avoiding excessive low-level noise
- **Comparative Performance**: Superior results compared to Canny and HED algorithms

## Performance Evaluation

### Dataset Testing
- **Primary Dataset**: ~10,000 high-resolution facial images (1024×1024)
- **Secondary Dataset**: Mixed resolution images for adaptability validation
- **Human Evaluation**: Subjective analysis by 39 volunteers
- **Cross-Resolution Testing**: Performance validation across different image resolutions

### Testing Methodology
- **Quantitative Analysis**: Objective metrics and parameter optimization
- **Comparative Study**: Direct comparison with Canny and HED algorithms
- **Subjective Assessment**: Human evaluator feedback on visual quality
- **Multi-Dataset Validation**: Algorithm robustness across diverse image sets

## 🏆 Comparative Results

### Algorithm Performance Comparison

| Algorithm | Facial Outline | Fine Details | Edge Continuity | Overall Quality |
|-----------|---------------|--------------|-----------------|-----------------|
| Canny     | Weak          | Poor         | Fragmented      | Low            |
| HED       | Good          | Limited      | Moderate        | Medium         |
| **Proposed** | **Excellent** | **High**     | **Strong**      | **Superior**   |
