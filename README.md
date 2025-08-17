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

## Comparative Results

### Algorithm Performance Comparison

| Algorithm | Facial Outline | Fine Details | Edge Continuity | Overall Quality |
|-----------|---------------|--------------|-----------------|-----------------|
| Canny     | Weak          | Poor         | Fragmented      | Low            |
| HED       | Good          | Limited      | Moderate        | Medium         |
| **Proposed** | **Excellent** | **High**     | **Strong**      | **Superior**   |

### Visualisation

## Applications
- **Face Recognition**: Enhanced feature extraction for identification systems
- **Facial Analysis**: Detailed study of facial structure and features
- **Biometric Systems**: Robust facial contour detection for security applications
- **Computer Vision**: Preprocessing for facial landmark detection
- **Medical Imaging**: Facial analysis in medical diagnosis
- **Animation/VR**: Character modeling and facial animation

## Citation
```
@InProceedings{10.1007/978-981-99-8612-5_26,
author="Timmanagoudar, Harshit
and Krishna, Gandra Sai
and Koti, Anirudh
and Jolad, Harsh
and Preethi, P.",
editor="So In, Chakchai
and Londhe, Narendra D.
and Bhatt, Nityesh
and Kitsing, Meelis",
title="Edge Map Extraction of High-Resolution Facial Images",
booktitle="Information Systems for Intelligent Systems",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="319--334",
abstract="This work focuses on edge detection, a vital task in several fields, such as computer vision, image processing, and pattern recognition. Since exact facial analysis and recognition have numerous applications in areas like biometrics, surveillance, and human-computer interaction, this study's focus is specifically on obtaining precise edge maps of facial pictures with exceptional resolution. Several essential processes are carried out in order to produce accurate edge maps. In order to lessen noise and attenuate the very low-level features in high-resolution photographs, a smoothing operation is first conducted. Utilizing a variety of edge detection methods, such as directional gradients and cumulative gradient magnitude, variations in pixel intensities along various directions are examined. The Zhang-Suen technique, renowned for its efficiency in thinning binary pictures, is used to remove extraneous pixels from the discovered edges to further refine them. By maintaining only the necessary boundary information, this phase ensures that the edge maps produced are more accurate. According to subjective and comparative evaluations and the outcomes of applying the algorithm to face photos from various datasets, the suggested approach is suitable for building edge maps of facial photographs with high resolution as well as low resolution.",
isbn="978-981-99-8612-5"
}
```

## Authors
- **Harshit Timmanagoudar** - harshit.utd@gmail.com
- **Gandra Sai Krishna** - saikrishnag03@gmail.com
- **Anirudh Koti** - ani040702@gmail.com
- **Harsh Jolad** - harshjolad@gmail.com
- **Dr. Preethi P** - preethip@pes.edu (Supervisor)
