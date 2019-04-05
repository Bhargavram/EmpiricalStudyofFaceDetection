# The implementation of Viola-Jones Algorithm

Participants:
- Hakju Oh hakju@umbc.edu
- John Zhu zhujohn64@gmail.com
- Bhargav Ram Mandadi sh27293@umbc.edu

This project aims to implement Viola-Jones Algorithm [1] and improve its performance.

## Python Libraries Required
- numpy>=1.14.2
- matplotlib>=2.2.2
- Pillow>=5.1.0
- scikit-learn>=0.19.1
- scipy>=1.1.0
- incremental>=17.5.0
- twisted>=18.4.0

## Usage
1. Facial features from the image are extracted by executing `gabor_extract.py`
2. Integral image with the facial features is formed by executing `integral_image.py`
3. Build a strong classifier by combining the weak classifiers usind `cascade_detector.py`


## Features
Viola, P., & Jones, M. [1] used three different kinds of features in the algorithm:
  
<img src="docs/images/original_features.png" width="239" height="209">

Lienhart, R., & Maydt, J. [2] introduced an extended set of twisted Haar-like feature:

<img src="docs/images/extended_features.png" width="239" height="209">


## References

1. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. In Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on (Vol. 1, pp. I-I). IEEE.
2. Lienhart, R., & Maydt, J. (2002). An extended set of haar-like features for rapid object detection. In Image Processing. 2002. Proceedings. 2002 International Conference on (Vol. 1, pp. I-I). IEEE.
 

