# Discriminative Dictionary Learning for (2D) Image Segmentation


## Setup

**Environment**: The following software/libraries are needed:
- [SPAMS 2.6.*](http://spams-devel.gforge.inria.fr/)
- [numpy 1.15.1](https://docs.scipy.org/doc/numpy/user/quickstart.html)   
- [matplotlib 2.0.2](https://matplotlib.org/users/installing.html)
- [scikit-image 0.14.1](https://scikit-image.org)
- [scikit-learn *.19.1](https://scikit-learn.org)
 
**Datasets**: The following datasets are used in our experiments:
- [DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/)
- [STARE](http://www.ces.clemson.edu/~ahoover/stare/) 

**Data preprocessing**: All the images are preprocessed using:
- Gray scale conversion 
- CLAHE normalization 
