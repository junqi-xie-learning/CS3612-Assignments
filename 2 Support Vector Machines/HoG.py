import numpy as np
from skimage.feature import hog


def hog_features(X):
    '''
    Compute the HoG features for all the images in X.
    '''
    features = []
    for image in X:
        feature = hog(image, orientations=9, pixels_per_cell=(
            2, 2), cells_per_block=(1, 1))
        features.append(feature)
    return np.array(features)
