import numpy as np
from PCA_self import PCA_self

def PCA_variance_captured(X, k):
    """
    X: numpy array of shape (n, d),
       where n is the number of data points and d is the dimensionality.
       assume that the data is already standardized and scaled
    k: int, number of principal components to use

    Returns: variance_explained: float, % of variance explained
             by the top k principal components
    """
    # Perform PCA on the data
    U, S = PCA_self(X, k)

    # Calculate the total variance
    total_variance = np.sum(S)

    # Calculate the variance explained by the top k principal components
    variance_explained = np.sum(S[:k]) / total_variance * 100

    return variance_explained

k = 2
variance_explained = PCA_variance_captured(X_train_scaled, k)
print(f"Variance explained by the top {k} principal components: {variance_explained:.2f}%")