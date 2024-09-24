#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def LDA_visualization(X_train, X_test, y_train, y_test):
    """
    X_train: numpy array of shape (n1, d),
       where n1 is the number training points and d is the dimensionality
    X_test: numpy array of shape (n2, d),
       where n2 is the number testing points and d is the dimensionality
    y_train: numpy array of shape (n1,),
       where n1 is the number training points and each element is
       either 0 or 1
    y_train: numpy array of shape (n2,),
       where n2 is the number testing points and each element is
       either 0 or 1

    Returns: N/A

    You may assume the data is standardized and scaled.
    """
    # Step 1: Compute the projection vector w
    # Compute the mean of each class
    mean_0 = np.mean(X_train[y_train == 0], axis=0)
    mean_1 = np.mean(X_train[y_train == 1], axis=0)

    # Compute the covariance matrix of each class
    cov_0 = np.cov(X_train[y_train == 0].T)
    cov_1 = np.cov(X_train[y_train == 1].T)

    # Compute the prior probability of each class
    prior_0 = len(y_train[y_train == 0]) / len(y_train)
    prior_1 = len(y_train[y_train == 1]) / len(y_train)

    # Compute the inverse of the sum of the covariance matrices
    cov_sum_inv = np.linalg.inv(cov_0 + cov_1)

    # Compute the mean difference
    mean_diff = mean_0 - mean_1

    # Compute the projection vector w
    w = cov_sum_inv @ mean_diff

    # Step 2: Project the data onto the projection vector w
    # Project the training data
    X_train_projected = X_train @ w

    # Project the testing data
    X_test_projected = X_test @ w

    # Step 3: Plot the projected data
    # Create a scatter plot of the projected training data
    plt.scatter(X_train_projected[y_train == 0], np.zeros(len(X_train_projected[y_train == 0])),
                color='blue', label='Class 0')
    plt.scatter(X_train_projected[y_train == 1], np.zeros(len(X_train_projected[y_train == 1])),
                color='red', label='Class 1')

    # Add labels and title to the plot
    plt.xlabel('Projected Data')
    plt.ylabel('Density')
    plt.title('LDA Visualization')
    plt.legend()

    # Show the plot
    plt.show()


    from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the breast cancer data set
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Visualize the projected data using LDA
LDA_visualization(X_train_scaled, X_test_scaled, y_train, y_test)