import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from PCA_self import PCA_self  

def ComparePCA(X_train, X_test, y_train, y_test, variances):
    """
    Compare the accuracies of decision trees trained on data reduced by PCA_self and sklearn's PCA.
    
    Parameters:
        variances (list): List of variances to explain in PCA, which determines the number of components.
    """
    self_accuracies = []
    sklearn_accuracies = []

    # Decision tree training and testing using PCA_self
    for variance in variances:
        # Calculate number of components to keep this variance
        pca = PCA(n_components=variance)
        pca.fit(X_train)
        k = pca.n_components_
        
        U, _ = PCA_self(X_train, k)
        X_train_pca = X_train @ U
        X_test_pca = X_test @ U

        clf = DecisionTreeClassifier()
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        self_accuracies.append(accuracy_score(y_test, y_pred))

    # Decision tree training and testing using sklearn PCA
    for variance in variances:
        pca = PCA(n_components=variance)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = DecisionTreeClassifier()
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        sklearn_accuracies.append(accuracy_score(y_test, y_pred))

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.scatter(variances, self_accuracies, color='blue', label='PCA_self')
    plt.scatter(variances, sklearn_accuracies, color='red', label='sklearn PCA')
    plt.title('Comparison of Decision Tree Accuracies with PCA_self vs sklearn PCA')
    plt.xlabel('Explained Variance')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load and preprocess data
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Variance levels to test
variances = [0.80, 0.85, 0.90, 0.95, 0.99]

# Running the comparison
ComparePCA(X_train_scaled, X_test_scaled, y_train, y_test, variances)