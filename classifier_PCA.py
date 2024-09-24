import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from PCA_self import PCA_self

# Load and preprocess the data
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to calculate the testing accuracy
def classifier_PCA(X_train, X_test, y_train, y_test, k):
    """
    X_train: numpy array of shape (n1, d),
       where n1 is the number of training points and d is the dimensionality
    X_test: numpy array of shape (n2, d),
       where n2 is the number of testing points and d is the dimensionality
    y_train: numpy array of shape (n1,),
       where n1 is the number of training points and each element is
       either 0 or 1
    y_test: numpy array of shape (n2,),
       where n2 is the number of testing points and each element is
       either 0 or 1
    k: int, number of principal components to use in PCA

    Returns: testing_accuracy: float, accuracy of decision tree
             classifier trained on PCA dimensionality reduced
             testing data set

    Assume the data is properly scaled
    """
    # Perform PCA on the training data
    U, _ = PCA_self(X_train, k)

    # Project the training and testing data onto the k principal components
    X_train_pca = X_train @ U
    X_test_pca = X_test @ U

    # Train the decision tree classifier on the PCA-reduced training data
    clf = DecisionTreeClassifier()
    clf.fit(X_train_pca, y_train)

    # Predict on the PCA-reduced testing data
    y_pred = clf.predict(X_test_pca)

    # Calculate the testing accuracy
    testing_accuracy = accuracy_score(y_test, y_pred)

    return testing_accuracy

# Test the implementation with k = 2 principal components
k = 2
testing_accuracy = classifier_PCA(X_train_scaled, X_test_scaled, y_train, y_test, k)
print(f"Testing accuracy with {k} principal components: {testing_accuracy:.4f}")