import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    """Loads the Iris dataset and returns a DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df
