from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model(df, model_path="../models/iris_model.joblib"):
    """Train the model and save it to disk."""
    X = df.drop(columns=['target', 'species'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")

    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    return model, acc
