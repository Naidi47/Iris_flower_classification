# 🌸 Iris Flower Classification – Machine Learning Project

## 📘 Overview
This project aims to classify **Iris flowers** into three species — *Setosa*, *Versicolor*, and *Virginica* — based on their **sepal** and **petal** measurements.  
It is one of the most famous beginner-friendly datasets in machine learning and helps you understand the **entire ML workflow** — from data exploration to model deployment.

---

## 🎯 Project Objective
Build a **machine learning model** that can accurately predict the Iris flower species based on the given input features:
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width

---

## 📊 Dataset Information
The **Iris dataset** is available directly in the `scikit-learn` library.

| Feature | Description | Unit |
|----------|--------------|------|
| sepal length (cm) | Length of the sepal | cm |
| sepal width (cm) | Width of the sepal | cm |
| petal length (cm) | Length of the petal | cm |
| petal width (cm) | Width of the petal | cm |
| species | Target variable (Setosa, Versicolor, Virginica) | - |

**Dataset size:** 150 samples  
**Classes:** 3 (Setosa, Versicolor, Virginica)  

---

## 🧱 Project Structure

```
iris_flower_classification/
│
├── data/
│   └── (optional dataset files)
│
├── notebooks/
│   └── iris_classification.ipynb       # Main Jupyter notebook
│
├── models/
│   ├── iris_model.joblib               # Trained model file
│   └── scaler.joblib                   # Scaler for preprocessing
│
├── app/
│   └── app.py                          # Streamlit web app (for deployment)
│
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

---

## ⚙️ Steps Followed

### 1️⃣ Import Libraries
Imported all required libraries for data manipulation, visualization, and machine learning (Pandas, Seaborn, scikit-learn).

### 2️⃣ Load Dataset
Loaded the built-in Iris dataset using:
```python
from sklearn.datasets import load_iris
```

### 3️⃣ Exploratory Data Analysis (EDA)
- Checked for missing values and data types  
- Visualized pairplots and heatmaps  
- Analyzed correlations between features

### 4️⃣ Data Preparation
- Split data into **train** and **test** sets (80/20)
- Scaled numerical features using `StandardScaler`

### 5️⃣ Model Building
Trained and compared three models:
- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)

### 6️⃣ Model Evaluation
Evaluated using metrics:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

### 7️⃣ Save Best Model
Saved the best-performing model using:
```python
import joblib
joblib.dump(model, '../models/iris_model.joblib')
```

### 8️⃣ Test on New Data
Predicted species for a new input sample.

---

## 📈 Results

| Model | Accuracy |
|--------|-----------|
| Logistic Regression | ~96% |
| Decision Tree | ~95% |
| K-Nearest Neighbors | ~97% |

✅ **KNN performed best** on this dataset.

---

## 🌐 Deployment (Optional)

You can create a simple web interface using **Streamlit**:

```bash
streamlit run app/app.py
```

Then open the local URL (shown in terminal) to interact with your Iris classifier.

---

## 🧰 Technologies Used
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit (optional for deployment)

---

## ✨ Author
**Brahmanaidu **  
*Data Science Enthusiast | Machine Learning Learner*
