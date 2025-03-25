# 🌟 Principal Component Analysis (PCA) on the Iris Dataset
🚀 **Dimensionality Reduction & Visualization using PCA**


## 📌 Project Overview
Principal Component Analysis (PCA) is a **dimensionality reduction technique** that transforms high-dimensional data into a lower-dimensional space while preserving as much information as possible.  

In this project, we apply **PCA to the famous Iris dataset**, reducing four features to two **principal components**, making visualization easier while retaining maximum variance.  

---

## 📂 Dataset Information: Iris Dataset
- **Total Samples:** 150  
- **Features:**  
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  
- **Target Variable:** Species (Setosa, Versicolor, Virginica)  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris)  

---

## 📖 What is PCA?
Principal Component Analysis (PCA) is a **statistical technique** used to:  
✅ Reduce the number of dimensions in a dataset  
✅ Remove redundancy while maintaining the most significant features  
✅ Help visualize high-dimensional data in **2D or 3D space**  

### 📊 Why Use PCA?
✔ **Feature Reduction**: Removes correlated features  
✔ **Visualization**: Transforms high-dimensional data into an interpretable format  
✔ **Speeds Up ML Models**: Reduces computational cost  
✔ **Removes Noise**: Improves model performance  

---

## ⚡ Step-by-Step Implementation
### 📌 Step 1: Load the Dataset
```python
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
df.head()
```

### 📌 Step 2: Data Visualization (Before PCA)
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue="species", diag_kind="kde")
plt.suptitle("Pairplot of Iris Dataset (Before PCA)", y=1.02)
plt.show()
```
### 📌 Step 3: Standardizing the Data
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
```

### 📌 Step 4: Applying PCA
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['species'] = df['species']
df_pca.head()
```
###  📌 Step 5: Explained Variance Ratio
```python
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
```
### 📌 Step 6: PCA Visualization
```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['species'], palette="viridis", s=100)
plt.title("PCA Visualization of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Species")
plt.grid(True)
plt.show()
```

### 📌 Step 7: Scree Plot (Variance Explained)
```python
plt.figure(figsize=(8, 5))
plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_, color=['blue', 'green'])
plt.xlabel("Principal Components")
plt.ylabel("Variance Explained")
plt.title("Scree Plot of PCA Components")
plt.show()
```

### 📌 Step 8: Heatmap of PCA Components
```python
plt.figure(figsize=(6, 4))
sns.heatmap(pca.components_, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=iris.feature_names, yticklabels=['PC1', 'PC2'])
plt.title("Heatmap of PCA Component Contributions")
plt.show()
```

## 📊 Key Insights & Conclusion

✔ PCA successfully **reduced the 4D dataset into 2D**, making visualization easier.  
✔ The **explained variance ratio** shows that most information is retained.  
✔ **Setosa is well-separated**, while **Versicolor & Virginica** show some overlap.  
✔ The **Scree Plot confirms PC1** is the most important.  
✔ The **Heatmap shows feature contribution**, helping in interpretation.  

📌 **Final Takeaway:**  
PCA is an **excellent tool** for **dimensionality reduction and visualization**, helping in **data preprocessing and feature selection** for machine learning models.



