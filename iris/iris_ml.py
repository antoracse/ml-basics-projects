# ==========================================================
# 🌸 IRIS FLOWER CLASSIFICATION PROJECT
# Author: Antora Akter
# Date: September 2022
# ==========================================================


# 🧩 STEP 1: IMPORTING LIBRARIES & LOADING DATA

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

print("🔹 Sample Data:")
print(df.head())


# 🔍 STEP 2: DATA ANALYSIS & VISUALIZATION

print("\n❌ Missing Values:\n", df.isnull().sum())

plt.figure(figsize=(6,4))
sns.countplot(x='species', data=df, palette='Set2')
plt.title("🪷 Distribution of Iris Species")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap="YlGnBu")
plt.title("🔗 Feature Correlation Heatmap")
plt.show()


# ⚙️ STEP 3: DATA PREPARATION

X = df.drop(['target', 'species'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 STEP 4: MODEL TRAINING & EVALUATION

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='rbf', C=1),
    "Naive Bayes": GaussianNB()
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    print(f"\n🔹 {name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-"*60)

# 📊 STEP 5: MODEL PERFORMANCE COMPARISON

plt.figure(figsize=(7,4))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="Set3")
plt.title("📈 Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)

for i, acc in enumerate(accuracies.values()):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', fontsize=10, weight='bold')

plt.show()

best_model = max(accuracies, key=accuracies.get)
print(f"\n🏆 Best Performing Model: {best_model} ({accuracies[best_model]:.2f})")

# ✅ STEP 6: CONCLUSION

print("\n🎯 Conclusion:")
print("This project explored multiple machine learning models for Iris flower classification.")
print("After comparison, the best-performing model was:", best_model)
print("All models demonstrated strong accuracy, confirming effective feature relationships in the Iris dataset.")
