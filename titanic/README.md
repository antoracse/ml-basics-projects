# 🛳 Titanic Survival Prediction — Machine Learning Project  
**Author:** Antora Akter  
**Date:** February 2022  

---

## 📖 Overview  
This project focuses on predicting the **survival of passengers aboard the Titanic** using **Machine Learning techniques**.  
By analyzing passenger data such as **age, gender, class, and fare**, the model identifies key factors that influenced survival chances.  

The Titanic dataset is one of the most iconic beginner-friendly datasets from **Kaggle**, ideal for practicing **data preprocessing**, **feature engineering**, and **model evaluation**.

---

## 🎯 Objectives  
- Understand and clean real-world data (handle missing values, encode categorical features).  
- Build predictive models to classify whether a passenger survived.  
- Compare model performances and identify the most accurate one.  

---

## 🧩 Dataset Information  
The dataset, sourced from the **Kaggle Titanic Competition**, contains details of **891 passengers**, including:  
| Feature | Description |
|----------|--------------|
| `Pclass` | Passenger Class (1st, 2nd, 3rd) |
| `Sex` | Gender |
| `Age` | Age of Passenger |
| `SibSp` | Number of Siblings/Spouses aboard |
| `Parch` | Number of Parents/Children aboard |
| `Fare` | Ticket Fare |
| `Embarked` | Port of Embarkation |

---

## ⚙️ Technologies & Libraries  
- 🐍 **Python 3**  
- **Pandas**, **NumPy** — Data cleaning and manipulation  
- **Scikit-learn (sklearn)** — Model building and evaluation  
- **Matplotlib**, **Seaborn** — Data visualization (optional)  

---

## 🧠 Models Used  
| Model | Accuracy |
|--------|-----------|
| Logistic Regression | 0.73 |
| Decision Tree Classifier | 0.72 |
| Random Forest Classifier | 0.73 |

All models were trained and tested using an **80–20 split**, ensuring fair comparison.

---

## 🚀 Project Workflow  
1. **Data Loading and Cleaning**  
   - Imported dataset and handled missing values.  
   - Encoded categorical variables (e.g., gender, embarked).  

2. **Feature Selection**  
   - Selected relevant features for model training after exploratory analysis.  

3. **Model Building and Evaluation**  
   - Trained multiple ML models and evaluated accuracy.  
   - Compared their performance to identify the most reliable model.  

---

## 📊 Results Summary  
- Both **Logistic Regression** and **Random Forest** achieved around **73% accuracy**.  
- Feature analysis showed that **gender and passenger class** were the most influential factors in survival prediction.  

---
## 🧾 Note from the Author  
> This project was originally developed during my learning phase and is now being organized and uploaded to GitHub for professional presentation.  
> The goal is to showcase my understanding of fundamental ML concepts and clean coding practices.

---
## 🏁 Conclusion  
Although the overall accuracy is moderate, this project demonstrates a strong understanding of:  
- Data preprocessing and cleaning  
- Feature engineering  
- Model training and evaluation  

This foundational project lays the groundwork for more advanced **machine learning and data science** work in the future.

---

⭐ *Developed with curiosity and care by* **Antora Akter**  
🎓 *Computer Science and Engineering Student, City University, Dhaka*
