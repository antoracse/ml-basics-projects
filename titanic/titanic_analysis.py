import pandas as pd

# Load the CSV file
df = pd.read_csv("train.csv")

# Print confirmation and the first 5 rows
print("✅ Dataset loaded successfully!\n")
print(df.head())
# ------------------------------------------
# Step 3 — Explore the dataset
# ------------------------------------------

# Show basic info about the dataset
print("\n📊 Dataset Info:")
print(df.info())

# Check for missing values
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# Show summary statistics for numerical columns
print("\n📈 Summary Statistics:")
print(df.describe())
# ------------------------------------------
# Step 4 — Data Cleaning
# ------------------------------------------

# Fill missing 'Age' values with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the most frequent value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column since too many values are missing
df.drop('Cabin', axis=1, inplace=True)

# Confirm that no missing values remain
print("\n✅ After cleaning, missing values are:")
print(df.isnull().sum())
# Step 4: Basic Data Analysis

# 1. Basic info about the dataset
print("\n📊 Basic Information:")
print(df.info())

# 2. Descriptive statistics
print("\n📈 Summary Statistics:")
print(df.describe())

# 3. Count of survivors and non-survivors
print("\n🚢 Survival Count:")
print(df['Survived'].value_counts())

# 4. Average age of survivors vs non-survivors
print("\n👩‍🦱 Average Age by Survival:")
print(df.groupby('Survived')['Age'].mean())

# 5. Average fare by class
print("\n💰 Average Fare by Class:")
print(df.groupby('Pclass')['Fare'].mean())

# step 5: Visualization
import matplotlib.pyplot as plt

# Count of survived vs not survived
survival_counts = df["Survived"].value_counts()

# Plot
plt.bar(survival_counts.index, survival_counts.values, color=["red", "green"])
plt.title("Survival Count in Titanic Dataset")
plt.xlabel("0 = Not Survived, 1 = Survived")
plt.ylabel("Number of Passengers")

# Show the plot
plt.show()

