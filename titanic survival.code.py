import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
titanicdata_set = pd.read_csv(r"F:\project.py\data set predection\titanic survival prediction\titonic survival data_set.csv")

# Select only numerical columns for training
x = titanicdata_set[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]  
y = titanicdata_set["Survived"]  # Target variable

# Convert categorical features to numerical values
x["Sex"] = x["Sex"].map({"male": 1, "female": 0})  # Male = 1, Female = 0
x["Embarked"] = x["Embarked"].map({"C": 0, "Q": 1, "S": 2})  # Embarked C=0, Q=1, S=2

# Fill missing values
x = x.fillna(x.mean())  

# Train the Decision Tree Model
model = DecisionTreeClassifier()
model.fit(x, y)

# Get user input
Pclass = int(input("Enter Pclass (1, 2, or 3): "))
Sex = input("Enter Sex (male/female): ")
Age = float(input("Enter Age: "))
SibSp = int(input("Enter SibSp (No. of siblings/spouses aboard): "))
Parch = int(input("Enter Parch (No. of parents/children aboard): "))
Fare = float(input("Enter Fare amount: "))
Embarked = input("Enter Embarked location (C/Q/S): ")

# Convert user input to match the model's format
Sex = 1 if Sex.lower() == "male" else 0
Embarked_dict = {"C": 0, "Q": 1, "S": 2}
Embarked = Embarked_dict.get(Embarked.upper(), -1)

# Ensure the input is in the correct format
sample_input = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

# Make the prediction
prediction = model.predict(sample_input)

# Display the result
print("Survival Prediction:", "YES (Survived)" if prediction[0] == 1 else "NO (Not Survived)") 