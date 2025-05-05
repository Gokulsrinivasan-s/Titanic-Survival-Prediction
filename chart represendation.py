import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load dataset
titanicdata_set = pd.read_csv(r"F:\project.py\data set predection\titanic survival prediction\titonic survival data_set.csv")

# Selecting relevant features (excluding non-numeric ones like "Name" and "Ticket")
x = titanicdata_set[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = titanicdata_set["Survived"]  # Target variable

# Convert categorical variables (Sex, Embarked) into numerical values
x["Sex"] = x["Sex"].map({"male": 1, "female": 0})  
x["Embarked"] = x["Embarked"].map({"C": 0, "Q": 1, "S": 2})  

# Handle missing values (fill with median values)
x.fillna(x.median(), inplace=True)

# Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(x, y)

# Take user input for prediction
Name = input("Enter Name: ")  # Name is not used in the prediction, just for display
Pclass = int(input("Enter Pclass (1, 2, or 3): "))
Sex = input("Enter Sex (male/female): ")
Age = float(input("Enter Age: "))
SibSp = int(input("Enter SibSp (No. of siblings/spouses aboard): "))
Parch = int(input("Enter Parch (No. of parents/children aboard): "))
Fare = float(input("Enter Fare amount: "))
Embarked = input("Enter Embarked location (C/Q/S): ")

# Convert user input into numerical format
Sex = 1 if Sex.lower() == "male" else 0  
Embarked_dict = {"C": 0, "Q": 1, "S": 2}  
Embarked = Embarked_dict.get(Embarked.upper(), -1)  # Default to -1 if invalid input

# Prepare input array (reshape to match model's expected format)
sample_input = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

# Predict survival
prediction = model.predict(sample_input)

# Display result
print(f"Name: {Name}")
result_text = "YES (Survived)" if prediction[0] == 1 else "NO (Not Survived)"
print("Survival Prediction:", result_text)

# Visualization using Matplotlib
labels = ["Not Survived", "Survived"]
values = [1 - prediction[0], prediction[0]]  # 1 if survived, 0 if not

# Bar Chart Representation
plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=["red", "green"])
plt.xlabel("Survival Status")
plt.ylabel("Probability")
plt.title(f"Survival Prediction for {Name}")
plt.ylim(0, 1)
plt.show()

# Pie Chart Representation
plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct="%1.1f%%", colors=["red", "green"], startangle=90)
plt.title(f"Survival Prediction for {Name}")
plt.show()
