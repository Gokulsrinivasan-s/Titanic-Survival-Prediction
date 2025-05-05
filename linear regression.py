import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# Load the dataset
file_path = r"F:\project.py\data set predection\titanic survival prediction\titonic survival data_set.csv"
titanicdata_set = pd.read_csv(file_path)

# Drop unnecessary text columns
titanicdata_set = titanicdata_set.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")

# Convert categorical columns to numerical
titanicdata_set["Sex"] = titanicdata_set["Sex"].map({"male": 1, "female": 0})
titanicdata_set["Embarked"] = titanicdata_set["Embarked"].map({"C": 0, "Q": 1, "S": 2})
titanicdata_set["Embarked"] = titanicdata_set["Embarked"].fillna(2)  # Default to 'S'

# Fill missing values with column means
titanicdata_set = titanicdata_set.fillna(titanicdata_set.mean(numeric_only=True))

# Define features and target
X = titanicdata_set.drop(columns=["Survived"])
y = titanicdata_set["Survived"]

# Split dataset into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### **1️⃣ Train Decision Tree Classifier**
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)

# Predict and evaluate Decision Tree
y_pred_dt = dt_model.predict(x_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")

### **2️⃣ Train Linear Regression Model**
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# Predict and evaluate Linear Regression
y_pred_lr = lr_model.predict(x_test)

# Convert predictions to binary (0 or 1) using a 0.5 threshold
y_pred_lr_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_lr]

# Evaluate regression model
r2 = r2_score(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr)
lr_accuracy = accuracy_score(y_test, y_pred_lr_binary)

print(f"Linear Regression R² Score: {r2:.2f}")
print(f"Linear Regression Mean Squared Error: {mse:.4f}")
print(f"Linear Regression Accuracy (Converted to Binary): {lr_accuracy:.2f}")
