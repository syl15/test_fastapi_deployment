import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import joblib 

# Load dataset 
data = pd.read_csv("data/iris.csv")
# print(data.head()) 

# print(data["species"].value_counts())

# Map species to numerical values 
data["species"] = data["species"].map({
    "setosa": 0, 
    "versicolor": 1, 
    "virginica": 2
})

# print(data["species"].value_counts())


# Define features and labels (X and y) 
X = data.drop("species", axis=1) 
y = data["species"]

# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model 
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on test dataset 
y_pred = model.predict(X_test)
# print(y_pred)

# Model evaluation 
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Save model as a pickle file 
joblib.dump(model, 'model/model.pkl')