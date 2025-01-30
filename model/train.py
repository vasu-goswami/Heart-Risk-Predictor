import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Assuming your data is loaded into a DataFrame called df
df = pd.read_csv('heart.csv')  # Replace with your dataset

# Define features and target
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and check accuracy
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the trained model as a .pkl file
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'heart_model.pkl'")
