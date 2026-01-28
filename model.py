import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset
data = {
    'study_hours': [1,2,3,4,5,6,7,8,9,10],
    'marks': [35,40,45,50,55,65,70,75,85,90]
}

df = pd.DataFrame(data)

# Features and target
X = df[['study_hours']]
y = df['marks']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("student_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
