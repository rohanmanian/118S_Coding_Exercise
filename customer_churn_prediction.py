import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Generate sample customer data
data = {
    'age': [39, 33, 41, 50, 32, 32, 50, 42, 30, 40, 30, 30, 37, 15, 17],
    'monthly_usage_hours': [21, 14, 34, 16, 8, 51, 26, 31, 8, 21, 31, 12, 35, 20, 25],
    'purchase_amount': [154, 338, 198, 120, 261, 108, 215, 53, 100, 214, 255, 212, 191, 177, 89],
    'customer_service_calls': [2, 3, 6, 4, 0, 4, 3, 2, 5, 6, 5, 2, 3, 4, 5],
    'region': ['South', 'North', 'East', 'West', 'West', 'South', 'East', 'North', 'West', 'East', 'East', 'South', 'West', 'West', 'North'],
    'churn': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
}
#Gemini randomized data set
df = pd.DataFrame(data)
# Features and target
X = df[['age', 'monthly_usage_hours', 'purchase_amount', 'customer_service_calls',
'region']]
y = df['churn']
# Preprocessing: Scale numerical features and one-hot encode categorical features
preprocessor = ColumnTransformer(
transformers=[
('num', StandardScaler(), ['age', 'monthly_usage_hours', 'purchase_amount',
'customer_service_calls']),
('cat', OneHotEncoder(sparse_output=False), ['region'])
])
# Create pipeline with preprocessing and model
model = Pipeline(steps=[
('preprocessor', preprocessor),
('classifier', LogisticRegression(random_state=42))
])
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Train model
model.fit(X_train, y_train)
# Predict churn probability for a new customer
new_customer = pd.DataFrame({
'age': [35],
'monthly_usage_hours': [20],
'purchase_amount': [150],
'customer_service_calls': [5],
'region': ['West']
})
churn_probability = model.predict_proba(new_customer)[0][1] # Probability of churn
(class 1)
# Classify based on threshold (0.5)
threshold = 0.5
churn_prediction = 1 if churn_probability > threshold else 0
print(f"Churn Probability for new customer: {churn_probability:.2f}")
print(f"Churn Prediction (1 = churn, 0 = no churn): {churn_prediction}")
# Display model coefficients
feature_names = (model.named_steps['preprocessor']
.named_transformers_['cat']
.get_feature_names_out(['region'])).tolist() + ['age',
'monthly_usage_hours', 'purchase_amount', 'customer_service_calls']
coefficients = model.named_steps['classifier'].coef_[0]
print("\nModel Coefficients:")
for feature, coef in zip(feature_names, coefficients):
print(f"{feature}: {coef:.2f}")
