import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("train.csv/train.csv")

# Select features
features = data[
    [
        "GrLivArea",
        "BedroomAbvGr",
        "FullBath",
        "LotArea",
        "OverallQual",
        "YearBuilt"
    ]
]

# Target variable
target = data["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Predictions
predictions = model.predict(X_test)

print("\nPredicted Prices:")
print(predictions[:10])

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)