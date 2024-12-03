import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Step 1: Load the data
data = pd.read_csv("data/house_data.csv")

# Step 2: Select features and target
X = data[["size", "bedrooms"]]
y = data["price"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
predictions = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Step 7: Save the trained model to a file
with open("model/house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)
