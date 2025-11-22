# -----------------------------
# 🚗 CAR PRICE PREDICTION MODEL
# -----------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# ====================================
# 1. Load Dataset
# ====================================

df = pd.read_csv("cars_ds_final_2021.csv")
print("\nColumns found:\n", df.columns)

# ------------------------------------
# 2. Clean price column
# ------------------------------------

if "Ex-Showroom_Price" in df.columns:
    price_col = "Ex-Showroom_Price"
else:
    raise Exception("Price column missing!")

print("\nUsing price column:", price_col)

# Convert price: Remove non-numeric characters
df[price_col] = (
    df[price_col]
    .astype(str)
    .str.replace("Rs.", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
    .astype(float)
)

# Remove rows with missing or 0 price
df = df[df[price_col] > 0]

# ====================================
# 3. Select useful features
# ====================================

feature_candidates = [
    "Make", "Model", "Variant",
    "Displacement", "Cylinders", "Valves_Per_Cylinder"
]

features = [col for col in feature_candidates if col in df.columns]
print("\nFinal features being used:", features)

X = df[features]
y = df[price_col]

# ====================================
# 4. Identify column types
# ====================================

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nCategorical Columns:", cat_cols)
print("\nNumeric Columns:", num_cols)

# ====================================
# 5. Preprocessing (FIX FOR NaN!)
# ====================================

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

# ====================================
# 6. Build Model
# ====================================

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

# ====================================
# 7. Train-Test Split
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================
# 8. FIT MODEL (Now NaN fixed)
# ====================================

print("\nTraining model...")
model.fit(X_train, y_train)

# ====================================
# 9. Evaluate
# ====================================

y_pred = model.predict(X_test)

print("\n---- Model Evaluation ----")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ====================================
# 10. Save Model
# ====================================

joblib.dump(model, "car_price_model.pkl")
print("\nModel saved as car_price_model.pkl")







