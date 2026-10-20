import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# 1. GENERATE PERFECTLY COHESIVE, REALISTIC DATA (500 Homes)
# =====================================================================
np.random.seed(42)
n_samples = 500

# Generate realistic Berlin apartment areas in square meters (40m² to 150m²)
m2_area = np.random.uniform(40, 150, n_samples)
sqft = np.round(m2_area * 10.7639).astype(int)

# Use your rule-based bedrooms: floor((sqft - 650) / 350) + 1
bedrooms = np.floor((sqft - 650) / 350) + 1
bedrooms = np.clip(bedrooms, 1, 5).astype(int)

# 40% New Construction (1), 60% Resale (0)
is_new_build = np.random.binomial(1, 0.4, n_samples)

# Apply logical Berlin pricing structures (New builds are significantly pricier)
base_m2_price = np.where(is_new_build == 1, 
                          np.random.normal(8300, 400, n_samples), # New builds
                          np.random.normal(5400, 300, n_samples)) # Resales
per_square_meter = np.round(base_m2_price).astype(int)

# Total Price = (Area in m²) * (Price per m²) + tiny natural market variance (+/- 2%)
market_noise = np.random.uniform(0.98, 1.02, n_samples)
price = np.round((sqft / 10.7639) * per_square_meter * market_noise).astype(int)

# Assemble your pristine dataframe
df = pd.DataFrame({
    'House_ID': np.arange(1001, 1001 + n_samples),
    'Square_Feet': sqft,
    'Bedrooms': bedrooms,
    'Neighborhood': np.where(is_new_build == 1, 'New Construction', 'Resale'),
    'per square meter': per_square_meter,
    'Price': price
})

sns.pairplot(df, hue="Neighborhood")
plt.show()

#df.to_csv('berlin_housing_data.csv', index=False)

# =====================================================================
# 2. RUN THE MACHINE LEARNING DATA PIPELINE
# =====================================================================
# Drop the unique identifiers
df.drop(columns=['House_ID'], inplace=True)

# CRITICAL: Drop 'per square meter' to eliminate Data Leakage
df.drop(columns=['per square meter'], inplace=True)

# Encode categorical values cleanly to 1 and 0
df['Neighborhood'] = df['Neighborhood'].map({'New Construction': 1, 'Resale': 0})

# Split features (X) and target label (y)
X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a robust Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("=== Model Evaluation on Pure Realistic Dataset ===")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: €{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue', label='Predictions')
# Draw a perfect diagonal reference line
perfect_line = [y_test.min(), y_test.max()]
plt.plot(perfect_line, perfect_line, color='red', linestyle='--', linewidth=2, label='Perfect Fit')

plt.title('Evaluation: Actual vs. Predicted Housing Prices')
plt.xlabel('Actual Market Price (€)')
plt.ylabel('Model Predicted Price (€)')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_plot.png', dpi=300)
plt.show()