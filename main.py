import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
dates = pd.date_range(start='2022-01-01', periods=365)
sales = 100 + 50 * np.sin(2 * np.pi * np.arange(365) / 365) + 20 * np.random.randn(365) # Seasonal pattern + noise
df = pd.DataFrame({'Date': dates, 'Sales': sales})
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
# --- 2. Data Cleaning and Feature Engineering ---
# In a real-world scenario, this section would involve handling missing values, outliers, etc.
# For this synthetic data, no cleaning is needed.
# --- 3. Analysis and Modeling ---
# Group data by month to analyze seasonal patterns
monthly_sales = df.groupby('Month')['Sales'].mean()
# Fit a simple linear regression model (for demonstration; more sophisticated models are possible)
X = np.arange(1,13).reshape(-1,1) # Month as independent variable
y = monthly_sales.values
X = sm.add_constant(X) # Add intercept
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, label='Actual Monthly Sales')
plt.plot(monthly_sales.index, predictions, label='Predicted Monthly Sales', linestyle='--')
plt.title('Actual vs. Predicted Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'sales_prediction.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 5. Model Evaluation (Illustrative) ---
# In a real-world scenario, a thorough model evaluation would be performed using appropriate metrics.
print("\nModel Summary:")
print(model.summary())
# --- 6. Inventory Optimization (Conceptual) ---
# Based on the predictions, inventory levels can be adjusted to meet anticipated demand, reducing excess inventory.
# This would involve incorporating cost parameters and setting inventory targets.  This step is omitted for brevity.