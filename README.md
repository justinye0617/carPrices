# Used Car Price Prediction Model

This project applies machine learning techniques to predict the price of used cars using a cleaned dataset sourced from Craigslist vehicle listings. The goal is to understand the most influential factors in used car pricing and build regression models to predict price with a high degree of accuracy.

## Dataset Overview

- **Source**: Kaggle Craigslist Cars and Trucks dataset
- **Rows**: 426,880 vehicle listings
- **Features**: 18 columns (numeric and categorical)

### Feature Summary

| Column          | Description                            |
|-----------------|----------------------------------------|
| id              | Unique identifier                      |
| region          | Geographic region                      |
| price           | Listed price of the vehicle (target)   |
| year            | Year of manufacture                    |
| manufacturer    | Manufacturer brand                     |
| model           | Model name                             |
| condition       | Condition of the vehicle               |
| cylinders       | Number of engine cylinders             |
| fuel            | Fuel type                              |
| odometer        | Mileage in miles                       |
| title_status    | Title status (e.g., clean, salvage)    |
| transmission    | Transmission type                      |
| VIN             | Vehicle Identification Number          |
| drive           | Drive type (e.g., FWD, RWD, 4WD)       |
| size            | Vehicle size classification            |
| type            | Vehicle category (e.g., sedan, SUV)    |
| paint_color     | Exterior color                         |
| state           | State where the listing was posted     |

## Data Cleaning and Preprocessing

- Removed listings with price below $500 or above $100,000 to eliminate outliers.
- Dropped columns with more than 25% missing values (`condition`, `cylinders`, `size`, `VIN`).
- Dropped high-cardinality categorical columns (e.g., `model`) and low-value identifiers (`id`, `VIN`).
- Engineered new feature `car_age` from `year` and dropped `year`.
- Missing values in numeric columns were filled using column-wise mean.
- Missing values in categorical columns were filled with `'missing'`.
- Categorical features were encoded using one-hot encoding.
- The target variable (`price`) was log-transformed to improve model learning.

## Modeling Approach

Two models were trained and evaluated:

1. **Linear Regression** — for interpretability and as a baseline.
2. **LightGBM Regressor** — to capture non-linear relationships and interactions.

### Evaluation Metrics

- **Root Mean Squared Error (RMSE)**: Measures average prediction error in dollars.
- **R^2 Score**: Measures the proportion of variance in the target explained by the model.

### Results

| Model             | RMSE ($)      | R^2 Score |
|-------------------|----------------|-----------|
| Linear Regression | 10,999.11      | 0.418     |
| LightGBM Regressor| **7,434.75**   | **0.734** |

LightGBM outperformed linear regression significantly in both RMSE and R^2, indicating that it captured complex, non-linear relationships between vehicle features and price.

## Visualizations

### Actual vs Predicted Prices (LightGBM)

A scatter plot of predicted vs actual prices demonstrated that the model closely approximated real-world price values:

![Actual vs Predicted Prices](https://github.com/justinye0617/Used-Car-Price-Prediction-Model/blob/main/actualVsPredicted.png)

The dense clustering around the red diagonal line confirms the model generalizes well to unseen data.

### Top 20 Feature Importances from LightGBM

A feature importance plot from the trained LightGBM model reveals the most influential predictors in the dataset:

![Feature Importance](https://github.com/justinye0617/Used-Car-Price-Prediction-Model/blob/main/featureImportance.png)

Key features included:
- `car_age`
- `odometer`
- `fuel_diesel`, `fuel_gas`
- `transmission_automatic`
- and various one-hot encoded region/state indicators

## Key Takeaways

- The most important features for predicting price include `car_age`, `odometer`, `fuel`, `drive`, and `transmission`.
- Linear regression captured only about 42% of the variance in price, while LightGBM explained about 73%, confirming the importance of non-linear modeling techniques in this problem space.
- Log-transforming the price and filtering outliers were critical steps in improving model performance.
- Feature importance visualization provides valuable insight into which attributes most influence used car valuations.

## Next Steps

- Explore NLP techniques on listing `description` fields (if available).
- Incorporate interaction terms or polynomial features for linear models.
- Investigate ensemble methods or model stacking to push performance further.
