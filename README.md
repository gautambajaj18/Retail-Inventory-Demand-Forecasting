# Retail Store Inventory: EDA and Demand Prediction

Exploratory data analysis and demand prediction on a retail inventory dataset (73,100 rows, 15 columns) from Kaggle.

**Dataset:** [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)

## Problem Statement
How much of each product will a store sell, and how accurate can a model be at predicting it? The goal is to understand what drives sales and to predict `Units_Sold` using machine learning.

## What I Did
1. **Data cleaning:** checked missing values and duplicates (none found), cleaned column names, converted Date and extracted Month and Year.
2. **EDA:** sales by category, region, month, store and product. Effect of holidays, discounts, weather and seasonality. Stock-out check, forecast error, correlation, distributions and outliers.
3. **Modeling:** built a Linear Regression baseline and a Random Forest model with a time-based 80/20 train-test split, and compared them with the forecast that comes with the dataset.

## Key Insights
- Sales are spread almost equally across categories (gap about 3%) and regions (gap about 1.6%).
- July had the highest sales and February the lowest.
- Holiday/Promotion, discount, weather and season show very little effect on average units sold.
- No stock-out was found (Inventory Level was never below Units Sold).
- The forecast included in the dataset has an average error of 8.34 units, about 6.1% of average sales.
- Units_Sold has almost no correlation with Price (0.001) or Discount (0.003), but is almost perfectly correlated with Demand_Forecast (0.997).

## Model Results (test set)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Dataset Forecast | 8.37 | 10.05 | 0.99 |
| Linear Regression | 68.66 | 87.67 | 0.34 |
| Random Forest | 69.61 | 89.54 | 0.32 |

My models did not beat the forecast provided in the dataset. They did not use `Demand_Forecast` as a feature, and the remaining columns have almost no relationship with `Units_Sold`, so the models could not learn much. The data looks synthetic. In feature importance, `Inventory_Level` ranked first, followed by Units_Ordered, Price and Competitor_Pricing.

## Tools Used
Python, pandas, matplotlib, seaborn, scikit-learn, Google Colab

## Files
- `.ipynb` notebook: full EDA and modeling with charts and outputs
- `retail_store_inventory.csv`: original dataset
- `cleaned_retail_data.csv`: dataset after cleaning and feature engineering

## How to Run
1. Open the notebook in Google Colab.
2. Upload `retail_store_inventory.csv` to the Colab session.
3. Run all cells.

## Next Steps
- Add lag features (sales of the previous 7 days)
- Try XGBoost
- Build separate models per store or product
- Power BI dashboard and SQL analysis
