# Lagos House Price Predictor

**Live App:** [Click here to view the Dashboard](https://share.streamlit.io)

## Project Overview
This project is an End-to-End Machine Learning solution that predicts real estate prices in Lagos, Nigeria. 

Unlike standard housing datasets, Lagos real estate is highly volatile with extreme outliers (e.g., a ₦50M flat vs. a ₦25B commercial tower). This project handles those challenges through robust data cleaning, log-transformations, and a custom "Neighborhood Tier" system.

## Key Features
1.  **Interactive Analytics Dashboard:** - Visualizes the "Lekki Effect" (Price decay as distance from Lekki Toll Gate increases).
    - Compare property prices across 5 distinct Neighborhood Tiers.
    - Heatmaps showing the price matrix of Location vs. Bedroom count.
2.  **AI Valuation Engine:** - A user-friendly sidebar allows users to input property details (Location, Pool, Bedrooms, etc.).
    - Returns an instant market value estimate using a pre-trained Random Forest model.

## Tech Stack
- **Python:** Core programming language.
- **Scikit-Learn:** Machine Learning (Random Forest Regressor).
- **Pandas & NumPy:** Data manipulation and Log-Transformation.
- **Plotly:** Interactive visualizations.
- **Streamlit:** Frontend web application framework.

## Project Structure
This repository contains both the **Research** (Notebooks) and the **Production** (App) code:

- **`app.py`**: The main Streamlit application script.
- **`lagos_model.pkl`**: The pre-trained Random Forest model (saved via Joblib).
- **`data_sourcing.ipynb`**: Web scraping and initial data gathering.
- **`Exploratory_data_analysis.ipynb`**: In-depth EDA, outlier detection, and visualization.
- **`Feature_Extraction.ipynb`**: Creating the "Neighborhood_Tier" and "Distance_to_Lekki" features.
- **`model.ipynb`**: Model training, hyperparameter tuning, and evaluation (Random Forest vs XGBoost).

## How to Run Locally
If you want to run this dashboard on your own machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Davey117/LAGOS-HOUSE-PRICE.git](https://github.com/Davey117/LAGOS-HOUSE-PRICE.git)
   cd LAGOS-HOUSE-PRICE