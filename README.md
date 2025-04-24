# Linear Regression Analysis Tool

A comprehensive tool for performing linear regression analysis with multiple models and detailed evaluation metrics.

## Features

### Regression Models
- Multiple regression models with configurable parameters:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Elastic Net Regression
  - RANSAC Regression
  - Huber Regression
  - Theil-Sen Regression

### Data Preprocessing
- Missing value handling
- Feature scaling
- Principal Component Analysis (PCA)
- Correlation analysis
- Feature importance analysis

### Model Evaluation
- Comprehensive metrics:
  - R-squared
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Explained Variance
  - Adjusted R-squared
- Residual analysis:
  - Normality tests
  - Heteroscedasticity tests
  - Autocorrelation tests

### Report Generation
- Multiple report formats:
  - HTML reports (individual models)
  - PDF reports (individual models)
  - Combined Word report (all models) - `combined_regression_analysis.docx`
- Report contents:
  - Model equations with coefficients and intercepts
  - Performance metrics for each model
  - Feature importance analysis
  - Correlation matrix (both as table and heatmap)
  - PCA analysis:
    - Explained variance ratio table
    - Cumulative variance
    - Component loadings table
    - PCA variance plot
  - Residual analysis:
    - Normality tests
    - Heteroscedasticity tests
    - Autocorrelation tests
  - Visualizations:
    - Actual vs. Predicted values for each model
    - Residual plots for each model
    - Error distribution for each model
    - PCA variance plot
    - Correlation heatmap

### Model Persistence
- Save trained models in `.joblib` format
- Load saved models for predictions on new data
- Automatic cleanup of temporary files after report generation

## Usage

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the script:
```bash
python main.py
```

3. Follow the prompts to:
   - Select your dataset
   - Choose the target variable
   - Configure test size and random state
   - Set hyperparameters for each model

4. Review the generated reports in the `reports` directory:
   - Individual model reports (HTML and PDF)
   - Combined report (DOCX) - contains all models' analysis in a single document
   - Saved models (`.joblib` files)

## Project Structure

```
.
├── main.py                 # Main script
├── data_preprocessing.py   # Data preprocessing functions
├── regression_models.py    # Regression model implementations
├── model_evaluation.py     # Model evaluation metrics and plots
├── report_generator.py     # Report generation utilities
├── requirements.txt        # Required Python packages
├── README.md              # This file
├── reports/               # Generated reports
├── models/                # Saved models
└── templates/             # Report templates
```

## Best Practices

1. Data Preparation:
   - Ensure your dataset is clean and properly formatted
   - Handle missing values appropriately
   - Consider feature scaling for better model performance

2. Model Selection:
   - Start with simple models (Linear Regression)
   - Use regularization (Ridge/Lasso) for high-dimensional data
   - Consider robust models (Huber/Theil-Sen) for outliers

3. Evaluation:
   - Review all metrics, not just R-squared
   - Check residual analysis for model assumptions
   - Consider feature importance and correlations

4. Report Interpretation:
   - Focus on model equations and coefficients
   - Review PCA analysis for dimensionality reduction
   - Check residual plots for model assumptions

## Contributing

Feel free to submit issues and enhancement requests! 