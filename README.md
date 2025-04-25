# Linear Regression Analysis Tool

A comprehensive tool for performing linear regression analysis with multiple models and detailed evaluation metrics, supporting both raw and standardized data analysis.

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
- Parallel analysis on both raw and standardized data

### Data Preprocessing
- Missing value handling
- Optional feature standardization using StandardScaler
- Principal Component Analysis (PCA)
- Correlation analysis with target variable inclusion
- Feature importance analysis

### Model Evaluation
- Comprehensive metrics for both raw and standardized data:
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
  1. Combined Word report (all models) - `combined_regression_analysis.docx`
  2. Individual HTML reports for each model - `{model_name}_{data_type}_analysis.html`
  3. General HTML report for correlation and PCA - `general_analysis.html`

#### Word Report Structure
1. Feature Correlation Analysis (raw data only):
   - Correlation matrix with target variable
   - Correlation heatmap
2. Principal Component Analysis (raw data only):
   - Explained variance ratio
   - Component loadings
   - PCA variance plot
3. Model Results for Raw Data:
   - Model equations with raw coefficients
   - Performance metrics
   - Feature importance
   - Residual analysis
   - Model-specific visualizations
4. Model Results for Standardized Data:
   - Model equations with standardized coefficients
   - Standardization parameters (means and standard deviations)
   - Performance metrics
   - Feature importance
   - Residual analysis
   - Model-specific visualizations

#### HTML Reports Features
1. General Analysis Report (`general_analysis.html`):
   - Interactive correlation matrix and heatmap
   - PCA analysis with explained variance plots
   - Component loadings visualization
   
2. Model-Specific Reports (`{model_name}_{data_type}_analysis.html`):
   - Detailed model metrics with interpretations
   - Interactive feature importance visualization
   - Model equation with proper mathematical formatting
   - Performance visualizations:
     - Actual vs Predicted Values
     - Residuals Plot
     - Error Distribution
   - Standardization parameters (for standardized models)
   - Color-coded metric interpretations

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
   - `combined_regression_analysis.docx` - comprehensive Word report
   - `general_analysis.html` - correlation and PCA analysis
   - Individual model reports in HTML format
   - Saved models (`.joblib` files)

## Project Structure

```
.
├── main.py                 # Main script with parallel raw/standardized analysis
├── data_preprocessing.py   # Data preprocessing with optional standardization
├── regression_models.py    # Regression model implementations
├── model_evaluation.py     # Model evaluation metrics and plots
├── report_generator.py     # Report generation in Word and HTML formats
├── requirements.txt        # Required Python packages
├── README.md              # This file
├── reports/               # Generated reports and visualizations
│   ├── plots/            # Generated plot images
│   ├── *.html           # HTML reports
│   └── *.docx           # Word reports
├── models/               # Saved models
└── templates/            # Report templates
```

## Best Practices

1. Data Preparation:
   - Ensure your dataset is clean and properly formatted
   - Handle missing values appropriately
   - Consider both raw and standardized analysis for comprehensive insights

2. Model Selection:
   - Start with simple models (Linear Regression)
   - Use regularization (Ridge/Lasso) for high-dimensional data
   - Consider robust models (Huber/Theil-Sen) for outliers
   - Compare model performance on both raw and standardized data

3. Evaluation:
   - Review metrics for both raw and standardized data
   - Check residual analysis for model assumptions
   - Consider feature importance in both contexts
   - Use correlation analysis to understand relationships with target variable

4. Report Interpretation:
   - Compare raw and standardized model equations
   - Use standardization parameters for converting between scales
   - Review PCA and correlation analysis from raw data
   - Check residual plots for model assumptions
   - Use both Word and HTML reports for different perspectives:
     - Word report for comprehensive documentation
     - HTML reports for interactive exploration and sharing

## Contributing

Feel free to submit issues and enhancement requests! 