import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from regression_models import RegressionModels
from model_evaluation import ModelEvaluator
from report_generator import ReportGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

def get_model_hyperparameters():
    """Get hyperparameters for each model from user input"""
    hyperparameters = {
        'LinearRegression': {},
        'Ridge': {
            'alpha': 1.0
        },
        'Lasso': {
            'alpha': 1.0
        },
        'ElasticNet': {
            'alpha': 1.0,
            'l1_ratio': 0.5
        },
        'RANSAC': {
            'min_samples': 0.5,
            'max_trials': 100,
            'residual_threshold': None
        },
        'Huber': {
            'epsilon': 1.35,
            'max_iter': 100,
            'alpha': 0.0001
        },
        'TheilSen': {
            'max_subpopulation': 10000,
            'n_subsamples': None,
            'max_iter': 300
        }
    }
    
    print("\n=== Model Hyperparameters Configuration ===")
    print("You can configure hyperparameters for each model.")
    print("Press Enter to use default values.")
    
    for model, params in hyperparameters.items():
        print(f"\n{model} Hyperparameters:")
        for param, default in params.items():
            while True:
                try:
                    value = input(f"  {param} (default: {default}): ").strip()
                    if value == "":
                        break
                    if param in ['min_samples', 'l1_ratio']:
                        value = float(value)
                        if 0 <= value <= 1:
                            hyperparameters[model][param] = value
                            break
                        else:
                            print(f"  {param} must be between 0 and 1")
                    elif param in ['max_trials', 'max_iter', 'n_subsamples', 'max_subpopulation']:
                        value = int(value)
                        if value > 0:
                            hyperparameters[model][param] = value
                            break
                        else:
                            print(f"  {param} must be greater than 0")
                    elif param == 'residual_threshold':
                        if value.lower() == 'none':
                            hyperparameters[model][param] = None
                            break
                        value = float(value)
                        if value > 0:
                            hyperparameters[model][param] = value
                            break
                        else:
                            print(f"  {param} must be greater than 0 or 'None'")
                    else:
                        value = float(value)
                        if value > 0:
                            hyperparameters[model][param] = value
                            break
                        else:
                            print(f"  {param} must be greater than 0")
                except ValueError:
                    print(f"  Please enter a valid number for {param}")
    
    return hyperparameters

def get_user_input():
    """Get user input for dataset path, target variable, test size, and random state"""
    print("\n=== Regression Analysis Configuration ===")
    
    # Get dataset path
    while True:
        dataset_path = input("\nEnter the path to your dataset (CSV file): ").strip()
        if not dataset_path:
            print("Dataset path cannot be empty. Please try again.")
            continue
        if not os.path.exists(dataset_path):
            print(f"File not found: {dataset_path}. Please try again.")
            continue
        if not dataset_path.endswith('.csv'):
            print("Please provide a CSV file. Please try again.")
            continue
        break
    
    # Load dataset to get column names
    try:
        data = pd.read_csv(dataset_path)
        print("\nAvailable columns in the dataset:")
        for i, col in enumerate(data.columns):
            print(f"{i+1}. {col}")
    except Exception as e:
        print(f"Error reading the dataset: {str(e)}")
        return None, None, None, None, None
    
    # Get target variable
    while True:
        try:
            target_idx = int(input("\nEnter the number of the target variable: ")) - 1
            if 0 <= target_idx < len(data.columns):
                target_variable = data.columns[target_idx]
                break
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get test size
    while True:
        try:
            test_size = float(input("\nEnter the test size (between 0 and 1, default: 0.2): ") or "0.2")
            if 0 < test_size < 1:
                break
            else:
                print("Test size must be between 0 and 1.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get random state
    while True:
        try:
            random_state = int(input("\nEnter the random state (integer, default: 42): ") or "42")
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    # Get hyperparameters
    hyperparameters = get_model_hyperparameters()
    
    return dataset_path, target_variable, test_size, random_state, hyperparameters

def main():
    # Get user input
    dataset_path, target_variable, test_size, random_state, hyperparameters = get_user_input()
    if dataset_path is None:  # Check if user input was successful
        return
    
    # Load and preprocess data
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize models with hyperparameters
    models = RegressionModels(hyperparameters)
    
    # Initialize evaluator and report generator
    evaluator = ModelEvaluator()
    report_generator = ReportGenerator()
    
    # Dictionary to store all reports
    all_reports = {}
    
    # Train and evaluate each model
    for name, model in models.get_models().items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        
        # Add model coefficients and intercept to metrics
        if hasattr(model, 'coef_'):
            metrics['coefficients'] = dict(zip(X.columns, model.coef_))
            metrics['intercept'] = model.intercept_
        elif hasattr(model, 'estimator_') and hasattr(model.estimator_, 'coef_'):
            metrics['coefficients'] = dict(zip(X.columns, model.estimator_.coef_))
            metrics['intercept'] = model.estimator_.intercept_
        
        # Get feature importance
        feature_importance = evaluator.get_feature_importance(model, X.columns)
        
        # Calculate correlation matrix
        correlation_matrix = evaluator.calculate_correlation_matrix(X)
        
        # Perform PCA
        pca_results = evaluator.perform_pca(X)
        
        # Perform residual analysis
        residual_analysis = evaluator.analyze_residuals(y_test, y_pred, X_test)
        
        # Generate visualizations
        actual_vs_predicted = evaluator.plot_actual_vs_predicted(y_test, y_pred, name)
        residuals_plot = evaluator.plot_residuals(y_test, y_pred, name)
        error_distribution = evaluator.plot_error_distribution(y_test, y_pred, name)
        pca_variance_plot = evaluator.plot_pca_variance(pca_results, name)
        correlation_heatmap = evaluator.plot_correlation_heatmap(correlation_matrix, name)
        
        # Generate HTML report
        html_report = report_generator.generate_report(
            name, metrics, feature_importance, correlation_matrix, 
            pca_results, residual_analysis, actual_vs_predicted, 
            residuals_plot, error_distribution, pca_variance_plot, 
            correlation_heatmap
        )
        
        # Store report information
        all_reports[name] = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'correlation_matrix': correlation_matrix,
            'pca_results': pca_results,
            'residual_analysis': residual_analysis,
            'actual_vs_predicted': actual_vs_predicted,
            'residuals_plot': residuals_plot,
            'error_distribution': error_distribution,
            'pca_variance_plot': pca_variance_plot,
            'correlation_heatmap': correlation_heatmap
        }
        
        # Print summary
        report_generator.generate_summary(name, metrics, feature_importance, residual_analysis)
    
    # Generate combined Word report
    combined_report = report_generator.generate_combined_word_report(
        'All Models',
        {name: report['metrics'] for name, report in all_reports.items()},
        {name: report['feature_importance'] for name, report in all_reports.items()},
        all_reports[list(all_reports.keys())[0]]['correlation_matrix'],  # Use first model's correlation matrix
        all_reports[list(all_reports.keys())[0]]['pca_results'],  # Use first model's PCA results
        {name: report['residual_analysis'] for name, report in all_reports.items()},
        {name: report['actual_vs_predicted'] for name, report in all_reports.items()},
        {name: report['residuals_plot'] for name, report in all_reports.items()},
        {name: report['error_distribution'] for name, report in all_reports.items()},
        {name: report['pca_variance_plot'] for name, report in all_reports.items()},
        {name: report['correlation_heatmap'] for name, report in all_reports.items()}
    )
    
    print(f"\nCombined report saved to: {combined_report}")

if __name__ == "__main__":
    main() 