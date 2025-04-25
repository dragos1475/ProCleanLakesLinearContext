import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from regression_models import RegressionModels
from model_evaluation import ModelEvaluator
from report_generator import ReportGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler
import glob

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
    
    # Load data
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    # Create standardized version of the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_standardized = pd.DataFrame(
        scaler_X.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    y_standardized = pd.Series(
        scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel(),
        index=y.index,
        name=y.name
    )
    
    # Split both raw and standardized data
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(
        X_standardized, y_standardized, test_size=test_size, random_state=random_state
    )
    
    # Initialize models with hyperparameters
    models_raw = RegressionModels(hyperparameters)
    models_std = RegressionModels(hyperparameters)
    
    # Initialize evaluator and report generator
    evaluator = ModelEvaluator()
    report_generator = ReportGenerator(output_dir='reports')
    
    # Dictionary to store all reports
    all_reports = {
        'raw': {},
        'standardized': {}
    }
    
    # Train and evaluate each model on raw data
    print("\nTraining and evaluating models on raw data...")
    for name, model in models_raw.get_models().items():
        print(f"\nTraining {name} on raw data...")
        
        # Train model
        model.fit(X_train_raw, y_train_raw)
        
        # Make predictions
        y_pred = model.predict(X_test_raw)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test_raw, y_pred)
        
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
        correlation_matrix = evaluator.calculate_correlation_matrix(X, y)
        
        # Perform PCA
        pca_results = evaluator.perform_pca(X)
        
        # Perform residual analysis
        residual_analysis = evaluator.analyze_residuals(y_test_raw, y_pred, X_test_raw)
        
        # Generate visualizations
        actual_vs_predicted = evaluator.plot_actual_vs_predicted(y_test_raw, y_pred, f"{name}_raw")
        residuals_plot = evaluator.plot_residuals(y_test_raw, y_pred, f"{name}_raw")
        error_distribution = evaluator.plot_error_distribution(y_test_raw, y_pred, f"{name}_raw")
        pca_variance_plot = evaluator.plot_pca_variance(pca_results, f"{name}_raw")
        correlation_heatmap = evaluator.plot_correlation_heatmap(correlation_matrix, f"{name}_raw")
        
        # Store results
        all_reports['raw'][name] = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'correlation_matrix': correlation_matrix,
            'pca_results': pca_results,
            'residual_analysis': residual_analysis,
            'plots': {
                'actual_vs_predicted': actual_vs_predicted,
                'residuals': residuals_plot,
                'error_distribution': error_distribution,
                'pca_variance': pca_variance_plot,
                'correlation_heatmap': correlation_heatmap
            }
        }
    
    # Train and evaluate each model on standardized data
    print("\nTraining and evaluating models on standardized data...")
    for name, model in models_std.get_models().items():
        print(f"\nTraining {name} on standardized data...")
        
        # Train model
        model.fit(X_train_std, y_train_std)
        
        # Make predictions
        y_pred_std = model.predict(X_test_std)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test_std, y_pred_std)
        
        # Add model coefficients and intercept to metrics
        if hasattr(model, 'coef_'):
            metrics['coefficients'] = dict(zip(X.columns, model.coef_))
            metrics['intercept'] = model.intercept_
        elif hasattr(model, 'estimator_') and hasattr(model.estimator_, 'coef_'):
            metrics['coefficients'] = dict(zip(X.columns, model.estimator_.coef_))
            metrics['intercept'] = model.estimator_.intercept_
        
        # Add scaling parameters
        metrics['scaling_params'] = {
            'feature_means': dict(zip(X.columns, scaler_X.mean_)),
            'feature_stds': dict(zip(X.columns, scaler_X.scale_)),
            'target_mean': float(scaler_y.mean_[0]),
            'target_std': float(scaler_y.scale_[0])
        }
        
        # Get feature importance
        feature_importance = evaluator.get_feature_importance(model, X.columns)
        
        # Calculate correlation matrix
        correlation_matrix = evaluator.calculate_correlation_matrix(X_standardized, y_standardized)
        
        # Perform PCA
        pca_results = evaluator.perform_pca(X_standardized)
        
        # Perform residual analysis
        residual_analysis = evaluator.analyze_residuals(y_test_std, y_pred_std, X_test_std)
        
        # Generate visualizations
        actual_vs_predicted = evaluator.plot_actual_vs_predicted(y_test_std, y_pred_std, f"{name}_standardized")
        residuals_plot = evaluator.plot_residuals(y_test_std, y_pred_std, f"{name}_standardized")
        error_distribution = evaluator.plot_error_distribution(y_test_std, y_pred_std, f"{name}_standardized")
        pca_variance_plot = evaluator.plot_pca_variance(pca_results, f"{name}_standardized")
        correlation_heatmap = evaluator.plot_correlation_heatmap(correlation_matrix, f"{name}_standardized")
        
        # Store results
        all_reports['standardized'][name] = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'correlation_matrix': correlation_matrix,
            'pca_results': pca_results,
            'residual_analysis': residual_analysis,
            'plots': {
                'actual_vs_predicted': actual_vs_predicted,
                'residuals': residuals_plot,
                'error_distribution': error_distribution,
                'pca_variance': pca_variance_plot,
                'correlation_heatmap': correlation_heatmap
            }
        }
    
    # Generate reports
    report_generator = ReportGenerator(output_dir='reports')
    
    # Generate general analysis report (correlation and PCA)
    general_report_path = report_generator.generate_general_html_report(
        correlation_matrix=all_reports['raw'][list(all_reports['raw'].keys())[0]]['correlation_matrix'],
        correlation_plot=all_reports['raw'][list(all_reports['raw'].keys())[0]]['plots']['correlation_heatmap'],
        pca_results=all_reports['raw'][list(all_reports['raw'].keys())[0]]['pca_results'],
        pca_plots={
            'Explained Variance': all_reports['raw'][list(all_reports['raw'].keys())[0]]['plots']['pca_variance']
        },
        target_name=target_variable
    )
    print(f"\nGeneral analysis report generated: {general_report_path}")
    
    # Generate model-specific reports
    for model_name, results in all_reports['raw'].items():
        raw_report_path = report_generator.generate_html_report(
            model_name=model_name,
            data_type='Raw',
            results={
                'metrics': results['metrics'],
                'model_equation': {
                    'coefficients': results['metrics']['coefficients'],
                    'intercept': results['metrics']['intercept']
                },
                'feature_importance': results['feature_importance'],
                'plots': {
                    'actual_vs_predicted': results['plots'].get('actual_vs_predicted', ''),
                    'residuals': results['plots'].get('residuals', ''),
                    'error_distribution': results['plots'].get('error_distribution', '')
                }
            },
            target_name=target_variable
        )
        print(f"\n{model_name} raw data report generated: {raw_report_path}")
    
    for model_name, results in all_reports['standardized'].items():
        standardized_report_path = report_generator.generate_html_report(
            model_name=model_name,
            data_type='Standardized',
            results={
                'metrics': results['metrics'],
                'model_equation': {
                    'coefficients': results['metrics']['coefficients'],
                    'intercept': results['metrics']['intercept']
                },
                'feature_importance': results['feature_importance'],
                'plots': {
                    'actual_vs_predicted': results['plots'].get('actual_vs_predicted', ''),
                    'residuals': results['plots'].get('residuals', ''),
                    'error_distribution': results['plots'].get('error_distribution', '')
                }
            },
            target_name=target_variable
        )
        print(f"\n{model_name} standardized data report generated: {standardized_report_path}")
    
    # Generate combined Word report
    word_report_path = report_generator.generate_combined_word_report(
        raw_results=all_reports['raw'],
        standardized_results=all_reports['standardized'],
        target_name=target_variable
    )
    print(f"\nCombined Word report generated: {word_report_path}")
    
    # Move all PNG files to reports directory before cleanup
    print("\nMoving plot files to reports directory...")
    for pattern in ['*_actual_vs_predicted.png', '*_residuals.png', '*_error_distribution.png', 
                   '*_pca_variance.png', '*_correlation_heatmap.png']:
        for file in glob.glob(pattern):
            try:
                if not os.path.exists('reports'):
                    os.makedirs('reports')
                os.rename(file, os.path.join('reports', file))
                print(f"Moved {file} to reports directory")
            except Exception as e:
                print(f"Error moving {file}: {e}")
    
    # Cleanup temporary files only after all reports are generated and files are moved
    report_generator.cleanup_temp_files()
    print("\nTemporary files cleaned up")

if __name__ == "__main__":
    main() 