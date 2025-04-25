import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_percentage_error,
    mean_absolute_error,
    explained_variance_score,
    max_error
)
import joblib
from sklearn.model_selection import train_test_split
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import acf
from statsmodels.stats.stattools import durbin_watson
import seaborn as sns
import pandas as pd
import os

# Set style for all plots
plt.style.use('seaborn')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

class ModelEvaluator:
    def __init__(self, test_size=0.2, random_state=42):
        self.metrics = {}
        self.test_size = test_size
        self.random_state = random_state
        self.residual_analysis = {}
        
        # Create directory structure for reports
        self.reports_dir = 'reports'
        self.plots_dir = os.path.join(self.reports_dir, 'plots')
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def get_plot_path(self, model_name, plot_type):
        """Get the full path for a plot file"""
        return os.path.join(self.plots_dir, f'{model_name}_{plot_type}.png')
    
    def split_data(self, X, y):
        """Split data into training and test sets"""
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive regression metrics"""
        metrics = {
            'R2': round(r2_score(y_true, y_pred), 2),
            'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
            'MAPE': round(mean_absolute_percentage_error(y_true, y_pred), 2),
            'MAE': round(mean_absolute_error(y_true, y_pred), 2),
            'Explained Variance': round(explained_variance_score(y_true, y_pred), 2),
            'Max Error': round(max_error(y_true, y_pred), 2),
            'Mean Error': round(np.mean(y_true - y_pred), 2),
            'Std Error': round(np.std(y_true - y_pred), 2)
        }
        self.metrics = metrics
        return metrics
    
    def analyze_residuals(self, y_true, y_pred, X):
        """Perform detailed residual analysis"""
        residuals = y_true - y_pred
        
        # Normality tests
        shapiro_test = stats.shapiro(residuals)
        ks_test = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
        
        # Heteroscedasticity test
        try:
            _, p_value, _, _ = het_breuschpagan(residuals, sm.add_constant(X))
        except:
            p_value = None
        
        # Autocorrelation test
        dw_stat = durbin_watson(residuals)
        
        # Store results
        self.residual_analysis = {
            'normality': {
                'shapiro_wilk': {
                    'statistic': round(shapiro_test.statistic, 2),
                    'p_value': round(shapiro_test.pvalue, 4)
                },
                'kolmogorov_smirnov': {
                    'statistic': round(ks_test.statistic, 2),
                    'p_value': round(ks_test.pvalue, 4)
                }
            },
            'heteroscedasticity': {
                'breusch_pagan_p_value': round(p_value, 4) if p_value is not None else None
            },
            'autocorrelation': {
                'durbin_watson': round(dw_stat, 2)
            }
        }
        
        return self.residual_analysis
    
    def plot_residual_analysis(self, y_true, y_pred, X, model_name):
        """Create comprehensive residual analysis plots"""
        residuals = y_true - y_pred
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Residuals vs Fitted
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')
        ax1.grid(True)
        
        # 2. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot')
        ax2.grid(True)
        
        # 3. Scale-Location Plot
        ax3.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('√|Standardized Residuals|')
        ax3.set_title('Scale-Location Plot')
        ax3.grid(True)
        
        # 4. Residuals vs Leverage
        leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
        ax4.scatter(leverage, residuals, alpha=0.5)
        ax4.set_xlabel('Leverage')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs Leverage')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.get_plot_path(model_name, 'residual_analysis'))
        plt.close()
        
        # Additional plots
        plt.figure(figsize=(10, 6))
        sm.graphics.tsa.plot_acf(residuals, lags=20, ax=plt.gca())
        plt.title('Autocorrelation Plot')
        plt.grid(True)
        plt.savefig(self.get_plot_path(model_name, 'autocorrelation'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, density=True)
        x = np.linspace(min(residuals), max(residuals), 100)
        plt.plot(x, stats.norm.pdf(x, np.mean(residuals), np.std(residuals)), 'r-')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Histogram of Residuals')
        plt.grid(True)
        plt.savefig(self.get_plot_path(model_name, 'residual_histogram'))
        plt.close()
    
    def plot_actual_vs_predicted(self, y_true, y_pred, model_name):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        
        output_path = self.get_plot_path(model_name, 'actual_vs_predicted')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def plot_residuals(self, y_true, y_pred, model_name):
        """Plot residuals"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        
        output_path = self.get_plot_path(model_name, 'residuals')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def plot_error_distribution(self, y_true, y_pred, model_name):
        """Plot error distribution"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.ylabel('Count')
        plt.title('Distribution of Residuals')
        
        output_path = self.get_plot_path(model_name, 'error_distribution')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def save_model(self, model, filename):
        """Save trained model to file"""
        joblib.dump(model, filename)
    
    def load_model(self, filename):
        """Load trained model from file"""
        return joblib.load(filename)
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance based on model coefficients"""
        if hasattr(model, 'coef_'):
            coef = model.coef_
        elif hasattr(model, 'estimator_') and hasattr(model.estimator_, 'coef_'):
            coef = model.estimator_.coef_
        else:
            return {name: 0.0 for name in feature_names}
        
        # Convert coefficients to absolute values for importance
        importance = np.abs(coef)
        # Normalize importance to sum to 1
        importance = importance / np.sum(importance)
        
        return dict(zip(feature_names, importance))
    
    def calculate_correlation_matrix(self, X, y=None):
        """Calculate correlation matrix for features and optionally include target variable"""
        if y is None:
            return X.corr()
        
        # Create a DataFrame with both features and target
        data = pd.concat([X, pd.Series(y, name=y.name, index=X.index)], axis=1)
        return data.corr()
    
    def perform_pca(self, X):
        """Perform PCA on the features"""
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(X)
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_
        }
    
    def plot_pca_variance(self, pca_results, model_name):
        """Plot PCA explained variance"""
        explained_variance = pca_results['explained_variance_ratio']
        cumulative_variance = np.cumsum(explained_variance)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), 
                explained_variance, 
                alpha=0.7, 
                label='Individual explained variance')
        
        plt.plot(range(1, len(cumulative_variance) + 1), 
                cumulative_variance, 
                'r-', 
                label='Cumulative explained variance')
        
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.grid(True)
        
        output_path = self.get_plot_path(model_name, 'pca_variance')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def plot_correlation_heatmap(self, correlation_matrix, model_name):
        """Plot correlation heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True)
        plt.title('Feature Correlation Heatmap')
        
        output_path = self.get_plot_path(model_name, 'correlation_heatmap')
        plt.savefig(output_path)
        plt.close()
        
        return output_path 