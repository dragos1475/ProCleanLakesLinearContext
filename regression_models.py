from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, HuberRegressor, TheilSenRegressor
from sklearn.model_selection import train_test_split
import numpy as np

class RegressionModels:
    def __init__(self, hyperparameters=None):
        """Initialize regression models with optional hyperparameters"""
        self.hyperparameters = hyperparameters or {}
        
        # Initialize models with default or specified hyperparameters
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(**self.hyperparameters.get('Ridge', {'alpha': 1.0})),
            'Lasso': Lasso(**self.hyperparameters.get('Lasso', {'alpha': 1.0})),
            'Elastic Net': ElasticNet(**self.hyperparameters.get('ElasticNet', {'alpha': 1.0, 'l1_ratio': 0.5})),
            'RANSAC': RANSACRegressor(**self.hyperparameters.get('RANSAC', {
                'min_samples': 0.5,
                'max_trials': 100,
                'residual_threshold': None
            })),
            'Huber': HuberRegressor(**self.hyperparameters.get('Huber', {
                'epsilon': 1.35,
                'max_iter': 100,
                'alpha': 0.0001
            })),
            'Theil-Sen': TheilSenRegressor(**self.hyperparameters.get('TheilSen', {
                'max_subpopulation': 10000,
                'n_subsamples': None,
                'max_iter': 300
            }))
        }
        
        self.coefficients = {}
        self.best_model = None
    
    def get_models(self):
        """Return the dictionary of initialized models"""
        return self.models
    
    def get_model(self, name):
        """Get a specific model by name"""
        return self.models.get(name)
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train all regression models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'model': model,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred
            }
            # Handle RANSAC differently since it's a meta-estimator
            if name == 'RANSAC':
                self.coefficients[name] = model.estimator_.coef_
            else:
                self.coefficients[name] = model.coef_
        
        return results
    
    def get_coefficients(self, model_name):
        """Get coefficients for a specific model"""
        return self.coefficients.get(model_name)
    
    def get_feature_importance(self, model_name):
        """Get feature importance based on coefficients"""
        coef = self.get_coefficients(model_name)
        if coef is not None:
            return np.abs(coef)
        return None
    
    def set_best_model(self, name):
        """Set the best model based on evaluation"""
        self.best_model = self.models.get(name)
    
    def predict_new_data(self, model_name, new_data):
        """Make predictions using a specific model"""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        return model.predict(new_data) 