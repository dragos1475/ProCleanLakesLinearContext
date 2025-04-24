import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        
    def handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        return data.fillna(data.mean())
    
    def get_correlation_matrix(self, data):
        """Calculate and visualize correlation matrix"""
        corr_matrix = data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
        return corr_matrix
    
    def scale_features(self, X):
        """Scale features using StandardScaler"""
        return self.scaler.fit_transform(X)
    
    def perform_pca(self, X, n_components=None):
        """Perform PCA analysis"""
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        # Plot explained variance ratio
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.savefig('pca_variance.png')
        plt.close()
        
        return X_pca
    
    def get_feature_importance(self):
        """Get feature importance from PCA"""
        if self.pca is None:
            raise ValueError("PCA has not been performed yet")
        return self.pca.components_ 