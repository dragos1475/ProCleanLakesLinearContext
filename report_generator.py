from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from docx import Document
from docx.shared import Inches
import io
from jinja2 import Template
import glob

class ReportGenerator:
    def __init__(self, filename="regression_report.pdf", output_dir='reports'):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=letter)
        self.story = []
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        )
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=20
        )
        self.output_dir = output_dir
        self.env = Environment(loader=FileSystemLoader('templates'))
        os.makedirs(output_dir, exist_ok=True)
    
    def add_title(self, text):
        self.story.append(Paragraph(text, self.title_style))
        self.story.append(Spacer(1, 12))
    
    def add_subtitle(self, text):
        self.story.append(Paragraph(text, self.subtitle_style))
        self.story.append(Spacer(1, 12))
    
    def add_text(self, text):
        self.story.append(Paragraph(text, self.styles["Normal"]))
        self.story.append(Spacer(1, 12))
    
    def add_correlation_matrix(self, corr_matrix):
        self.add_subtitle("Correlation Matrix")
        
        # Convert correlation matrix to DataFrame for better formatting
        df = pd.DataFrame(corr_matrix).round(2)
        data = [df.columns.tolist()] + df.values.tolist()
        
        # Create table
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(t)
        self.story.append(Spacer(1, 12))
    
    def add_pca_analysis(self, pca, feature_names):
        self.add_subtitle("PCA Analysis")
        
        # Explained variance ratio
        self.add_text("Explained Variance Ratio:")
        data = [['Component', 'Explained Variance', 'Cumulative Variance']]
        cumulative = 0
        for i, var in enumerate(pca.explained_variance_ratio_):
            cumulative += var
            data.append([f'PC{i+1}', f'{var:.2f}', f'{cumulative:.2f}'])
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(t)
        self.story.append(Spacer(1, 12))
        
        # Component loadings
        self.add_text("Component Loadings:")
        loadings = pd.DataFrame(
            pca.components_,
            columns=feature_names,
            index=[f'PC{i+1}' for i in range(len(pca.components_))]
        ).round(2)
        data = [[''] + loadings.columns.tolist()] + \
               [[index] + row.tolist() for index, row in loadings.iterrows()]
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(t)
        self.story.append(Spacer(1, 12))
    
    def add_model_equation(self, model_name, coefficients, intercept, feature_names):
        self.add_subtitle(f"{model_name} Model Equation")
        
        equation = f"y = {intercept:.2f}"
        for i, coef in enumerate(coefficients):
            equation += f" + {coef:.2f} * {feature_names[i]}"
        
        self.add_text(equation)
        self.story.append(Spacer(1, 12))
    
    def add_model_metrics(self, model_name, metrics):
        self.add_subtitle(f"{model_name} Performance Metrics")
        
        data = [['Metric', 'Value']]
        for metric, value in metrics.items():
            data.append([metric, f'{value:.2f}'])
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(t)
        self.story.append(Spacer(1, 12))
    
    def add_plot(self, plot_func, *args, **kwargs):
        """Add a plot to the report"""
        # Create a BytesIO object to save the plot
        buf = BytesIO()
        plot_func(*args, **kwargs)
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Add the image to the report
        img = Image(buf, width=6*inch, height=4*inch)
        self.story.append(img)
        self.story.append(Spacer(1, 12))
    
    def generate_report(self, model_name, metrics, feature_importance, correlation_matrix, 
                       pca_results, residual_analysis, actual_vs_predicted, residuals_plot, 
                       error_distribution, pca_variance_plot, correlation_heatmap):
        """Generate HTML report with all analysis results"""
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Create HTML content using Jinja2 template
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ model_name }} Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #2c3e50; }
                .section { margin-bottom: 30px; }
                .metric { margin: 10px 0; }
                .plot { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .good { color: green; }
                .warning { color: orange; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>{{ model_name }} Analysis Report</h1>
            
            <div class="section">
                <h2>Model Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
                    <tr>
                        <td>R² Score</td>
                        <td>{{ metrics.R2 }}</td>
                        <td>{% if metrics.R2 >= 0.7 %}<span class="good">Good fit</span>
                            {% elif metrics.R2 >= 0.5 %}<span class="warning">Moderate fit</span>
                            {% else %}<span class="error">Poor fit</span>{% endif %}</td>
                    </tr>
                    <tr>
                        <td>RMSE</td>
                        <td>{{ metrics.RMSE }}</td>
                        <td>{% if metrics.RMSE <= 0.1 %}<span class="good">Low error</span>
                            {% elif metrics.RMSE <= 0.3 %}<span class="warning">Moderate error</span>
                            {% else %}<span class="error">High error</span>{% endif %}</td>
                    </tr>
                    <tr>
                        <td>MAE</td>
                        <td>{{ metrics.MAE }}</td>
                        <td>{% if metrics.MAE <= 0.1 %}<span class="good">Low error</span>
                            {% elif metrics.MAE <= 0.3 %}<span class="warning">Moderate error</span>
                            {% else %}<span class="error">High error</span>{% endif %}</td>
                    </tr>
                    <tr>
                        <td>Explained Variance</td>
                        <td>{{ metrics['Explained Variance'] }}</td>
                        <td>{% if metrics['Explained Variance'] >= 0.7 %}<span class="good">Good explanation</span>
                            {% elif metrics['Explained Variance'] >= 0.5 %}<span class="warning">Moderate explanation</span>
                            {% else %}<span class="error">Poor explanation</span>{% endif %}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Feature Importance</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
                    {% for feature, importance in feature_importance.items() %}
                    <tr>
                        <td>{{ feature }}</td>
                        <td>{{ "%.4f"|format(importance) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>Residual Analysis</h2>
                <table>
                    <tr>
                        <th>Test</th>
                        <th>Statistic</th>
                        <th>p-value</th>
                        <th>Interpretation</th>
                    </tr>
                    <tr>
                        <td>Shapiro-Wilk (Normality)</td>
                        <td>{{ residual_analysis.normality.shapiro_wilk.statistic }}</td>
                        <td>{{ residual_analysis.normality.shapiro_wilk.p_value }}</td>
                        <td>{% if residual_analysis.normality.shapiro_wilk.p_value > 0.05 %}
                            <span class="good">Normal</span>
                            {% else %}<span class="error">Not normal</span>{% endif %}</td>
                    </tr>
                    <tr>
                        <td>Breusch-Pagan (Heteroscedasticity)</td>
                        <td>-</td>
                        <td>{{ residual_analysis.heteroscedasticity.breusch_pagan_p_value }}</td>
                        <td>{% if residual_analysis.heteroscedasticity.breusch_pagan_p_value > 0.05 %}
                            <span class="good">Homoscedastic</span>
                            {% else %}<span class="error">Heteroscedastic</span>{% endif %}</td>
                    </tr>
                    <tr>
                        <td>Durbin-Watson (Autocorrelation)</td>
                        <td>{{ residual_analysis.autocorrelation.durbin_watson }}</td>
                        <td>-</td>
                        <td>{% if 1.5 <= residual_analysis.autocorrelation.durbin_watson <= 2.5 %}
                            <span class="good">No autocorrelation</span>
                            {% else %}<span class="error">Autocorrelation present</span>{% endif %}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="plot">
                    <h3>Actual vs Predicted Values</h3>
                    <img src="{{ actual_vs_predicted }}" alt="Actual vs Predicted Values">
                </div>
                <div class="plot">
                    <h3>Residuals Plot</h3>
                    <img src="{{ residuals_plot }}" alt="Residuals Plot">
                </div>
                <div class="plot">
                    <h3>Error Distribution</h3>
                    <img src="{{ error_distribution }}" alt="Error Distribution">
                </div>
                <div class="plot">
                    <h3>PCA Explained Variance</h3>
                    <img src="{{ pca_variance_plot }}" alt="PCA Explained Variance">
                </div>
                <div class="plot">
                    <h3>Feature Correlation Heatmap</h3>
                    <img src="{{ correlation_heatmap }}" alt="Feature Correlation Heatmap">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Render template with data
        html_content = Template(template).render(
            model_name=model_name,
            metrics=metrics,
            feature_importance=feature_importance,
            residual_analysis=residual_analysis,
            actual_vs_predicted=actual_vs_predicted,
            residuals_plot=residuals_plot,
            error_distribution=error_distribution,
            pca_variance_plot=pca_variance_plot,
            correlation_heatmap=correlation_heatmap
        )
        
        # Save HTML report
        output_path = f'reports/{model_name}_report.html'
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_summary(self, model_name, metrics, feature_importance, residual_analysis):
        """Print a console summary of the model, metrics, feature importance, and residual analysis."""
        print(f"\n=== {model_name} Summary ===")
        print("\nMetrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        print("\nFeature Importance:")
        for feature, importance in feature_importance.items():
            print(f"{feature}: {importance:.4f}")
        
        print("\nResidual Analysis:")
        for test, result in residual_analysis.items():
            if isinstance(result, dict):
                print(f"\n{test}:")
                for sub_test, sub_result in result.items():
                    if isinstance(sub_result, dict):
                        print(f"  {sub_test}:")
                        for stat, value in sub_result.items():
                            if isinstance(value, (int, float)):
                                print(f"    {stat}: {value:.4f}")
                            else:
                                print(f"    {stat}: {value}")
                    else:
                        if isinstance(sub_result, (int, float)):
                            print(f"  {sub_test}: {sub_result:.4f}")
                        else:
                            print(f"  {sub_test}: {sub_result}")
            else:
                if isinstance(result, (int, float)):
                    print(f"{test}: {result:.4f}")
                else:
                    print(f"{test}: {result}")

    def generate_pdf_report(self):
        """Generate and save the PDF report"""
        self.doc.build(self.story)

    def cleanup_temp_files(self):
        """Remove temporary plot files after report generation."""
        # Patterns for temporary plot files
        patterns = [
            '*_actual_vs_predicted.png',
            '*_residuals.png',
            '*_error_distribution.png',
            'pca_variance.png',
            'correlation_matrix.png'
        ]
        
        # Directories to clean
        directories = ['.', 'reports']
        
        # Remove files matching patterns in each directory
        for directory in directories:
            for pattern in patterns:
                for file in glob.glob(os.path.join(directory, pattern)):
                    try:
                        os.remove(file)
                        print(f"Removed temporary file: {file}")
                    except Exception as e:
                        print(f"Error removing {file}: {e}")

    def generate_combined_word_report(self, model_name, metrics, feature_importance, correlation_matrix, pca_results, 
                                    residual_analysis, actual_vs_predicted, residuals_plot, error_distribution, 
                                    pca_variance_plot, correlation_heatmap):
        """Generate a Word document containing all reports."""
        doc = Document()
        
        # Add title
        doc.add_heading(f'Regression Analysis Report - {model_name}', 0)
        doc.add_paragraph(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Add correlation matrix section
        doc.add_heading('Feature Correlation Analysis', level=1)
        doc.add_paragraph('The correlation matrix shows the relationships between different features:')
        
        # Add correlation matrix as table
        correlation_table = doc.add_table(rows=1, cols=len(correlation_matrix.columns) + 1)
        correlation_table.style = 'Table Grid'
        
        # Add headers
        header_cells = correlation_table.rows[0].cells
        header_cells[0].text = 'Feature'
        for i, col in enumerate(correlation_matrix.columns):
            header_cells[i + 1].text = col
        
        # Add correlation values
        for i, (index, row) in enumerate(correlation_matrix.iterrows()):
            row_cells = correlation_table.add_row().cells
            row_cells[0].text = index
            for j, value in enumerate(row):
                row_cells[j + 1].text = f'{value:.3f}'
        
        # Add correlation heatmap (only once)
        if isinstance(correlation_heatmap, dict):
            # Get the first available plot path
            plot_path = next((path for path in correlation_heatmap.values() if path and os.path.exists(path)), None)
            if plot_path:
                doc.add_paragraph('\nCorrelation Heatmap:')
                doc.add_picture(plot_path, width=Inches(6))
        
        # Add PCA analysis section
        doc.add_heading('Principal Component Analysis', level=1)
        
        # Add explained variance ratio
        doc.add_heading('Explained Variance Ratio', level=2)
        pca_table = doc.add_table(rows=1, cols=3)
        pca_table.style = 'Table Grid'
        header_cells = pca_table.rows[0].cells
        header_cells[0].text = 'Component'
        header_cells[1].text = 'Explained Variance'
        header_cells[2].text = 'Cumulative Variance'
        
        cumulative_variance = 0
        for i, var in enumerate(pca_results['explained_variance_ratio']):
            cumulative_variance += var
            row_cells = pca_table.add_row().cells
            row_cells[0].text = f'PC{i+1}'
            row_cells[1].text = f'{var:.4f}'
            row_cells[2].text = f'{cumulative_variance:.4f}'
        
        # Add component loadings table
        doc.add_heading('Component Loadings', level=2)
        loadings_table = doc.add_table(rows=1, cols=len(pca_results['components']) + 1)
        loadings_table.style = 'Table Grid'
        
        # Add headers
        header_cells = loadings_table.rows[0].cells
        header_cells[0].text = 'Feature'
        for i in range(len(pca_results['components'])):
            header_cells[i + 1].text = f'PC{i+1}'
        
        # Add loadings values
        for i, feature in enumerate(correlation_matrix.columns):
            row_cells = loadings_table.add_row().cells
            row_cells[0].text = feature
            for j in range(len(pca_results['components'])):
                row_cells[j + 1].text = f'{pca_results["components"][j][i]:.4f}'
        
        # Add PCA variance plot (only once)
        if isinstance(pca_variance_plot, dict):
            # Get the first available plot path
            plot_path = next((path for path in pca_variance_plot.values() if path and os.path.exists(path)), None)
            if plot_path:
                doc.add_paragraph('\nPCA Explained Variance Plot:')
                doc.add_picture(plot_path, width=Inches(6))
        
        # Add model equations section
        doc.add_heading('Model Equations', level=1)
        for model_name, model_metrics in metrics.items():
            if 'coefficients' in model_metrics and 'intercept' in model_metrics:
                doc.add_heading(f'{model_name} Equation', level=2)
                equation = f"y = {model_metrics['intercept']:.4f}"
                for feature, coef in model_metrics['coefficients'].items():
                    equation += f" + {coef:.4f} * {feature}"
                doc.add_paragraph(equation)
        
        # Add model metrics section
        doc.add_heading('Model Metrics', level=1)
        for model_name, model_metrics in metrics.items():
            doc.add_heading(model_name, level=2)
            metrics_table = doc.add_table(rows=1, cols=2)
            metrics_table.style = 'Table Grid'
            header_cells = metrics_table.rows[0].cells
            header_cells[0].text = 'Metric'
            header_cells[1].text = 'Value'
            
            for metric, value in model_metrics.items():
                if metric not in ['coefficients', 'intercept']:  # Skip coefficients and intercept as they're in equations
                    row_cells = metrics_table.add_row().cells
                    row_cells[0].text = metric
                    row_cells[1].text = f'{value:.4f}'
        
        # Add feature importance section
        doc.add_heading('Feature Importance', level=1)
        for model_name, importance in feature_importance.items():
            doc.add_heading(model_name, level=2)
            importance_table = doc.add_table(rows=1, cols=2)
            importance_table.style = 'Table Grid'
            header_cells = importance_table.rows[0].cells
            header_cells[0].text = 'Feature'
            header_cells[1].text = 'Importance'
            
            for feature, imp in importance.items():
                row_cells = importance_table.add_row().cells
                row_cells[0].text = feature
                row_cells[1].text = f'{imp:.4f}'
        
        # Add residual analysis section
        doc.add_heading('Residual Analysis', level=1)
        for model_name, analysis in residual_analysis.items():
            doc.add_heading(model_name, level=2)
            residual_table = doc.add_table(rows=1, cols=2)
            residual_table.style = 'Table Grid'
            header_cells = residual_table.rows[0].cells
            header_cells[0].text = 'Test'
            header_cells[1].text = 'Result'
            
            for test, result in analysis.items():
                row_cells = residual_table.add_row().cells
                row_cells[0].text = test
                row_cells[1].text = str(result)
        
        # Add plots section
        doc.add_heading('Model Visualizations', level=1)
        
        # Add actual vs predicted plots
        if isinstance(actual_vs_predicted, dict):
            doc.add_heading('Actual vs Predicted Values', level=2)
            for model_name, plot_path in actual_vs_predicted.items():
                if plot_path and os.path.exists(plot_path):
                    doc.add_heading(f'{model_name}', level=3)
                    doc.add_picture(plot_path, width=Inches(6))
        
        # Add residuals plots
        if isinstance(residuals_plot, dict):
            doc.add_heading('Residuals Plots', level=2)
            for model_name, plot_path in residuals_plot.items():
                if plot_path and os.path.exists(plot_path):
                    doc.add_heading(f'{model_name}', level=3)
                    doc.add_picture(plot_path, width=Inches(6))
        
        # Add error distribution plots
        if isinstance(error_distribution, dict):
            doc.add_heading('Error Distribution', level=2)
            for model_name, plot_path in error_distribution.items():
                if plot_path and os.path.exists(plot_path):
                    doc.add_heading(f'{model_name}', level=3)
                    doc.add_picture(plot_path, width=Inches(6))
        
        # Save the document
        output_path = os.path.join(self.output_dir, f'combined_regression_analysis.docx')
        doc.save(output_path)
        
        # Clean up temporary files
        self.cleanup_temp_files()
        
        return output_path 