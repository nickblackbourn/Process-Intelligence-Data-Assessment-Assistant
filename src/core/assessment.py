"""Assessment engine for data quality and process intelligence analysis."""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)


class AssessmentEngine:
    """Engine for assessing data quality and process intelligence metrics."""
    
    def __init__(self):
        """Initialize the AssessmentEngine."""
        self.quality_thresholds = {
            'completeness': 0.95,
            'consistency': 0.90,
            'accuracy': 0.85,
            'validity': 0.90
        }
        logger.info("AssessmentEngine initialized")
    
    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality assessment.
        
        Args:
            data: Input DataFrame to assess
            
        Returns:
            Dictionary containing quality assessment results
        """
        logger.info("Starting data quality assessment")
        
        results = {
            'overall_score': 0.0,
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'validity_score': 0.0,
            'duplicate_percentage': 0.0,
            'missing_data_percentage': 0.0,
            'data_types_assessment': {},
            'column_analysis': {},
            'recommendations': []
        }
        
        # Completeness assessment
        results['completeness_score'] = self._assess_completeness(data)
        results['missing_data_percentage'] = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        
        # Consistency assessment
        results['consistency_score'] = self._assess_consistency(data)
        
        # Validity assessment
        results['validity_score'] = self._assess_validity(data)
        
        # Duplicate assessment
        results['duplicate_percentage'] = (data.duplicated().sum() / len(data)) * 100
        
        # Column-level analysis
        results['column_analysis'] = self._analyze_columns(data)
        
        # Data types assessment
        results['data_types_assessment'] = self._assess_data_types(data)
        
        # Calculate overall score
        results['overall_score'] = self._calculate_overall_score(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results, data)
        
        logger.info(f"Data quality assessment completed. Overall score: {results['overall_score']:.2f}")
        
        return results
    
    def _assess_completeness(self, data: pd.DataFrame) -> float:
        """Assess data completeness (non-null values)."""
        total_cells = len(data) * len(data.columns)
        non_null_cells = data.notna().sum().sum()
        completeness = (non_null_cells / total_cells) * 100
        return completeness
    
    def _assess_consistency(self, data: pd.DataFrame) -> float:
        """Assess data consistency across similar columns."""
        consistency_scores = []
        
        # Check for consistent data types in similar columns
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check string consistency (e.g., case, format)
                unique_values = data[col].dropna().unique()
                if len(unique_values) > 0:
                    # Calculate variation in string formats
                    string_lengths = [len(str(val)) for val in unique_values]
                    if len(string_lengths) > 1:
                        length_std = np.std(string_lengths)
                        consistency_score = max(0, 100 - (length_std * 10))
                    else:
                        consistency_score = 100
                    consistency_scores.append(consistency_score)
        
        return np.mean(consistency_scores) if consistency_scores else 100.0
    
    def _assess_validity(self, data: pd.DataFrame) -> float:
        """Assess data validity (reasonable values, proper formats)."""
        validity_scores = []
        
        for col in data.columns:
            col_data = data[col].dropna()
            
            if col_data.dtype in ['int64', 'float64']:
                # Check for outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_percentage = len(outliers) / len(col_data) * 100
                validity_score = max(0, 100 - outlier_percentage * 2)
                validity_scores.append(validity_score)
            
            elif col_data.dtype == 'object':
                # Check for valid string patterns
                non_empty_strings = col_data[col_data.astype(str).str.strip() != '']
                validity_score = (len(non_empty_strings) / len(col_data)) * 100
                validity_scores.append(validity_score)
        
        return np.mean(validity_scores) if validity_scores else 100.0
    
    def _analyze_columns(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze individual columns for detailed insights."""
        column_analysis = {}
        
        for col in data.columns:
            col_info = {
                'data_type': str(data[col].dtype),
                'missing_count': data[col].isnull().sum(),
                'missing_percentage': (data[col].isnull().sum() / len(data)) * 100,
                'unique_count': data[col].nunique(),
                'unique_percentage': (data[col].nunique() / len(data)) * 100
            }
            
            if data[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': data[col].mean(),
                    'median': data[col].median(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                })
            
            column_analysis[col] = col_info
        
        return column_analysis
    
    def _assess_data_types(self, data: pd.DataFrame) -> Dict[str, int]:
        """Assess distribution of data types."""
        type_counts = {}
        for dtype in data.dtypes:
            dtype_str = str(dtype)
            type_counts[dtype_str] = type_counts.get(dtype_str, 0) + 1
        
        return type_counts
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        weights = {
            'completeness': 0.3,
            'consistency': 0.25,
            'validity': 0.25,
            'duplicates': 0.2
        }
        
        duplicate_score = max(0, 100 - results['duplicate_percentage'] * 2)
        
        overall_score = (
            results['completeness_score'] * weights['completeness'] +
            results['consistency_score'] * weights['consistency'] +
            results['validity_score'] * weights['validity'] +
            duplicate_score * weights['duplicates']
        )
        
        return min(100, max(0, overall_score))
    
    def _generate_recommendations(self, results: Dict[str, Any], data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on assessment results."""
        recommendations = []
        
        # Missing data recommendations
        if results['missing_data_percentage'] > 5:
            recommendations.append(
                f"Address missing data: {results['missing_data_percentage']:.1f}% of data is missing. "
                "Consider data imputation or collection improvement."
            )
        
        # Duplicate data recommendations
        if results['duplicate_percentage'] > 1:
            recommendations.append(
                f"Remove duplicate records: {results['duplicate_percentage']:.1f}% of records are duplicates."
            )
        
        # Data consistency recommendations
        if results['consistency_score'] < 90:
            recommendations.append(
                "Improve data consistency: Standardize formats, case, and naming conventions."
            )
        
        # Column-specific recommendations
        for col, analysis in results['column_analysis'].items():
            if analysis['missing_percentage'] > 20:
                recommendations.append(
                    f"Column '{col}' has {analysis['missing_percentage']:.1f}% missing values. "
                    "Consider if this column is necessary or needs better data collection."
                )
        
        # Overall score recommendations
        if results['overall_score'] < 70:
            recommendations.append(
                "Overall data quality is below acceptable standards. "
                "Implement data governance policies and quality monitoring."
            )
        
        return recommendations
    
    def calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate a simple quality score for quick assessment."""
        completeness = self._assess_completeness(data)
        duplicates_penalty = (data.duplicated().sum() / len(data)) * 100
        
        # Simple scoring formula
        score = (completeness - duplicates_penalty) / 10
        return max(0, min(10, score))
    
    def get_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Get quick recommendations for data improvement."""
        recommendations = []
        
        # Check for common issues
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        duplicate_pct = (data.duplicated().sum() / len(data)) * 100
        
        if missing_pct > 10:
            recommendations.append("Reduce missing data through better data collection processes")
        
        if duplicate_pct > 5:
            recommendations.append("Implement duplicate detection and removal procedures")
        
        if len(data.select_dtypes(include=['object']).columns) > len(data.columns) * 0.7:
            recommendations.append("Consider converting text fields to categorical or numeric where appropriate")
        
        if len(recommendations) == 0:
            recommendations.append("Data quality looks good! Consider advanced analytics and automation")
        
        return recommendations
    
    def perform_process_clustering(self, data: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """Perform clustering analysis on process data."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            logger.warning("Insufficient numeric columns for clustering")
            return {'error': 'Insufficient numeric data for clustering'}
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data.fillna(numeric_data.mean()))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_data = numeric_data[clusters == i]
            cluster_analysis[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(data)) * 100,
                'characteristics': cluster_data.mean().to_dict()
            }
        
        return {
            'cluster_labels': clusters.tolist(),
            'cluster_analysis': cluster_analysis,
            'silhouette_score': 'Not calculated',  # Would need additional import
            'n_clusters': n_clusters
        }
