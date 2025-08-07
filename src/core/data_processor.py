"""Data ingestion engine for handling various data formats and operations."""

import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class DataIngestionEngine:
    """Handles data loading from various file formats and initial analysis."""
    
    def __init__(self):
        """Initialize the DataIngestionEngine."""
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
        logger.info("DataIngestionEngine initialized")
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded data as pandas DataFrame
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            if extension == '.csv':
                # Try different encodings and separators
                data = self._load_csv_flexible(file_path)
            elif extension in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            elif extension == '.json':
                data = self._load_json_flexible(file_path)
            elif extension == '.parquet':
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Handler not implemented for {extension}")
                
            logger.info(f"Successfully loaded {len(data)} rows and {len(data.columns)} columns")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _load_csv_flexible(self, file_path: str) -> pd.DataFrame:
        """Load CSV with flexible encoding and separator detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    data = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    if len(data.columns) > 1:  # Must have multiple columns
                        logger.info(f"CSV loaded with encoding={encoding}, separator='{sep}'")
                        return data
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
        
        # Fallback to default
        return pd.read_csv(file_path)
    
    def _load_json_flexible(self, file_path: str) -> pd.DataFrame:
        """Load JSON with flexible structure handling."""
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if isinstance(json_data, list):
            return pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            # Try to find the main data array
            for key, value in json_data.items():
                if isinstance(value, list) and len(value) > 0:
                    return pd.DataFrame(value)
            # If no array found, treat as single record
            return pd.DataFrame([json_data])
        else:
            raise ValueError("Unsupported JSON structure")
    
    def analyze_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the structure and characteristics of the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with structural analysis
        """
        analysis = {
            'shape': {
                'rows': len(data),
                'columns': len(data.columns)
            },
            'columns': {},
            'data_types': {},
            'missing_data': {},
            'potential_identifiers': [],
            'potential_timestamps': [],
            'potential_activities': [],
            'data_quality_indicators': {}
        }
        
        # Analyze each column
        for col in data.columns:
            col_analysis = self._analyze_column(data, col)
            analysis['columns'][col] = col_analysis
            
            # Categorize columns based on characteristics
            if col_analysis['could_be_identifier']:
                analysis['potential_identifiers'].append(col)
            
            if col_analysis['could_be_timestamp']:
                analysis['potential_timestamps'].append(col)
                
            if col_analysis['could_be_activity']:
                analysis['potential_activities'].append(col)
        
        # Data type distribution
        type_counts = data.dtypes.value_counts()
        analysis['data_types'] = {str(dtype): int(count) for dtype, count in type_counts.items()}
        
        # Missing data analysis
        missing_counts = data.isnull().sum()
        analysis['missing_data'] = {
            'total_missing': int(missing_counts.sum()),
            'missing_percentage': float((missing_counts.sum() / (len(data) * len(data.columns))) * 100),
            'columns_with_missing': [col for col in data.columns if data[col].isnull().any()]
        }
        
        # Data quality indicators
        analysis['data_quality_indicators'] = {
            'completeness': float(100 - analysis['missing_data']['missing_percentage']),
            'duplicate_rows': int(data.duplicated().sum()),
            'duplicate_percentage': float((data.duplicated().sum() / len(data)) * 100),
            'empty_strings': self._count_empty_strings(data),
            'potential_data_issues': self._identify_data_issues(data)
        }
        
        return analysis
    
    def _analyze_column(self, data: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Analyze individual column characteristics."""
        col_data = data[col]
        
        analysis = {
            'dtype': str(col_data.dtype),
            'unique_count': col_data.nunique(),
            'unique_percentage': (col_data.nunique() / len(col_data)) * 100,
            'missing_count': col_data.isnull().sum(),
            'missing_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
            'could_be_identifier': False,
            'could_be_timestamp': False,
            'could_be_activity': False,
            'sample_values': []
        }
        
        # Get sample values (non-null)
        non_null_values = col_data.dropna()
        if len(non_null_values) > 0:
            sample_size = min(5, len(non_null_values))
            analysis['sample_values'] = non_null_values.head(sample_size).tolist()
        
        # Check if could be identifier
        if analysis['unique_percentage'] > 80 and analysis['missing_percentage'] < 10:
            analysis['could_be_identifier'] = True
        
        # Check if could be timestamp
        if col_data.dtype == 'object':
            # Try to parse as datetime
            try:
                pd.to_datetime(non_null_values.head(100), errors='raise')
                analysis['could_be_timestamp'] = True
            except:
                pass
            
            # Check for activity-like patterns
            if (analysis['unique_count'] < len(col_data) * 0.5 and 
                analysis['unique_count'] > 2 and 
                analysis['missing_percentage'] < 20):
                # Check if values look like activities
                sample_str = str(non_null_values.iloc[0]).lower() if len(non_null_values) > 0 else ""
                activity_keywords = ['create', 'process', 'send', 'receive', 'approve', 'reject', 
                                   'start', 'end', 'complete', 'cancel', 'update', 'delete']
                if any(keyword in sample_str for keyword in activity_keywords):
                    analysis['could_be_activity'] = True
        
        # Additional numeric analysis
        if col_data.dtype in ['int64', 'float64']:
            numeric_data = col_data.dropna()
            if len(numeric_data) > 0:
                analysis.update({
                    'mean': float(numeric_data.mean()),
                    'median': float(numeric_data.median()),
                    'std': float(numeric_data.std()) if len(numeric_data) > 1 else 0,
                    'min': float(numeric_data.min()),
                    'max': float(numeric_data.max())
                })
        
        return analysis
    
    def _count_empty_strings(self, data: pd.DataFrame) -> int:
        """Count empty strings across all object columns."""
        empty_count = 0
        for col in data.select_dtypes(include=['object']).columns:
            empty_count += (data[col].astype(str).str.strip() == '').sum()
        return int(empty_count)
    
    def _identify_data_issues(self, data: pd.DataFrame) -> List[str]:
        """Identify potential data quality issues."""
        issues = []
        
        # Check for columns with very high missing data
        for col in data.columns:
            missing_pct = (data[col].isnull().sum() / len(data)) * 100
            if missing_pct > 50:
                issues.append(f"Column '{col}' has {missing_pct:.1f}% missing data")
        
        # Check for columns with very low variability
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].nunique() == 1:
                issues.append(f"Column '{col}' has only one unique value")
        
        # Check for suspicious patterns
        if len(data) < 10:
            issues.append("Dataset is very small (less than 10 rows)")
        
        if len(data.columns) < 3:
            issues.append("Dataset has very few columns for process mining")
        
        return issues
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data for analysis.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Cleaned data DataFrame
        """
        logger.info("Starting data cleaning")
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Remove completely empty rows and columns
        cleaned_data = cleaned_data.dropna(how='all', axis=0)
        cleaned_data = cleaned_data.dropna(how='all', axis=1)
        
        # Strip whitespace from string columns
        string_cols = cleaned_data.select_dtypes(include=['object']).columns
        for col in string_cols:
            cleaned_data[col] = cleaned_data[col].astype(str).str.strip()
            # Replace empty strings with NaN
            cleaned_data[col] = cleaned_data[col].replace('', pd.NA)
        
        # Convert potential numeric columns
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(cleaned_data[col], errors='ignore')
                if not numeric_series.equals(cleaned_data[col]):
                    cleaned_data[col] = numeric_series
        
        logger.info(f"Data cleaning completed. Shape: {cleaned_data.shape}")
        return cleaned_data
    
    def export_to_yaml(self, data: Dict[str, Any], file_path: str) -> None:
        """Export analysis results to YAML file.
        
        Args:
            data: Data to export
            file_path: Output file path
        """
        import yaml
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Data exported to YAML: {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to YAML: {e}")
            raise
