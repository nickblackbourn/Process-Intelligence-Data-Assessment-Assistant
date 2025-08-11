# Process Intelligence Data Assessment Assistant - User Guide

## Overview

The Process Intelligence Data Assessment Assistant is a Python application designed to help organizations analyze their business processes and assess data quality. It provides comprehensive insights into process efficiency, data completeness, and recommendations for improvement.

## Features

### Data Processing
- **Multi-format Support**: Load data from CSV, Excel, JSON, and Parquet files
- **Data Cleaning**: Automatic removal of empty rows/columns and data standardization
- **Statistics Calculation**: Basic statistical analysis of your process data

### Data Quality Assessment
- **Completeness Analysis**: Identify missing data and calculate completeness scores
- **Consistency Checking**: Detect inconsistencies in data formats and values
- **Validity Assessment**: Find outliers and invalid data patterns
- **Duplicate Detection**: Identify and report duplicate records

### Process Intelligence
- **Process Clustering**: Group similar processes for analysis
- **Performance Metrics**: Calculate duration, complexity, and automation scores
- **Recommendations**: Get actionable insights for process improvement

## Quick Start

### 1. Installation
The project is already set up with a virtual environment and all dependencies installed.

### 2. Running the Application

#### Demo Mode (No Input File)
```bash
python -m src.main
```

#### With Your Own Data
```bash
python -m src.main -i your_data_file.csv -o output_directory
```

#### With Verbose Logging
```bash
python -m src.main -i your_data_file.csv -o output_directory --verbose
```

### 3. Using VS Code Tasks

You can use the pre-configured VS Code tasks:

1. **Ctrl+Shift+P** → **Tasks: Run Task**
2. Select one of:
   - "Run Process Intelligence Assistant" (demo mode)
   - "Run Process Intelligence Assistant (with input file)"
   - "Install Dependencies"
   - "Run Tests"
   - "Format Code with Black"

### 4. Debugging

Use VS Code's built-in debugger:

1. **F5** to start debugging
2. Choose configuration:
   - "Run Process Intelligence Assistant (Demo)"
   - "Run Process Intelligence Assistant (with file)"

## Input Data Format

Your data should be in CSV, Excel, or JSON format with columns representing process attributes. Example:

```csv
process_id,process_name,duration_hours,complexity_score,automation_level,department
1,Order Processing,2.5,7,0.8,Sales
2,Payment Processing,1.0,4,0.9,Finance
3,Shipping,24.0,9,0.6,Logistics
```

### Recommended Columns
- **process_id**: Unique identifier for each process
- **process_name**: Descriptive name of the process
- **duration_hours**: Time taken to complete the process
- **complexity_score**: Subjective complexity rating (1-10)
- **automation_level**: Percentage of automation (0.0-1.0)
- **department**: Organizational unit responsible

## Output

The application generates:

### 1. JSON Results (`assessment_results.json`)
Detailed assessment including:
- Overall quality score
- Individual metric scores
- Column-by-column analysis
- Data type distribution
- Recommendations

### 2. Summary Report (`summary.txt`)
Human-readable summary of key findings

## Understanding the Scores

### Overall Quality Score
- **90-100**: Excellent data quality
- **70-89**: Good data quality with minor issues
- **50-69**: Moderate quality, improvement needed
- **Below 50**: Poor quality, significant action required

### Individual Metrics
- **Completeness**: Percentage of non-missing values
- **Consistency**: Uniformity of data formats and patterns
- **Validity**: Absence of outliers and invalid values
- **Duplicates**: Percentage of duplicate records

## Common Use Cases

### 1. Process Audit
Assess the quality of process documentation and identify gaps:
```bash
python -m src.main -i process_inventory.csv -o audit_results
```

### 2. Process Optimization
Identify processes suitable for automation:
```bash
python -m src.main -i process_metrics.csv -o optimization_analysis
```

### 3. Data Governance
Regular quality checks on process data:
```bash
python -m src.main -i monthly_processes.csv -o quality_reports
```

## Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Code Formatting
```bash
python -m black src/ tests/
```

### Type Checking
```bash
python -m mypy src/
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the virtual environment is activated
2. **File Not Found**: Check file paths are correct and files exist
3. **Memory Issues**: For large files, consider processing in chunks

### Getting Help

1. Check the logs in the `logs/` directory
2. Run with `--verbose` flag for detailed output
3. Review the error messages and stack traces

## Extending the Application

The modular design allows easy extension:

- **Add new data sources**: Extend `DataProcessor.load_data()`
- **Custom assessments**: Add methods to `AssessmentEngine`
- **New output formats**: Extend `DataProcessor.save_results()`
- **Additional metrics**: Implement in the core modules

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Recent Changes
- Removed unused scripts and redundant test data files.
- Enhanced AI prompt for better process identification.
