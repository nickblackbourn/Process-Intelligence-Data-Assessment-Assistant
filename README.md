# Process Mining Event Log Assessment Assistant

A comprehensive Python tool designed to help process mining consultants efficiently assess and prepare data from various source systems to create high-quality event logs for process mining analysis.

## 🚀 **Version 2.0 - Enhanced Multi-Tab Excel & Output Management**

### **New Features**
- **🗂️ Multi-Tab Excel Processing**: Automatically processes all Excel sheets with embedded schema detection
- **📁 Intelligent Output Management**: Organized file structure with date-based folders and contextual naming
- **🔍 Enhanced Schema Detection**: Identifies data dictionary, process mapping, and lookup tables within Excel files
- **🎯 UX-Focused Design**: Clean workspace organization and professional output management

## Features

### **Core Capabilities**
- **Multi-format Data Ingestion**: CSV, Excel (single/multi-tab), JSON, database schemas (XSD, SQL DDL)
- **AI-Powered Analysis**: Leverage Azure OpenAI for intelligent data structure and business context analysis
- **Advanced Excel Processing**: Multi-tab file handling with automatic schema detection and cross-tab analysis
- **Comprehensive Assessment**: Case ID detection, activity discovery, attribute mapping, data quality evaluation
- **Professional Output**: Organized results with intelligent naming and archiving

### **Enhanced Excel Processing (v2.0)**
- **Multi-Tab Analysis**: Processes all Excel sheets simultaneously
- **Schema Detection**: Automatically identifies embedded schemas in metadata tabs
- **Tab Classification**: Distinguishes between data tabs and schema definition tabs
- **Cross-Tab Relationships**: Analyzes relationships between different Excel sheets
- **Enterprise Ready**: Handles complex business documents with mixed content types

### **Intelligent Output Management (v2.0)**
- **Organized Structure**: Date-based folders with logical file hierarchy
- **Contextual Naming**: File names reflect analyzed data sources
- **Automatic Archiving**: Previous results preserved when running new analyses
- **Latest Links**: Easy access to most recent results
- **Multiple Formats**: YAML and JSON output options

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Process-Intelligence-Data-Assessment-Assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Install the package in development mode:
```bash
pip install -e .
```

6. Configure Azure OpenAI:
Create a `.env` file in the project root:
```env
AZURE_OPENAI_ENDPOINT=your-endpoint-url
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Usage

### **Enhanced Assessment (v2.0)**
New organized output management with intelligent file naming:
```bash
# Modern organized output (recommended)
python -m src.main assess --data-files data.xlsx --output-name "Order_Analysis"

# Multi-tab Excel processing
python -m src.main assess --data-files complex_workbook.xlsx --output-name "Enterprise_Analysis"

# Custom output directory and format
python -m src.main assess --data-files data.csv --output-dir custom_results --output-format json

# Legacy mode (backward compatibility)
python -m src.main assess --data-files data.csv --output results.yaml
```

### **Multi-Tab Excel Processing**
Enhanced Excel analysis with schema detection:
```bash
# Process multi-tab Excel with embedded schemas
python -m src.main assess --data-files enterprise_data.xlsx --output-name "Multi_Tab_Analysis"

# Results automatically organized:
# results/assessments/2025-08-08/Multi_Tab_Analysis_2025-08-08_15-30-45.yaml
# results/assessments/latest/latest_assessment.yaml
```

### **Output Management**
New commands for managing organized results:
```bash
# View organized file structure and statistics
python -m src.main manage-outputs

# Clean up old files (30+ days)
python -m src.main manage-outputs --cleanup-days 30

# Organize legacy messy files
python -m src.main organize-legacy-files
```

### **Traditional Features**
All existing functionality enhanced:
```bash
# Interactive mode for guided analysis
python -m src.main interactive

# Schema analysis with sample data
python -m src.main assess --schema schema.sql --data-files sample_data.csv

# Directory processing (batch analysis)
python -m src.main assess --directory ./data --schema-files ./schemas --context process_description.txt

# Demo mode to see capabilities
python -m src.main demo
```

### **Advanced Options**
```bash
# Keep history vs overwrite
python -m src.main assess --data-files data.xlsx --keep-history  # Archives previous results
python -m src.main assess --data-files data.xlsx --overwrite     # Replaces previous results

# Multiple schema files (XSD, SQL, XML)
python -m src.main assess --schema-files schema1.xsd schema2.sql --data-files data.csv

# Comprehensive analysis with business context
python -m src.main assess --data-files data.xlsx --context business_rules.txt --output-name "Full_Analysis"
```

## Development

### Setting up development environment

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

3. Run code formatting:
```bash
black src/ tests/
```

4. Run linting:
```bash
flake8 src/ tests/
mypy src/
```

## Project Structure

```
Process-Intelligence-Data-Assessment-Assistant/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Enhanced CLI with output management
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Multi-tab Excel & data loading  
│   │   ├── schema_analyzer.py     # Database schema parsing
│   │   ├── ai_analyzer.py         # Azure OpenAI integration
│   │   └── event_log_analyzer.py  # Process mining assessment
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py             # Utility functions
│   │   └── output_manager.py      # NEW: Intelligent output management
│   └── results/                   # NEW: Organized output directory
│       ├── assessments/
│       │   ├── 2025-08-08/       # Date-based organization
│       │   └── latest/           # Latest results
│       ├── reports/              # Future: HTML/PDF reports
│       ├── sql/                  # Future: Generated SQL
│       └── archives/             # Archived results
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   └── test_core/
│       ├── __init__.py
│       ├── test_data_ingestion.py
│       └── test_event_log_analyzer.py
├── data/
│   ├── sample_processes.csv       # Sample event log data
│   └── business_context.txt       # Sample business context
├── test_data/                     # Test files for development
├── docs/
├── requirements.txt
├── setup.py
├── .env.example                   # Azure OpenAI configuration template
├── README.md
└── .gitignore
```

## Key Enhancements (v2.0)

### **Multi-Tab Excel Processing**
- **Automatic Detection**: Identifies Excel files with multiple sheets
- **Schema Recognition**: Detects embedded schemas in metadata tabs
- **Tab Classification**: Data tabs vs schema definition tabs
- **Cross-Reference**: Maintains relationships between tabs
- **Enterprise Ready**: Handles complex business documents

### **Intelligent Output Management**
- **Organized Structure**: `results/assessments/YYYY-MM-DD/` format
- **Contextual Naming**: Files named after analyzed sources
- **History Management**: Automatic archiving of previous results
- **Latest Access**: Symlinks to newest results for easy access
- **Format Options**: YAML and JSON output support

### **Enhanced User Experience**
- **Clean Workspace**: No more file clutter in project root
- **Predictable Organization**: Know exactly where results are saved
- **Professional Output**: Enterprise-ready file management
- **Backward Compatibility**: Legacy output options still supported

## Example Usage

### Basic Assessment
```bash
# Analyze sample data
python -m src.main assess --data-files data/sample_processes.csv --context data/business_context.txt

# Analyze multiple files with schema
python -m src.main assess --data-files data1.csv data2.xlsx --schema schema.sql --context description.txt

# Multiple schema files (mixed formats)
python -m src.main assess --data-files data1.csv --schema-files schema.xsd database.sql --context description.txt

# Directory processing (recursively discovers files)
python -m src.main assess --data-files ./data/ --schema-files ./schemas/ --context process_info.txt

# Run demo to see capabilities
python -m src.main demo
```

### Sample Output
The tool generates a comprehensive YAML assessment including:
- **Case ID candidates** with confidence scores
- **Activity mapping** recommendations
- **Timestamp analysis** and temporal coverage
- **Attribute suggestions** for case and event data
- **Data quality assessment** with specific issues
- **Process mining readiness score**
- **Step-by-step transformation plan**
- **Suggested SQL queries** for data extraction
- **Files considered** for full provenance tracking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
