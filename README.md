# Process Mining Event Log Assessment Assistant

A Python tool designed to help process mining consultants efficiently assess and prepare data from various source systems to create high-quality event logs for process mining analysis.

## Features

- **Multi-format Data Ingestion**: Parse CSV, Excel, JSON files, and database schemas
- **AI-Powered Analysis**: Leverage Azure OpenAI to analyze data structures and business context
- **Case ID Detection**: Identify potential unique identifiers for process cases
- **Activity Discovery**: Find and categorize potential process activities in source data
- **Attribute Mapping**: Discover valuable case and event attributes for process analysis
- **Data Quality Assessment**: Evaluate data completeness and suitability for process mining
- **Event Log Recommendations**: Generate structured recommendations in YAML format

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

### Basic Assessment
Analyze data files and get event log recommendations:
```bash
python -m src.main assess --data-files data1.csv data2.xlsx --context "description.txt" --output results.yaml
```

### Interactive Mode
Run in interactive mode for guided analysis:
```bash
python -m src.main interactive
```

### Schema Analysis
Analyze database schemas along with sample data:
```bash
# Single schema file
python -m src.main assess --schema schema.sql --data-files sample_data.csv --context process_description.txt

# Multiple schema files (XSD, SQL, XML)
python -m src.main assess --schema-files schema1.xsd schema2.sql --data-files sample_data.csv --context process_description.txt

# Directory processing (recursively finds all data and schema files)
python -m src.main assess --data-files ./data_directory/ --schema-files ./schemas/ --context process_description.txt
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
│   ├── main.py                 # Main CLI application
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py   # Data loading and initial analysis
│   │   ├── schema_analyzer.py  # Database schema parsing
│   │   ├── ai_analyzer.py      # Azure OpenAI integration
│   │   └── event_log_analyzer.py # Process mining assessment
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   └── test_core/
│       ├── __init__.py
│       ├── test_data_ingestion.py
│       └── test_event_log_analyzer.py
├── data/
│   ├── sample_processes.csv    # Sample event log data
│   └── business_context.txt    # Sample business context
├── docs/
├── requirements.txt
├── setup.py
├── .env.example               # Azure OpenAI configuration template
├── README.md
└── .gitignore
```

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
