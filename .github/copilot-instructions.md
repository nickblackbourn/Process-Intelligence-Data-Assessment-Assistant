<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Copilot Instructions for Process Mining Event Log Assessment Assistant

## Project Overview
This is a Python tool designed to help process mining consultants efficiently assess and prepare data from various source systems to create high-quality event logs for process mining analysis.

## Core Functionality
- **Data Ingestion**: Load and analyze data from CSV, Excel, JSON, and database schemas
- **AI-Powered Analysis**: Leverage Azure OpenAI to understand business context and data structures
- **Process Mining Assessment**: Identify case IDs, activities, timestamps, and attributes
- **Event Log Recommendations**: Generate structured YAML recommendations for event log creation

## Code Style and Standards
- Use Python 3.8+ features
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all classes and functions using Google style
- Prefer pandas for data manipulation
- Use Azure OpenAI for intelligent analysis

## Project Structure
- `src/main.py` - CLI application with Click framework
- `src/core/data_ingestion.py` - Data loading and initial analysis
- `src/core/schema_analyzer.py` - Database schema parsing (SQL, XML)
- `src/core/ai_analyzer.py` - Azure OpenAI integration for intelligent insights
- `src/core/event_log_analyzer.py` - Comprehensive process mining assessment
- `src/utils/` - Utility functions and helpers
- `tests/` - Unit tests using pytest

## Coding Preferences
- Use meaningful variable names that describe process mining concepts
- Focus on case-centric data structures
- Handle multiple data sources and formats gracefully
- Include comprehensive error handling and logging
- Write unit tests for all new functionality
- Generate structured YAML output for assessments

## Process Mining Domain Knowledge
- **Case ID**: Unique identifier for process instances (orders, tickets, applications)
- **Activity**: What happened in the process (events, tasks, milestones)
- **Timestamp**: When activities occurred (event time)
- **Case Attributes**: Properties of the process instance (amount, customer, region)
- **Event Attributes**: Properties of individual activities (user, department, resource)

## AI Integration Guidelines
- Use Azure OpenAI for business context analysis
- Provide fallback analysis when AI is unavailable
- Structure prompts for process mining domain expertise
- Parse AI responses into structured recommendations
- Combine AI insights with rule-based analysis

## Dependencies
- pandas for data manipulation
- click for CLI interface
- openai and azure-identity for AI integration
- sqlparse for database schema parsing
- pyyaml for output formatting
- nltk/textblob for text processing
