#!/usr/bin/env python3
"""
Process Mining Event Log Assessment Assistant - Main CLI Application

This module provides the command-line interface for the Process Mining Event Log 
Assessment Assistant, a tool designed to help process mining consultants efficiently 
assess and prepare data from various source systems to create high-quality event logs.

Key Features:
- Multi-format data ingestion (CSV, Excel, JSON, database schemas)
- AI-powered analysis with Azure OpenAI integration
- Schema analysis for XSD, SQL DDL, and embedded Excel schemas
- Multi-tab Excel file processing with automatic schema detection
- Intelligent output management with organized file structure
- Comprehensive process mining readiness assessment

Usage:
    python main.py assess --data-files data.xlsx --schema-files schema.xsd
    python main.py interactive
    python main.py demo
    python main.py manage-outputs

Author: Process Intelligence Team
Version: 2.0.0 - Enhanced Multi-Tab Excel & Output Management
"""

import glob
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import click
import yaml

from core.data_ingestion import DataIngestionEngine
from core.schema_analyzer import SchemaAnalyzer
from core.ai_analyzer import AIAnalyzer
from core.event_log_analyzer import EventLogAnalyzer
from utils.helpers import setup_logging, load_environment
from utils.output_manager import OutputManager


def validate_ai_environment() -> bool:
    """Validate that Azure OpenAI environment variables are configured."""
    required_vars = {
        'AZURE_OPENAI_ENDPOINT': 'Azure OpenAI service endpoint URL',
        'AZURE_OPENAI_API_KEY': 'Azure OpenAI API key', 
        'AZURE_OPENAI_DEPLOYMENT_NAME': 'Azure OpenAI deployment name (e.g., gpt-4)'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  {var}: {description}")
    
    if missing_vars:
        click.echo("⚠️  Azure OpenAI environment variables not configured:")
        for var in missing_vars:
            click.echo(var)
        click.echo("\n💡 AI-powered analysis will be disabled. Set these variables in your .env file for enhanced analysis.")
        click.echo("   Tool will continue with rule-based analysis.")
        return False
    else:
        click.echo("✅ Azure OpenAI environment configured")
        return True


@click.group()
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Set the logging level'
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Enable verbose output'
)
def cli(log_level: str, verbose: bool) -> None:
    """Process Mining Event Log Assessment Assistant.
    
    A comprehensive tool for process mining consultants to assess and prepare data
    from various source systems. Supports multi-format data ingestion, AI-powered
    analysis, schema detection, and intelligent output management.
    
    Enhanced Features (v2.0):
    - Multi-tab Excel processing with embedded schema detection
    - Organized output management with date-based folders
    - Contextual file naming and automatic archiving
    - Cross-platform compatibility and enterprise readiness
    """
    setup_logging(log_level, verbose)
    load_environment()


@cli.command()
@click.option(
    '--data-files',
    '-d',
    multiple=True,
    type=click.Path(exists=True),
    help='Data files to analyze (CSV, Excel, JSON)'
)
@click.option(
    '--schema',
    '-s',
    type=click.Path(exists=True),
    help='Database schema file (SQL DDL, XML)'
)
@click.option(
    '--schema-files',
    multiple=True,
    type=click.Path(exists=True),
    help='One or more schema files (SQL DDL, XSD/XML)'
)
@click.option(
    '--directory',
    '--dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Directory containing multiple data files or schemas to analyze'
)
@click.option(
    '--context',
    '-c',
    type=click.Path(exists=True),
    help='Text file with business context and process description'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help='Output file for assessment results (legacy option, use --output-name instead)'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='results',
    help='Base directory for organized output files'
)
@click.option(
    '--output-name',
    type=str,
    help='Custom base name for output files (auto-generated if not provided)'
)
@click.option(
    '--output-format',
    type=click.Choice(['yaml', 'json']),
    default='yaml',
    help='Output file format'
)
@click.option(
    '--keep-history/--overwrite',
    default=True,
    help='Keep previous results (archive) or overwrite them'
)
def assess(
    data_files: tuple,
    schema: Optional[str],
    schema_files: tuple,
    directory: Optional[str],
    context: Optional[str],
    output: Optional[str],
    output_dir: str,
    output_name: Optional[str],
    output_format: str,
    keep_history: bool
) -> None:
    """Assess data sources for process mining event log creation.
    
    This command performs comprehensive analysis of data sources to determine their
    suitability for process mining event log creation. It supports:
    
    Data Sources:
    - CSV, Excel (single/multi-tab), JSON files
    - Database schemas (XSD, SQL DDL)
    - Directory scanning for batch processing
    
    Analysis Features:
    - Multi-tab Excel processing with schema detection
    - AI-powered business context analysis (when configured)
    - Case ID, activity, and timestamp candidate identification
    - Data quality assessment and readiness scoring
    - SQL generation for event log assembly
    
    Output Management:
    - Organized results in date-based folders
    - Contextual file naming based on analyzed sources
    - Automatic archiving of previous results
    - Latest symlinks for easy access
    
    Examples:
        python main.py assess --data-files orders.xlsx --output-name "Order_Analysis"
        python main.py assess --directory ./data --schema-files schema.xsd
        python main.py assess --data-files data.csv --context business_rules.txt
    """
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Process Mining Event Log Assessment")
    
    try:
        # Load business context if provided
        business_context = ""
        if context:
            with open(context, 'r', encoding='utf-8') as f:
                business_context = f.read()
            logger.info(f"Loaded business context from {context}")
        
        # Collect all files to process
        files_to_process: List[str] = []
        files_to_process.extend(list(data_files) if data_files else [])
        files_to_process.extend(list(schema_files) if schema_files else [])
        
        # Add directory files if specified (recursive)
        if directory:
            dir_path = Path(directory)
            patterns = ['*.csv', '*.xlsx', '*.xls', '*.json', '*.xsd', '*.xml', '*.sql', '*.ddl']
            for pattern in patterns:
                for p in dir_path.rglob(pattern):
                    files_to_process.append(str(p))
            logger.info(f"Found {len(files_to_process)} files (including directory scan) from {directory}")
        
        # Add single schema file if specified separately
        if schema:
            files_to_process.append(schema)
        
        # Deduplicate while preserving order
        files_to_process = list(dict.fromkeys(files_to_process))
        
        if not files_to_process:
            click.echo("❌ No files specified for analysis. Use --data-files, --schema/--schema-files, or --directory")
            return
        
        # Validate Azure OpenAI environment
        ai_available = validate_ai_environment()
        if ai_available:
            click.echo("🔬 Enhanced AI analysis enabled")
        else:
            click.echo("📊 Using rule-based analysis with business intelligence fallback")
        
        logger.info(f"Processing {len(files_to_process)} files: {[os.path.basename(f) for f in files_to_process]}")
        
        # Initialize components
        data_ingestion = DataIngestionEngine()
        schema_analyzer = SchemaAnalyzer()
        ai_analyzer = AIAnalyzer()
        event_log_analyzer = EventLogAnalyzer()
        
        # Process all files
        datasets = []
        schema_analyses = []
        
        for file_path in files_to_process:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_path}")
            
            if file_ext in ['.csv', '.xlsx', '.xls', '.json']:
                # Process data files
                try:
                    data_result = data_ingestion.load_file(file_path)
                    
                    # Handle multi-tab Excel files
                    if isinstance(data_result, dict) and 'multi_tab_file' in data_result.get('metadata', {}):
                        # Multi-tab Excel file
                        click.echo(f"✅ Loaded multi-tab Excel: {os.path.basename(file_path)} ({data_result['metadata']['total_tabs']} tabs)")
                        
                        for tab_name, tab_info in data_result['tabs'].items():
                            datasets.append({
                                'file_path': tab_info['source_reference'],
                                'data': tab_info['data'],
                                'metadata': tab_info['metadata'],
                                'excel_tab_name': tab_name,
                                'excel_analysis': tab_info['analysis'],
                                'is_multi_tab': True
                            })
                            
                            # If tab contains schema information, add to schema analyses
                            if tab_info['analysis']['is_schema_definition']:
                                schema_analyses.append({
                                    'file_path': tab_info['source_reference'],
                                    'schema_info': {
                                        'type': 'excel_embedded_schema',
                                        'elements': tab_info['analysis']['schema_elements'],
                                        'schema_type': tab_info['analysis']['schema_type'],
                                        'confidence': tab_info['analysis']['confidence'],
                                        'source_tab': tab_name
                                    }
                                })
                                click.echo(f"  📋 Schema detected in tab: {tab_name}")
                            else:
                                click.echo(f"  📊 Data tab: {tab_name} ({len(tab_info['data'])} rows)")
                    
                    else:
                        # Single file or single-tab Excel
                        if data_result is not None:
                            datasets.append({
                                'file_path': file_path,
                                'data': data_result,
                                'metadata': data_ingestion.analyze_structure(data_result),
                                'is_multi_tab': False
                            })
                            click.echo(f"✅ Loaded data file: {os.path.basename(file_path)} ({len(data_result)} rows)")
                    
                except Exception as e:
                    click.echo(f"⚠️ Failed to load {file_path}: {str(e)}")
                    logger.error(f"Error loading {file_path}: {str(e)}")
            
            elif file_ext in ['.xsd', '.xml', '.sql', '.ddl']:
                # Process schema files
                try:
                    schema_info = schema_analyzer.parse_schema_file(file_path)
                    if schema_info:
                        schema_analyses.append({
                            'file_path': file_path,
                            'schema_info': schema_info
                        })
                        click.echo(f"✅ Analyzed schema: {os.path.basename(file_path)}")
                except Exception as e:
                    click.echo(f"⚠️ Failed to analyze schema {file_path}: {str(e)}")
                    logger.error(f"Error analyzing schema {file_path}: {str(e)}")
        
        if not datasets and not schema_analyses:
            click.echo("❌ No valid files could be processed")
            return
        
        # Combine schema information for AI analysis
        combined_schema_info = None
        if schema_analyses:
            combined_schema_info = {
                'schemas': [s['schema_info'] for s in schema_analyses],
                'source_files': [s['file_path'] for s in schema_analyses]
            }
        
        # Perform AI-powered analysis
        logger.info("Performing AI-powered analysis")
        ai_insights = ai_analyzer.analyze_for_process_mining(
            datasets=datasets,
            schema_info=combined_schema_info,
            business_context=business_context
        )
        
        # Generate event log assessment
        logger.info("Generating event log assessment")
        assessment = event_log_analyzer.generate_assessment(
            datasets=datasets,
            schema_info=combined_schema_info,
            ai_insights=ai_insights,
            business_context=business_context
        )
        
        # Initialize output manager for organized file handling
        output_manager = OutputManager(base_dir=output_dir)
        
        # Collect all source files for contextual naming
        all_source_files = files_to_process.copy()
        
        # Save results using intelligent output management
        if output:
            # Legacy output option - save to specified path
            with open(output, 'w', encoding='utf-8') as f:
                yaml.safe_dump(assessment, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            output_path = Path(output)
            logger.info(f"Assessment results saved to {output} (legacy mode)")
        else:
            # New organized output management
            output_path = output_manager.save_assessment(
                assessment_data=assessment,
                data_sources=all_source_files,
                custom_name=output_name,
                output_format=output_format,
                keep_history=keep_history
            )
            click.echo(f"📁 Results saved to: {output_path}")
            click.echo(f"📂 Latest available at: {output_manager.base_dir}/assessments/latest/")
        
        # Display summary
        display_assessment_summary(assessment)
        if assessment.get('suggested_sql'):
            click.echo("\n🧩 Suggested SQL generated (see output file).")
        
    except Exception as e:
        logger.error(f"Error during assessment: {e}")
        sys.exit(1)


@cli.command()
def interactive() -> None:
    """Run in interactive mode for guided analysis."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting interactive mode")
    
    click.echo("\n🔍 Process Mining Event Log Assessment Assistant")
    click.echo("=" * 55)
    
    # Collect user inputs interactively
    data_files = []
    while True:
        file_path = click.prompt(
            "\nEnter path to data file (or 'done' to finish)",
            type=str
        )
        if file_path.lower() == 'done':
            break
        if Path(file_path).exists():
            data_files.append(file_path)
            click.echo(f"✓ Added {file_path}")
        else:
            click.echo(f"❌ File not found: {file_path}")
    
    # Get business context
    context_text = click.prompt(
        "\nDescribe the business process and what you know about the data",
        type=str
    )
    
    # Get output preferences
    output_file = click.prompt(
        "\nOutput file name",
        default="interactive_assessment.yaml",
        type=str
    )
    
    # Create temporary context file
    context_file = "temp_context.txt"
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write(context_text)
    
    # Run assessment
    try:
        assess.callback(
            data_files=tuple(data_files),
            schema=None,
            schema_files=(),
            directory=None,
            context=context_file,
            output=output_file
        )
    finally:
        # Clean up temporary file
        if Path(context_file).exists():
            Path(context_file).unlink()


@cli.command()
def demo() -> None:
    """Run a demonstration of the tool's capabilities."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting demo mode")
    
    click.echo("\n🚀 Process Mining Event Log Assessment Assistant - Demo")
    click.echo("=" * 60)
    
    # Create sample data for demonstration
    import pandas as pd
    
    # Sample order processing data
    sample_data = pd.DataFrame({
        'order_id': [1001, 1001, 1001, 1002, 1002, 1003, 1003, 1003],
        'timestamp': [
            '2024-01-15 09:00:00',
            '2024-01-15 10:30:00',
            '2024-01-15 14:45:00',
            '2024-01-16 11:15:00',
            '2024-01-16 15:20:00',
            '2024-01-17 08:30:00',
            '2024-01-17 12:00:00',
            '2024-01-17 16:30:00'
        ],
        'activity': [
            'Order Created',
            'Payment Processed',
            'Order Shipped',
            'Order Created',
            'Order Shipped',
            'Order Created',
            'Payment Processed',
            'Order Delivered'
        ],
        'user_id': ['user123', 'system', 'warehouse1', 'user456', 'warehouse2', 'user789', 'system', 'courier1'],
        'amount': [150.00, 150.00, 150.00, 75.50, 75.50, 299.99, 299.99, 299.99],
        'region': ['North', 'North', 'North', 'South', 'South', 'East', 'East', 'East']
    })
    
    click.echo("\n📊 Sample Data Preview:")
    click.echo(sample_data.to_string(index=False))
    
    # Demonstrate key insights
    click.echo("\n🔍 Key Process Mining Insights:")
    click.echo("✓ Potential Case ID: order_id")
    click.echo("✓ Activity Column: activity")
    click.echo("✓ Timestamp Column: timestamp")
    click.echo("✓ Case Attributes: amount, region")
    click.echo("✓ Event Attributes: user_id")
    
    click.echo("\n💡 Recommendations:")
    click.echo("• Convert timestamp to datetime format")
    click.echo("• Validate case completeness (some orders missing delivery)")
    click.echo("• Consider user_id as resource/actor dimension")
    click.echo("• Group similar activities if needed")
    
    click.echo("\n✨ Demo completed! To analyze your own data:")
    click.echo("  python -m src.main assess --data-files your_data.csv --context description.txt")


def display_assessment_summary(assessment: dict) -> None:
    """Simple MVP summary - just the essentials."""
    
    click.echo("\n" + "=" * 50)
    click.echo("� PROCESS MINING ASSESSMENT")
    click.echo("=" * 50)
    
    # Event-centric display - check both ai_insights and direct assessment
    event_candidates = assessment.get('event_candidates', [])
    
    # Also check in ai_insights section
    ai_insights = assessment.get('ai_insights', {})
    if not event_candidates and ai_insights:
        event_candidates = ai_insights.get('event_candidates', [])
    
    if event_candidates:
        # Show best event structure
        best_event = event_candidates[0]
        confidence = int(best_event.get('confidence', 0) * 100)
        click.echo(f"✅ Complete Events Found: {len(event_candidates)} structure(s)")
        click.echo(f"   Best Event: {best_event.get('case_id_column')} + {best_event.get('activity_column')} + {best_event.get('timestamp_column')}")
        click.echo(f"   Confidence: {confidence}%")
        click.echo(f"   Description: {best_event.get('event_description', 'Event structure detected')}")
    else:
        click.echo("❌ Complete Events: Not found")
        click.echo("   Need: Case ID + Activity + Timestamp together")
    
    # Simple readiness
    readiness = assessment.get('readiness', {})
    if readiness.get('ready'):
        click.echo("\n� Status: Ready for event log creation")
    else:
        missing = readiness.get('missing_elements', [])
        click.echo(f"\n⚠️  Status: Missing {', '.join(missing)}")
    
    # Critical issues only
    if assessment.get('data_issues'):
        click.echo(f"\n� Data Issues: {len(assessment['data_issues'])} found")
    
    click.echo("\n� SQL query generated in output file.")
    click.echo("=" * 50)


@cli.command()
@click.option(
    '--output-dir',
    type=click.Path(),
    default='results',
    help='Base directory to manage'
)
@click.option(
    '--cleanup-days',
    type=int,
    default=30,
    help='Number of days to keep old files (0 = no cleanup)'
)
def manage_outputs(output_dir: str, cleanup_days: int) -> None:
    """Manage and organize output files."""
    
    output_manager = OutputManager(base_dir=output_dir)
    
    click.echo(f"\n📂 Output Management for: {output_dir}")
    click.echo("=" * 50)
    
    # Get summary
    summary = output_manager.get_results_summary()
    
    click.echo(f"📊 Total files: {summary['total_files']}")
    click.echo(f"📁 Base directory: {summary['base_directory']}")
    
    if summary['by_type']:
        click.echo("\n📂 Files by type:")
        for file_type, count in summary['by_type'].items():
            click.echo(f"  {file_type}: {count} files")
    
    if summary['latest_files']:
        click.echo("\n🔗 Latest files:")
        for file_type, path in summary['latest_files'].items():
            click.echo(f"  {file_type}: {path}")
    
    # Cleanup if requested
    if cleanup_days > 0:
        click.echo(f"\n🧹 Cleaning up files older than {cleanup_days} days...")
        cleaned_count = output_manager.cleanup_old_files(cleanup_days)
        click.echo(f"✅ Cleaned up {cleaned_count} old files")
    
    click.echo("\n💡 Tips:")
    click.echo(f"  • Latest results: {output_dir}/assessments/latest/")
    click.echo(f"  • Archived files: {output_dir}/archives/")
    click.echo("  • Use --output-name for custom filenames")
    click.echo("  • Use --keep-history to preserve previous results")


@cli.command()
@click.option(
    '--output-dir',
    type=click.Path(),
    default='results',
    help='Base directory to organize'
)
def organize_legacy_files(output_dir: str) -> None:
    """Organize existing messy output files into the new structure."""
    
    click.echo("\n🔄 Organizing Legacy Output Files")
    click.echo("=" * 50)
    
    output_manager = OutputManager(base_dir=output_dir)
    root_dir = Path(".")
    
    # Find assessment files in root directory
    legacy_files = list(root_dir.glob("*assessment*.yaml")) + list(root_dir.glob("*_results.yaml"))
    
    if not legacy_files:
        click.echo("✅ No legacy assessment files found in root directory")
        return
    
    click.echo(f"📋 Found {len(legacy_files)} legacy files to organize:")
    
    organized_count = 0
    for file_path in legacy_files:
        if file_path.is_file():
            click.echo(f"  📄 {file_path.name}")
            
            # Move to organized structure
            new_path = output_manager.base_dir / "assessments" / "legacy" / file_path.name
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(file_path), str(new_path))
            organized_count += 1
    
    click.echo(f"\n✅ Organized {organized_count} files into: {output_dir}/assessments/legacy/")
    click.echo("💡 Future assessments will be automatically organized by date")


def main() -> None:
    """Entry point for the application."""
    cli()


if __name__ == "__main__":
    main()
