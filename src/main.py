#!/usr/bin/env python3
"""Main entry point for the Process Mining Event Log Assessment Assistant."""

import glob
import logging
import os
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
    
    A tool for helping process mining consultants assess and prepare data
    from various source systems to create high-quality event logs.
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
    default='assessment_results.yaml',
    help='Output file for assessment results'
)
def assess(
    data_files: tuple,
    schema: Optional[str],
    schema_files: tuple,
    directory: Optional[str],
    context: Optional[str],
    output: str
) -> None:
    """Assess data sources for process mining event log creation."""
    
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
                    data = data_ingestion.load_file(file_path)
                    if data is not None:
                        datasets.append({
                            'file_path': file_path,
                            'data': data,
                            'metadata': data_ingestion.analyze_structure(data)
                        })
                        click.echo(f"✅ Loaded data file: {os.path.basename(file_path)} ({len(data)} rows)")
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
        
        # Save results
        with open(output, 'w', encoding='utf-8') as f:
            yaml.safe_dump(assessment, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        logger.info(f"Assessment results saved to {output}")
        
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
    
    # Show top candidates only
    if assessment.get('case_id_candidates'):
        top_case = assessment['case_id_candidates'][0]
        col = top_case.get('column') or top_case.get('name')
        click.echo(f"✅ Case ID: {col}")
    else:
        click.echo("❌ Case ID: Not found")
    
    if assessment.get('activity_candidates'):
        top_activity = assessment['activity_candidates'][0]
        col = top_activity.get('column') or top_activity.get('name')
        click.echo(f"✅ Activity: {col}")
    else:
        click.echo("❌ Activity: Not found")
    
    if assessment.get('timestamp_candidates'):
        top_timestamp = assessment['timestamp_candidates'][0]
        col = top_timestamp.get('column') or top_timestamp.get('name')
        click.echo(f"✅ Timestamp: {col}")
    else:
        click.echo("❌ Timestamp: Not found")
    
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


def main() -> None:
    """Entry point for the application."""
    cli()


if __name__ == "__main__":
    main()
