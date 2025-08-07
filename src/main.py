#!/usr/bin/env python3
"""Main entry point for the Process Mining Event Log Assessment Assistant."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
import yaml

from .core.data_ingestion import DataIngestionEngine
from .core.schema_analyzer import SchemaAnalyzer
from .core.ai_analyzer import AIAnalyzer
from .core.event_log_analyzer import EventLogAnalyzer
from .utils.helpers import setup_logging, load_environment


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
        
        # Initialize components
        data_ingestion = DataIngestionEngine()
        schema_analyzer = SchemaAnalyzer()
        ai_analyzer = AIAnalyzer()
        event_log_analyzer = EventLogAnalyzer()
        
        # Load and analyze data files
        datasets = []
        for file_path in data_files:
            logger.info(f"Loading data from {file_path}")
            data = data_ingestion.load_file(file_path)
            datasets.append({
                'file_path': file_path,
                'data': data,
                'metadata': data_ingestion.analyze_structure(data)
            })
        
        # Analyze schema if provided
        schema_info = None
        if schema:
            logger.info(f"Analyzing schema from {schema}")
            schema_info = schema_analyzer.parse_schema_file(schema)
        
        # Perform AI-powered analysis
        logger.info("Performing AI-powered analysis")
        ai_insights = ai_analyzer.analyze_for_process_mining(
            datasets=datasets,
            schema_info=schema_info,
            business_context=business_context
        )
        
        # Generate event log assessment
        logger.info("Generating event log assessment")
        assessment = event_log_analyzer.generate_assessment(
            datasets=datasets,
            schema_info=schema_info,
            ai_insights=ai_insights,
            business_context=business_context
        )
        
        # Save results
        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(assessment, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Assessment results saved to {output}")
        
        # Display summary
        display_assessment_summary(assessment)
        
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
    """Display a summary of the assessment results."""
    
    click.echo("\n" + "=" * 60)
    click.echo("📋 ASSESSMENT SUMMARY")
    click.echo("=" * 60)
    
    # Case ID recommendations
    if 'case_id_candidates' in assessment:
        click.echo("\n🆔 Case ID Candidates:")
        for candidate in assessment['case_id_candidates'][:3]:  # Top 3
            confidence = candidate.get('confidence', 0)
            click.echo(f"  • {candidate['column']} (confidence: {confidence:.1%})")
    
    # Activity recommendations
    if 'activity_analysis' in assessment:
        click.echo("\n⚡ Activity Analysis:")
        activities = assessment['activity_analysis']
        if 'recommended_activities' in activities:
            click.echo(f"  • Recommended activities: {len(activities['recommended_activities'])}")
        if 'activities_to_aggregate' in activities:
            click.echo(f"  • Activities to consider aggregating: {len(activities['activities_to_aggregate'])}")
    
    # Data quality summary
    if 'data_quality' in assessment:
        quality = assessment['data_quality']
        overall_score = quality.get('overall_score', 0)
        click.echo(f"\n📊 Data Quality Score: {overall_score:.1%}")
    
    # Key recommendations
    if 'recommendations' in assessment:
        click.echo("\n💡 Key Recommendations:")
        for i, rec in enumerate(assessment['recommendations'][:5], 1):
            click.echo(f"  {i}. {rec}")
    
    click.echo("\n" + "=" * 60)


def main() -> None:
    """Entry point for the application."""
    cli()


if __name__ == "__main__":
    main()
