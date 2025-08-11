"""Event log analyzer for generating comprehensive process mining assessments."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for YAML serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


class EventLogAnalyzer:
    """Streamlined analyzer for identifying process mining elements in data schemas.
    
    Core focus:
    - Identify potential UIDs (Case IDs) 
    - Find activities/events with timestamps
    - Classify attributes for case/event context
    - Generate SQL for event log extraction
    """
    
    def __init__(self):
        """Initialize the EventLogAnalyzer."""
        logger.info("EventLogAnalyzer initialized")
    
    def generate_assessment(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        ai_insights: Dict[str, Any],
        business_context: str
    ) -> Dict[str, Any]:
        """Generate event-centric process mining assessment with executive summary.
        
        Args:
            datasets: List of dataset information
            schema_info: Parsed schema information
            ai_insights: AI-generated insights
            business_context: Business context description
            
        Returns:
            Event-centric assessment with executive summary and technical details
        """
        logger.info("Generating business-focused process mining assessment")
        
        # Extract event candidates from AI insights or generate fallback
        if ai_insights and 'event_candidates' in ai_insights:
            event_candidates = ai_insights['event_candidates']
            business_analysis = ai_insights.get('business_process_analysis', {})
            readiness_scores = ai_insights.get('readiness_assessment', {})
            recommendations = ai_insights.get('actionable_recommendations', [])
            sql_code = ai_insights.get('event_log_sql', self._generate_event_sql_fallback(datasets))
            business_value = ai_insights.get('business_value', 'Process mining analysis will provide insights into process efficiency and improvement opportunities')
            data_concerns = ai_insights.get('data_quality_concerns', [])
            mining_potential = ai_insights.get('process_mining_potential', 'Assessment pending')
        else:
            # Enhanced fallback with event-centric business intelligence
            fallback = self._generate_enhanced_business_fallback(datasets)
            event_candidates = fallback.get('event_candidates', [])
            business_analysis = fallback['business_process_analysis']
            readiness_scores = fallback['readiness_score']
            recommendations = fallback['actionable_recommendations']
            sql_code = fallback['working_sql']
            business_value = fallback['business_value']
            data_concerns = fallback['data_quality_concerns']
            mining_potential = fallback['process_mining_potential']
        
        # Legacy technical analysis for backward compatibility
        case_id_candidates = self._compile_case_id_candidates(datasets, schema_info, ai_insights)
        activity_analysis = self._compile_activity_analysis(datasets, schema_info, ai_insights)
        timestamp_analysis = self._compile_timestamp_analysis(datasets, schema_info, ai_insights)
        case_attributes, event_attributes = self._compile_attributes(datasets, case_id_candidates, 
                                                                   activity_analysis.get('activity_candidates', []), 
                                                                   timestamp_analysis.get('timestamp_candidates', []),
                                                                   schema_info)
        
        # Generate readiness assessment
        readiness = self._assess_readiness(datasets, ai_insights)
        data_quality = self._assess_data_quality(datasets)
        
        # Build event-centric assessment with executive summary
        assessment = {
            'executive_summary': {
                'process_type': business_analysis.get('process_type', 'Unknown Business Process'),
                'confidence': f"{business_analysis.get('confidence', 0.1):.0%}",
                'readiness_score': f"{readiness_scores.get('overall_score', readiness.get('overall_score', 0))}/10",
                'mining_potential': mining_potential,
                'key_finding': self._get_event_key_finding(event_candidates, readiness_scores),
                'primary_recommendation': recommendations[0] if recommendations else "Manual data review required",
                'timeline_estimate': self._estimate_timeline(readiness_scores),
                'business_value': business_value
            },
            
            'business_assessment': {
                'process_identification': {
                    'identified_process': business_analysis.get('process_type', 'Unknown'),
                    'confidence_level': business_analysis.get('confidence', 0.1),
                    'reasoning': business_analysis.get('reasoning', 'Limited data for process identification')
                },
                'event_readiness': {
                    'overall_score': readiness_scores.get('overall_score', readiness.get('overall_score', 0)),
                    'event_completeness': readiness_scores.get('event_completeness', 0),
                    'temporal_coverage': readiness_scores.get('temporal_coverage', 0),
                    'case_id_quality': readiness_scores.get('case_id_quality', 0),
                    'breakdown_explanation': readiness_scores.get('reasoning', 'Assessment based on event structure analysis')
                },
                'immediate_actions': recommendations[:3] if recommendations else [
                    "Identify complete event structures (activity + timestamp pairs)",
                    "Validate event business meaning with stakeholders", 
                    "Ensure temporal data quality for process flow analysis"
                ],
                'data_issues_summary': data_concerns if data_concerns else data_quality.get('issues', []),
                'next_steps': self._generate_event_next_steps(event_candidates, readiness_scores)
            },
            
            'technical_details': {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'files_analyzed': [d.get('file_path') for d in datasets],
                    'analysis_method': 'event-centric'
                },
                'event_candidates': event_candidates[:5],  # Top 5 complete event structures
                'case_attributes': case_attributes[:10],  # Top 10 case attributes
                'event_attributes': event_attributes[:10],  # Top 10 event attributes
                'suggested_sql': sql_code,
                'readiness': {
                    'ready': readiness.get('ready', False),
                    'missing_elements': readiness.get('missing_elements', [])
                },
                'data_issues': data_quality.get('issues', [])
            },
            
            'ai_insights': ai_insights if ai_insights else []
        }
        
        # Convert numpy types for YAML serialization
        return self._convert_numpy_types(assessment)

    def _get_event_key_finding(self, event_candidates: List[Dict[str, Any]], readiness_scores: Dict[str, Any]) -> str:
        """Generate key finding based on event analysis."""
        if not event_candidates:
            return "Critical Issue: No complete events (activity+timestamp pairs) found - process mining requires temporal information"
        
        best_event = event_candidates[0]
        confidence = best_event.get('confidence', 0)
        
        if confidence > 0.8:
            return f"Strong event structure found: {best_event.get('event_description', 'events detected')}"
        elif confidence > 0.5:
            return f"Viable event structure identified with {confidence:.0%} confidence - some improvements needed"
        else:
            return "Weak event structure detected - significant data improvements required for process mining"

    def _generate_event_next_steps(self, event_candidates: List[Dict[str, Any]], readiness_scores: Dict[str, Any]) -> List[str]:
        """Generate next steps based on event analysis."""
        steps = []
        
        if not event_candidates:
            steps.extend([
                "Identify data sources with both activities and timestamps",
                "Request event logging enhancement from system administrators",
                "Consider process instrumentation to capture temporal data"
            ])
        else:
            best_event = event_candidates[0]
            steps.append(f"Validate {best_event['activity_column']} represents meaningful business activities")
            steps.append(f"Confirm {best_event['timestamp_column']} captures accurate event timing")
            steps.append("Review sample events with business stakeholders for accuracy")
        
        steps.extend([
            "Plan proof-of-concept process mining analysis",
            "Define success metrics and business value targets"
        ])
        
        return steps[:5]

    def _generate_event_sql_fallback(self, datasets: List[Dict[str, Any]]) -> str:
        """Generate SQL for event log when no AI insights available."""
        if not datasets:
            return "-- No datasets available for event log SQL generation"
        
        # This would be enhanced with actual event detection logic
        # For now, fall back to existing SQL generation
        return self._generate_working_sql(datasets)

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types for YAML serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _generate_enhanced_business_fallback(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate enhanced fallback analysis with business intelligence."""
        logger.info("Generating enhanced business fallback analysis")
        
        # Infer business process type
        process_type = self._infer_process_type_from_data(datasets)
        
        # Calculate readiness scores
        readiness_scores = self._calculate_detailed_readiness(datasets)
        
        # Generate business recommendations
        recommendations = self._generate_business_recommendations_fallback(datasets, process_type)
        
        # Generate working SQL
        working_sql = self._generate_working_sql(datasets)
        
        return {
            'business_process_analysis': {
                'process_type': process_type['name'],
                'confidence': process_type['confidence'],
                'reasoning': process_type['reasoning']
            },
            'readiness_score': readiness_scores,
            'actionable_recommendations': recommendations,
            'working_sql': working_sql,
            'business_value': f"Process mining this {process_type['name'].lower()} could reveal efficiency improvements and process optimization opportunities",
            'next_steps': [
                "Validate process identification with business stakeholders",
                "Review data quality issues identified",
                "Plan proof-of-concept process mining analysis"
            ],
            'data_quality_concerns': self._identify_data_quality_concerns(datasets),
            'process_mining_potential': self._assess_mining_potential_fallback(readiness_scores)
        }

    def _infer_process_type_from_data(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer business process type from data patterns."""
        
        # Process indicators based on common business processes
        process_indicators = {
            'Incident Management': ['incident', 'ticket', 'event', 'status', 'priority', 'assigned', 'mis', 'service', 'issue'],
            'Order Management': ['order', 'customer', 'product', 'quantity', 'shipment', 'delivery', 'invoice'],
            'Procurement': ['purchase', 'vendor', 'supplier', 'contract', 'approval', 'procurement', 'buying'],
            'HR Process': ['employee', 'hiring', 'performance', 'leave', 'payroll', 'training', 'staff'],
            'Finance Process': ['payment', 'invoice', 'account', 'transaction', 'budget', 'billing', 'financial'],
            'Manufacturing': ['production', 'machine', 'quality', 'batch', 'manufacturing', 'assembly'],
            'Logistics': ['shipment', 'warehouse', 'transport', 'delivery', 'logistics', 'supply']
        }
        
        scores = {}
        evidence = {}
        
        for process_name, keywords in process_indicators.items():
            score = 0
            found_evidence = []
            
            for dataset in datasets:
                # Check file path
                file_path = dataset.get('file_path', '').lower()
                for keyword in keywords:
                    if keyword in file_path:
                        score += 3  # File name evidence is weighted higher
                        found_evidence.append(f"filename: '{keyword}'")
                
                # Check column names
                if dataset.get('data') is not None:
                    columns = [str(col).lower() for col in dataset['data'].columns]
                    for keyword in keywords:
                        matching_cols = [col for col in columns if keyword in col]
                        score += len(matching_cols)
                        found_evidence.extend([f"column: '{col}'" for col in matching_cols])
            
            scores[process_name] = score
            evidence[process_name] = found_evidence[:5]  # Keep top 5 pieces of evidence
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            return {
                'name': 'Unknown Business Process',
                'confidence': 0.1,
                'reasoning': 'Insufficient data patterns to identify specific business process type'
            }
        
        best_process = max(scores, key=scores.get)
        best_score = scores[best_process]
        confidence = min(best_score / 10.0, 1.0)  # Normalize to 0-1
        
        evidence_text = ', '.join(evidence[best_process]) if evidence[best_process] else 'general data patterns'
        
        return {
            'name': best_process,
            'confidence': confidence,
            'reasoning': f"Identified {best_score} indicators for {best_process.lower()}. Evidence: {evidence_text}"
        }

    def _calculate_detailed_readiness(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed readiness scores for process mining."""
        
        if not datasets:
            return {
                'overall_score': 0,
                'case_id_quality': 0,
                'activity_quality': 0,
                'timestamp_quality': 0,
                'data_completeness': 0,
                'reasoning': 'No datasets available for analysis'
            }
        
        case_id_score = 0
        activity_score = 0
        timestamp_score = 0
        completeness_scores = []
        
        for dataset in datasets:
            if dataset.get('data') is None:
                continue
            
            columns = [str(col).lower() for col in dataset['data'].columns]
            
            # Case ID assessment
            case_id_indicators = ['id', 'number', 'key', 'reference', 'event id', 'case', 'incident']
            case_id_matches = sum(1 for col in columns for indicator in case_id_indicators if indicator in col)
            case_id_score += min(case_id_matches, 3)  # Cap at 3 points per dataset
            
            # Activity assessment
            activity_indicators = ['status', 'state', 'activity', 'event', 'action', 'step', 'stage']
            activity_matches = sum(1 for col in columns for indicator in activity_indicators if indicator in col)
            activity_score += min(activity_matches, 3)
            
            # Timestamp assessment
            timestamp_indicators = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 'when']
            timestamp_matches = sum(1 for col in columns for indicator in timestamp_indicators if indicator in col)
            timestamp_score += min(timestamp_matches, 3)
            
            # Data completeness
            if hasattr(dataset['data'], 'isnull'):
                total_cells = dataset['data'].shape[0] * dataset['data'].shape[1]
                if total_cells > 0:
                    missing_cells = dataset['data'].isnull().sum().sum()
                    completeness = ((total_cells - missing_cells) / total_cells) * 100
                    completeness_scores.append(completeness)
        
        # Normalize scores to 0-10 scale
        num_datasets = len([d for d in datasets if d.get('data') is not None])
        
        if num_datasets > 0:
            case_id_final = min((case_id_score / num_datasets) * 3.33, 10)  # Scale to 10
            activity_final = min((activity_score / num_datasets) * 3.33, 10)
            timestamp_final = min((timestamp_score / num_datasets) * 3.33, 10)
            completeness_final = sum(completeness_scores) / len(completeness_scores) / 10 if completeness_scores else 0
        else:
            case_id_final = activity_final = timestamp_final = completeness_final = 0
        
        overall = (case_id_final + activity_final + timestamp_final + completeness_final) / 4
        
        return {
            'overall_score': round(overall, 1),
            'case_id_quality': round(case_id_final, 1),
            'activity_quality': round(activity_final, 1),
            'timestamp_quality': round(timestamp_final, 1),
            'data_completeness': round(completeness_final, 1),
            'reasoning': f"Assessed {num_datasets} datasets for process mining readiness"
        }

    def _generate_business_recommendations_fallback(self, datasets: List[Dict[str, Any]], process_type: Dict[str, Any]) -> List[str]:
        """Generate business-focused recommendations."""
        recommendations = []
        
        # Process-specific recommendations
        process_name = process_type['name'].lower()
        if 'incident' in process_name:
            recommendations.extend([
                "Focus on incident resolution lifecycle and SLA compliance",
                "Identify bottlenecks in assignment and escalation processes"
            ])
        elif 'order' in process_name:
            recommendations.extend([
                "Analyze order fulfillment cycle time and delivery performance",
                "Identify opportunities to streamline order processing"
            ])
        elif 'procurement' in process_name:
            recommendations.extend([
                "Examine purchase approval workflows and vendor performance",
                "Optimize procurement cycle time and cost efficiency"
            ])
        
        # Data quality recommendations
        has_timestamps = any('date' in str(col).lower() or 'time' in str(col).lower()
                           for dataset in datasets if dataset.get('data') is not None
                           for col in dataset['data'].columns)
        
        if not has_timestamps:
            recommendations.insert(0, "CRITICAL: Obtain timestamp data to enable process flow analysis")
        
        # General recommendations
        recommendations.extend([
            "Validate case ID definitions with business stakeholders",
            "Confirm activity names represent meaningful business events",
            "Plan proof-of-concept process mining analysis with limited scope"
        ])
        
        return recommendations[:8]

    def _identify_data_quality_concerns(self, datasets: List[Dict[str, Any]]) -> List[str]:
        """Identify specific data quality issues."""
        concerns = []
        
        for dataset in datasets:
            if dataset.get('data') is None:
                continue
                
            file_name = self._get_clean_filename(dataset.get('file_path', 'dataset'))
            
            # Check for high missing data
            if hasattr(dataset['data'], 'isnull'):
                total_cells = dataset['data'].shape[0] * dataset['data'].shape[1]
                if total_cells > 0:
                    missing_cells = dataset['data'].isnull().sum().sum()
                    missing_pct = (missing_cells / total_cells) * 100
                    if missing_pct > 30:
                        concerns.append(f"High missing data in {file_name}: {missing_pct:.1f}%")
            
            # Check for unnamed/problematic columns
            columns = list(dataset['data'].columns)
            unnamed_cols = [col for col in columns if 'unnamed' in str(col).lower()]
            if unnamed_cols:
                concerns.append(f"Unnamed columns in {file_name}: {len(unnamed_cols)} columns need proper names")
        
        return concerns

    def _assess_mining_potential_fallback(self, readiness_scores: Dict[str, Any]) -> str:
        """Assess overall process mining potential."""
        score = readiness_scores.get('overall_score', 0)
        
        if score >= 7:
            return "High - Good foundation for immediate process mining analysis"
        elif score >= 4:
            return "Medium - Requires data improvements but viable with effort"
        else:
            return "Low - Significant data quality issues need resolution first"

    def _get_key_finding(self, datasets: List[Dict[str, Any]], readiness_scores: Dict[str, Any], case_id_candidates: List[Dict], timestamp_analysis: Dict[str, Any]) -> str:
        """Generate key finding summary."""
        
        # Check critical elements
        has_case_ids = len(case_id_candidates) > 0
        has_timestamps = len(timestamp_analysis.get('timestamp_candidates', [])) > 0
        overall_score = readiness_scores.get('overall_score', 0)
        
        if not has_timestamps:
            return "Critical Issue: No timestamp data found - process mining requires temporal information"
        elif not has_case_ids:
            return "Major Issue: No clear case identifiers found - need to define process instances"
        elif overall_score >= 7:
            return "Good foundation for process mining - data structure supports analysis"
        elif overall_score >= 4:
            return "Moderate readiness - data improvements needed but analysis is feasible"
        else:
            return "Significant data quality issues require resolution before process mining"

    def _estimate_timeline(self, readiness_scores: Dict[str, Any]) -> str:
        """Estimate timeline for process mining readiness."""
        score = readiness_scores.get('overall_score', 0)
        
        if score >= 8:
            return "Ready now - can begin analysis immediately"
        elif score >= 6:
            return "1-2 weeks - minor data preparation needed"
        elif score >= 4:
            return "2-4 weeks - moderate data quality improvements required"
        else:
            return "4+ weeks - significant data preparation and quality work needed"

    def _calculate_data_completeness(self, datasets: List[Dict[str, Any]]) -> float:
        """Calculate overall data completeness percentage."""
        if not datasets:
            return 0.0
        
        completeness_scores = []
        
        for dataset in datasets:
            if dataset.get('data') is not None and hasattr(dataset['data'], 'isnull'):
                total_cells = dataset['data'].shape[0] * dataset['data'].shape[1]
                if total_cells > 0:
                    missing_cells = dataset['data'].isnull().sum().sum()
                    completeness = ((total_cells - missing_cells) / total_cells) * 100
                    completeness_scores.append(completeness)
        
        return round(sum(completeness_scores) / len(completeness_scores), 1) if completeness_scores else 0.0

    def _get_clean_filename(self, file_path: str) -> str:
        """Extract clean filename from path."""
        if '#' in file_path:
            # Handle Excel tab references
            return file_path.split('/')[-1] if '/' in file_path else file_path
        else:
            return file_path.split('/')[-1] if '/' in file_path else file_path
    

    
    def _analyze_schema(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema information for process mining relevance."""
        
        analysis = {
            'schema_type': 'multiple' if (isinstance(schema_info, dict) and 'schemas' in schema_info) else (schema_info.get('type') if isinstance(schema_info, dict) else 'unknown'),
            'summary': {},
            'process_mining_elements': {}
        }
        
        if not isinstance(schema_info, dict):
            return analysis
        
        # Handle multiple schemas (robust to dict/list element shapes)
        if 'schemas' in schema_info:
            schemas = schema_info.get('schemas', [])
            source_files = schema_info.get('source_files', [])
            
            total_elements = 0
            schema_types = set()
            all_elements = []
            
            for i, schema in enumerate(schemas):
                schema_type = schema.get('type', 'unknown')
                schema_types.add(schema_type)
                src_file = source_files[i] if i < len(source_files) else 'unknown'
                
                if schema_type == 'xml':
                    elements = schema.get('elements', {})
                    # elements may be dict{name:info} or list[{name,type,...}]
                    if isinstance(elements, dict):
                        items = [{'name': k, **(v or {})} for k, v in elements.items()]
                    elif isinstance(elements, list):
                        items = elements
                    else:
                        items = []
                    total_elements += len(items)
                    for el in items:
                        all_elements.append({
                            'name': el.get('name'),
                            'source_file': src_file,
                            'type': el.get('type'),
                            'attributes': el.get('attributes', [])
                        })
                elif schema_type == 'sql':
                    tables = schema.get('tables', {})
                    if isinstance(tables, dict):
                        total_elements += sum(len(t.get('columns', {})) for t in tables.values())
            
            analysis['summary'] = {
                'total_schemas': len(schemas),
                'schema_types': list(schema_types),
                'total_elements': total_elements,
                'source_files': [os.path.basename(f) for f in source_files]
            }
            
            # Analyze elements for process mining potential
            process_elements = self._analyze_elements_for_process_mining(all_elements)
            analysis['process_mining_elements'] = process_elements
            return analysis
        
        # Handle single schema (legacy format)
        if schema_info.get('type') == 'sql':
            tables = schema_info.get('tables', {})
            analysis['summary'] = {
                'total_tables': len(tables) if isinstance(tables, dict) else 0,
                'tables_with_relationships': len([t for t in (tables.values() if isinstance(tables, dict) else []) if t.get('foreign_keys')]),
                'total_columns': sum(len(t.get('columns', {})) for t in (tables.values() if isinstance(tables, dict) else []))
            }
            
            # Get process mining candidates from schema analyzer
            if 'process_mining_candidates' in schema_info:
                analysis['process_mining_elements'] = schema_info['process_mining_candidates']
        
        return analysis
    
    def _compile_case_id_candidates(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        ai_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compile potential UID candidates from schema, AI analysis, and data analysis."""
        
        candidates = []
        
        # PRIORITY 1: Schema analysis results (PRIMARY SOURCE)
        if schema_info:
            schema_analysis = self._analyze_schema(schema_info)
            schema_elements = schema_analysis.get('process_mining_elements', {})
            
            # Extract case ID candidates from schema
            for candidate in schema_elements.get('case_id_candidates', []):
                candidates.append({
                    'table': candidate['source_file'],
                    'column': candidate['name'],
                    'confidence': candidate['confidence'],
                    'reasoning': candidate['reason'],
                    'source': 'schema_analysis'
                })
        
        # PRIORITY 2: AI analysis (if available)
        if ai_insights.get('ai_analysis') and 'case_id_analysis' in ai_insights:
            case_analysis = ai_insights['case_id_analysis']
            
            if case_analysis.get('primary_recommendation'):
                primary = case_analysis['primary_recommendation']
                primary['source'] = 'ai_primary'
                candidates.append(primary)
            
            for alt in case_analysis.get('alternatives', []):
                alt['source'] = 'ai_alternative'
                candidates.append(alt)
        
        # PRIORITY 3: Data analysis - look for potential identifiers (validation)
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            for col in metadata.get('potential_identifiers', []):
                col_info = metadata.get('columns', {}).get(col, {})
                candidates.append({
                    'table': dataset['file_path'],
                    'column': col,
                    'confidence': self._calculate_case_id_confidence(col, col_info),
                    'reasoning': 'Data structure analysis - potential unique identifier',
                    'source': 'data_analysis'
                })
        
        # Remove duplicates and sort by confidence
        unique_candidates = {}
        for candidate in candidates:
            key = f"{candidate['table']}::{candidate['column']}"
            if key not in unique_candidates or candidate['confidence'] > unique_candidates[key]['confidence']:
                unique_candidates[key] = candidate
        
        return sorted(unique_candidates.values(), key=lambda x: x['confidence'], reverse=True)[:25]  # Top 25
    
    def _compile_activity_analysis(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile activity candidates from schema, AI analysis, and data analysis."""
        
        analysis = {
            'activity_candidates': [],
            'recommended_activities': [],
            'activities_to_aggregate': [],
            'activity_patterns': []
        }
        
        # PRIORITY 1: Schema analysis results (PRIMARY SOURCE)
        if schema_info:
            schema_analysis = self._analyze_schema(schema_info)
            schema_elements = schema_analysis.get('process_mining_elements', {})
            
            # Extract activity candidates from schema
            for candidate in schema_elements.get('activity_candidates', []):
                analysis['activity_candidates'].append({
                    'table': candidate['source_file'],
                    'column': candidate['name'],
                    'confidence': candidate['confidence'],
                    'reasoning': candidate['reason'],
                    'source': 'schema_analysis'
                })
        
        # PRIORITY 2: AI analysis (if available)
        if ai_insights.get('ai_analysis') and 'activity_analysis' in ai_insights:
            activity_ai = ai_insights['activity_analysis']
            
            if activity_ai.get('primary_recommendation'):
                primary = activity_ai['primary_recommendation']
                primary['source'] = 'ai_primary'
                analysis['activity_candidates'].append(primary)
            
            for alt in activity_ai.get('alternatives', []):
                alt['source'] = 'ai_alternative'
                analysis['activity_candidates'].append(alt)
            
            analysis['recommended_activities'] = activity_ai.get('recommended_activities', [])
            analysis['activities_to_aggregate'] = activity_ai.get('aggregation_suggestions', [])
        
        # PRIORITY 3: Data analysis (validation)
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            for col in metadata.get('potential_activities', []):
                col_info = metadata.get('columns', {}).get(col, {})
                analysis['activity_candidates'].append({
                    'table': dataset['file_path'],
                    'column': col,
                    'confidence': self._calculate_activity_confidence(col, col_info),
                    'reasoning': 'Data structure analysis',
                    'source': 'data_analysis'
                })
                
                # Analyze activity patterns in the data
                if 'data' in dataset:
                    patterns = self._analyze_activity_patterns(dataset['data'], col)
                    if patterns:
                        analysis['activity_patterns'].extend(patterns)
        
        # Remove duplicates from candidates
        unique_candidates = {}
        for candidate in analysis['activity_candidates']:
            key = f"{candidate['table']}::{candidate['column']}"
            if key not in unique_candidates or candidate['confidence'] > unique_candidates[key]['confidence']:
                unique_candidates[key] = candidate
        
        analysis['activity_candidates'] = sorted(unique_candidates.values(), key=lambda x: x['confidence'], reverse=True)[:25]
        
        return analysis
    
    def _compile_timestamp_analysis(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile timestamp candidates from schema, AI analysis, and data analysis."""
        
        analysis = {
            'timestamp_candidates': [],
            'temporal_coverage': {},
            'timestamp_patterns': []
        }
        
        # PRIORITY 1: Schema analysis results (PRIMARY SOURCE)
        if schema_info:
            schema_analysis = self._analyze_schema(schema_info)
            schema_elements = schema_analysis.get('process_mining_elements', {})
            
            # Extract timestamp candidates from schema
            for candidate in schema_elements.get('timestamp_candidates', []):
                analysis['timestamp_candidates'].append({
                    'table': candidate['source_file'],
                    'column': candidate['name'],
                    'confidence': candidate['confidence'],
                    'reasoning': candidate['reason'],
                    'source': 'schema_analysis'
                })
        
        # PRIORITY 2: AI analysis (if available)
        if ai_insights.get('ai_analysis') and 'timestamp_analysis' in ai_insights:
            timestamp_ai = ai_insights['timestamp_analysis']
            
            if timestamp_ai.get('primary_recommendation'):
                primary = timestamp_ai['primary_recommendation']
                primary['source'] = 'ai_primary'
                analysis['timestamp_candidates'].append(primary)
            
            for alt in timestamp_ai.get('alternatives', []):
                alt['source'] = 'ai_alternative'
                analysis['timestamp_candidates'].append(alt)
        
        # PRIORITY 3: Data analysis (validation)
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            for col in metadata.get('potential_timestamps', []):
                col_info = metadata.get('columns', {}).get(col, {})
                analysis['timestamp_candidates'].append({
                    'table': dataset['file_path'],
                    'column': col,
                    'confidence': self._calculate_timestamp_confidence(col, col_info),
                    'reasoning': 'Data structure analysis',
                    'source': 'data_analysis'
                })
                
                # Analyze temporal coverage in the data
                if 'data' in dataset:
                    coverage = self._analyze_temporal_coverage(dataset['data'], col)
                    if coverage:
                        analysis['temporal_coverage'][col] = coverage
        
        # Remove duplicates and sort by confidence
        unique_candidates = {}
        for candidate in analysis['timestamp_candidates']:
            key = f"{candidate['table']}::{candidate['column']}"
            if key not in unique_candidates or candidate['confidence'] > unique_candidates[key]['confidence']:
                unique_candidates[key] = candidate
        
        analysis['timestamp_candidates'] = sorted(unique_candidates.values(), key=lambda x: x['confidence'], reverse=True)[:25]
        
        return analysis
    

    
    def _assess_data_quality(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple data quality check - just flag obvious issues."""
        
        issues = []
        
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            missing_pct = metadata.get('missing_data', {}).get('missing_percentage', 0)
            
            if missing_pct > 30:  # Only flag serious issues
                issues.append(f"High missing data in {dataset['file_path']}: {missing_pct:.1f}%")
        
        return {
            'issues': issues,
            'has_critical_issues': len(issues) > 0
        }
    
    def _assess_readiness(self, datasets: List[Dict[str, Any]], ai_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Simple readiness check - do we have the basics?"""
        
        # Check if we found essential elements
        has_case_id = any(dataset.get('metadata', {}).get('potential_identifiers') for dataset in datasets)
        has_activity = any(dataset.get('metadata', {}).get('potential_activities') for dataset in datasets)
        has_timestamp = any(dataset.get('metadata', {}).get('potential_timestamps') for dataset in datasets)
        
        missing = []
        if not has_case_id:
            missing.append('Case ID')
        if not has_activity:
            missing.append('Activity')
        if not has_timestamp:
            missing.append('Timestamp')
        
        return {
            'ready': len(missing) == 0,
            'missing_elements': missing
        }
    
    def _calculate_readiness_score(self, event_completeness: float, temporal_coverage: float, case_id_quality: float) -> float:
        """Calculate readiness score based on key metrics."""
        # Weighted average of key metrics
        readiness_score = (
            event_completeness * 0.5 +  # Event completeness is most important
            temporal_coverage * 0.3 +  # Temporal coverage is secondary
            case_id_quality * 0.2      # Case ID quality is least important
        )
        return round(readiness_score, 1)
    
    def _compile_attributes(
        self,
        datasets: List[Dict[str, Any]],
        case_candidates: List[Dict[str, Any]],
        activity_candidates: List[Dict[str, Any]],
        timestamp_candidates: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Compile context attributes from schema and data - case-level and event-level.
        
        Returns:
            Tuple of (case_attributes, event_attributes)
        """
        case_attributes = []
        event_attributes = []
        
        # Get column names that are already identified as core elements
        identified_elements = set()
        for candidates in [case_candidates, activity_candidates, timestamp_candidates]:
            for candidate in candidates:
                identified_elements.add(candidate.get('column', '').lower())
        
        # PRIORITY 1: Schema attributes (PRIMARY SOURCE)
        if schema_info:
            schema_analysis = self._analyze_schema(schema_info)
            schema_elements = schema_analysis.get('process_mining_elements', {})
            
            for attr in schema_elements.get('attribute_candidates', []):
                attr_name = attr['name'].lower()
                if attr_name not in identified_elements:
                    attribute_info = {
                        'table': attr['source_file'],
                        'column': attr['name'],
                        'data_type': attr.get('type', 'unknown'),
                        'source': 'schema_analysis',
                        'category': attr.get('category', 'general_attribute')
                    }
                    
                    # Categorize as case or event attribute
                    if attribute_info['category'] == 'case_attribute':
                        case_attributes.append(attribute_info)
                    elif attribute_info['category'] == 'event_attribute':
                        event_attributes.append(attribute_info)
                    else:
                        # Default categorization for general attributes
                        case_attributes.append(attribute_info)
        
        # PRIORITY 2: Data-derived attributes (validation/enhancement)
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            
            for col_name, col_info in metadata.get('columns', {}).items():
                # Skip columns already identified as core elements
                if col_name.lower() in identified_elements:
                    continue
                
                attribute_info = {
                    'table': dataset['file_path'],
                    'column': col_name,
                    'data_type': col_info.get('dtype', 'unknown'),
                    'unique_values': col_info.get('unique_count', 0),
                    'missing_percentage': col_info.get('missing_percentage', 0),
                    'source': 'data_analysis',
                    'category': self._categorize_attribute(col_name)
                }
                
                # Simple classification: low variability = case attribute, high = event attribute
                unique_pct = col_info.get('unique_percentage', 0)
                if unique_pct < 20:  # Low variability - likely case attribute
                    case_attributes.append(attribute_info)
                else:  # Higher variability - likely event attribute
                    event_attributes.append(attribute_info)
        
        return case_attributes, event_attributes
    

    

    

    

    
    def _calculate_case_id_confidence(self, col_name: str, col_info: Dict[str, Any]) -> float:
        """Calculate confidence for case ID candidate."""
        score = 0.0
        name_lower = col_name.lower()
        
        if 'case' in name_lower and 'id' in name_lower:
            score += 0.9
        elif any(pattern in name_lower for pattern in ['order_id', 'ticket_id', 'instance_id']):
            score += 0.8
        elif 'id' in name_lower:
            score += 0.5
        
        # Boost score if unique
        if col_info.get('unique_percentage', 0) > 90:
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_activity_confidence(self, col_name: str, col_info: Dict[str, Any]) -> float:
        """Calculate confidence for activity candidate."""
        score = 0.0
        name_lower = col_name.lower()
        
        activity_keywords = ['activity', 'event', 'action', 'status', 'state', 'step']
        for keyword in activity_keywords:
            if keyword in name_lower:
                score += 0.8
                break
        
        # Check if reasonable number of unique values for activities
        unique_pct = col_info.get('unique_percentage', 0)
        if 5 <= unique_pct <= 50:  # Sweet spot for activities
            score += 0.4
        
        return min(1.0, score)
    
    def _calculate_timestamp_confidence(self, col_name: str, col_info: Dict[str, Any]) -> float:
        """Calculate confidence for timestamp candidate."""
        score = 0.0
        name_lower = col_name.lower()
        
        timestamp_keywords = ['time', 'date', 'timestamp', 'created', 'updated', 'occurred']
        for keyword in timestamp_keywords:
            if keyword in name_lower:
                score += 0.8
                break
        
        # Check data type
        dtype = col_info.get('dtype', '').lower()
        if 'datetime' in dtype or 'timestamp' in dtype:
            score += 0.5
        
        return min(1.0, score)
    

    

    

    
    def _analyze_elements_for_process_mining(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze schema elements for process mining potential with enhanced keyword matching."""
        
        case_id_candidates = []
        activity_candidates = []
        timestamp_candidates = []
        attribute_candidates = []
        
        # Enhanced keyword sets for better domain coverage
        case_keywords = [
            'id', 'gid', 'case', 'order', 'ticket', 'transaction', 'application', 'request', 'invoice',
            'shipment', 'release', 'instance', 'number', 'ref', 'reference', 'key', 'identifier'
        ]
        
        activity_keywords = [
            'status', 'state', 'activity', 'event', 'action', 'step', 'phase', 'stage', 'method', 
            'type', 'code', 'instruction', 'service', 'mode', 'process', 'transaction'
        ]
        
        timestamp_keywords = [
            'date', 'time', 'timestamp', 'created', 'updated', 'modified', 'when', 'effective', 
            'expiration', 'appointment', 'delivery', 'pickup', 'schedule', 'start', 'end'
        ]
        
        for element in elements:
            element_name = element['name'].lower() if element.get('name') else ''
            element_type = (element.get('type') or '').lower()
            source_file = element.get('source_file', 'unknown')
            
            if not element_name:  # Skip elements without names
                continue
            
            # Enhanced case ID detection with confidence scoring
            case_confidence = 0.0
            if any(keyword in element_name for keyword in case_keywords):
                if 'gid' in element_name:  # Global ID is very strong indicator
                    case_confidence = 0.95
                elif element_name.endswith('id') or element_name.endswith('gid'):
                    case_confidence = 0.9
                elif 'id' in element_name and ('order' in element_name or 'trans' in element_name):
                    case_confidence = 0.85
                elif 'id' in element_name:
                    case_confidence = 0.7
                elif 'number' in element_name or 'ref' in element_name:
                    case_confidence = 0.6
                else:
                    case_confidence = 0.5
                
                if case_confidence > 0.5:
                    case_id_candidates.append({
                        'name': element['name'],
                        'source_file': os.path.basename(source_file),
                        'confidence': case_confidence,
                        'reason': f"Case ID indicator: {element_name} (confidence: {case_confidence:.2f})"
                    })
            
            # Enhanced activity detection
            activity_confidence = 0.0
            if any(keyword in element_name for keyword in activity_keywords):
                if 'transactioncode' in element_name.replace('_', '').replace('-', ''):
                    activity_confidence = 0.9
                elif 'status' in element_name or 'state' in element_name:
                    activity_confidence = 0.8
                elif 'type' in element_name or 'method' in element_name:
                    activity_confidence = 0.75
                elif 'code' in element_name:
                    activity_confidence = 0.7
                else:
                    activity_confidence = 0.6
                
                if activity_confidence > 0.5:
                    activity_candidates.append({
                        'name': element['name'],
                        'source_file': os.path.basename(source_file),
                        'confidence': activity_confidence,
                        'reason': f"Activity indicator: {element_name} (confidence: {activity_confidence:.2f})"
                    })
            
            # Enhanced timestamp detection
            timestamp_confidence = 0.0
            if any(keyword in element_name for keyword in timestamp_keywords) or 'date' in element_type or 'time' in element_type:
                if 'timestamp' in element_name:
                    timestamp_confidence = 0.95
                elif element_name.endswith('date') or element_name.endswith('time'):
                    timestamp_confidence = 0.9
                elif 'date' in element_name or 'time' in element_name:
                    timestamp_confidence = 0.85
                elif 'created' in element_name or 'updated' in element_name:
                    timestamp_confidence = 0.8
                elif 'effective' in element_name or 'expiration' in element_name:
                    timestamp_confidence = 0.75
                else:
                    timestamp_confidence = 0.6
                
                if timestamp_confidence > 0.5:
                    timestamp_candidates.append({
                        'name': element['name'],
                        'source_file': os.path.basename(source_file),
                        'confidence': timestamp_confidence,
                        'reason': f"Timestamp indicator: {element_name} (confidence: {timestamp_confidence:.2f})"
                    })
            
            # Categorize remaining elements as attributes
            if not any([
                any(keyword in element_name for keyword in case_keywords),
                any(keyword in element_name for keyword in activity_keywords),
                any(keyword in element_name for keyword in timestamp_keywords)
            ]):
                attribute_candidates.append({
                    'name': element['name'],
                    'source_file': os.path.basename(source_file),
                    'type': element.get('type', 'unknown'),
                    'category': self._categorize_attribute(element_name)
                })
        
        # Sort by confidence and return top candidates
        case_id_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        activity_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        timestamp_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'case_id_candidates': case_id_candidates[:25],  # Top 25
            'activity_candidates': activity_candidates[:25],
            'timestamp_candidates': timestamp_candidates[:25],
            'attribute_candidates': attribute_candidates[:50],
            'summary': {
                'total_case_candidates': len(case_id_candidates),
                'total_activity_candidates': len(activity_candidates),
                'total_timestamp_candidates': len(timestamp_candidates),
                'total_attributes': len(attribute_candidates)
            }
        }
    
    def _categorize_attribute(self, element_name: str) -> str:
        """Categorize an attribute as case-level or event-level based on naming patterns."""
        element_name = element_name.lower()
        
        # Case-level attributes (properties of the entire process instance)
        case_patterns = [
            'customer', 'order', 'total', 'priority', 'type', 'amount', 'value',
            'mode', 'profile', 'group', 'plan', 'destination', 'source'
        ]
        
        # Event-level attributes (properties of individual activities)
        event_patterns = [
            'user', 'location', 'service', 'equipment', 'route', 'provider',
            'instruction', 'detail', 'point', 'reference'
        ]
        
        if any(pattern in element_name for pattern in case_patterns):
            return 'case_attribute'
        elif any(pattern in element_name for pattern in event_patterns):
            return 'event_attribute'
        else:
            return 'general_attribute'
    
    def _generate_suggested_sql(
        self,
        case_candidates: List[Dict[str, Any]],
        activity_candidates: List[Dict[str, Any]],
        timestamp_candidates: List[Dict[str, Any]],
        datasets: List[Dict[str, Any]]
    ) -> str:
        """Generate an MVP suggested SQL to assemble a case-centric event log."""
        if not datasets and not (case_candidates or activity_candidates or timestamp_candidates):
            return "-- No inputs available to suggest SQL."
        
        def pick(cands: List[Dict[str, Any]], preferred_prefix: str) -> Optional[Dict[str, Any]]:
            if not cands:
                return None
            ai = [c for c in cands if str(c.get('source', '')).startswith(preferred_prefix)]
            return (ai or cands)[0]
        
        case_c = pick(case_candidates, 'ai_')
        act_c = pick(activity_candidates, 'ai_')
        ts_c = pick(timestamp_candidates, 'ai_')
        
        def sanitize_ident(name: str) -> str:
            return ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in name)
        
        def table_from_path(path: str) -> str:
            if not path:
                return 'source_table'
            base = os.path.splitext(os.path.basename(path))[0]
            return sanitize_ident(base)
        
        # Derive tables/columns with fallbacks
        case_table = table_from_path(case_c.get('table')) if case_c and case_c.get('table') else (table_from_path(datasets[0]['file_path']) if datasets else 'source_table')
        case_col = case_c.get('column', 'case_id') if case_c else 'case_id'
        
        act_table = table_from_path(act_c.get('table')) if act_c and act_c.get('table') else case_table
        act_col = act_c.get('column', 'activity') if act_c else 'activity'
        
        ts_table = table_from_path(ts_c.get('table')) if ts_c and ts_c.get('table') else act_table
        ts_col = ts_c.get('column', 'event_time') if ts_c else 'event_time'
        
        join_key = case_col
        
        sql = f"""-- Suggested SQL (MVP) to assemble a case-centric event log
-- Assumptions:
--  - Tables derived from provided files (update schema/table names as needed)
--  - Join keys and filters may require refinement

WITH activities AS (
    SELECT
        a.{case_col} AS case_id,
        a.{act_col} AS activity,
        a.{ts_col}  AS event_time,
        a.*
    FROM {act_table} a
),
cases AS (
    SELECT
        c.{case_col} AS case_id,
        c.*
    FROM {case_table} c
)
SELECT
    e.case_id,
    e.activity,
    e.event_time AS timestamp,
    -- Add selected attributes below (case- and event-level)
    -- e.attribute1, e.attribute2, c.case_attribute1
FROM activities e
JOIN cases c
  ON e.{join_key} = c.{case_col}
WHERE e.event_time IS NOT NULL
ORDER BY e.case_id, e.event_time;
"""
        return sql
    
    def generate_yaml_output(self, readiness_score: float, success_metrics: dict, recommendations: list) -> str:
        """Generate structured YAML output for assessment results."""
        import yaml

        output = {
            "readiness_status": "Ready" if readiness_score >= 0.8 else "Needs Improvement",
            "readiness_score": readiness_score,
            "success_metrics": success_metrics,
            "recommendations": recommendations,
        }

        return yaml.dump(output, default_flow_style=False)
    
    def _analyze_activity_patterns(self, data: pd.DataFrame, activity_col: str) -> List[Dict[str, Any]]:
        """Analyze patterns in activity data."""
        if activity_col not in data.columns:
            return []
        
        patterns = []
        
        # Get activity distribution
        activity_counts = data[activity_col].value_counts()
        
        if len(activity_counts) > 0:
            patterns.append({
                'type': 'activity_distribution',
                'total_activities': len(activity_counts),
                'most_common': dict(activity_counts.head().items()),
                'activity_balance': float(activity_counts.std() / activity_counts.mean()) if activity_counts.mean() > 0 else 0
            })
        
        return patterns
    
    def _analyze_temporal_coverage(self, data: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze temporal coverage of timestamp data."""
        if timestamp_col not in data.columns:
            return {}
        
        try:
            timestamps = pd.to_datetime(data[timestamp_col], errors='coerce').dropna()
            if len(timestamps) == 0:
                return {}
            
            return {
                'date_range': {
                    'start': timestamps.min().isoformat(),
                    'end': timestamps.max().isoformat(),
                    'span_days': (timestamps.max() - timestamps.min()).days
                },
                'coverage': len(timestamps) / len(data),
                'has_temporal_order': timestamps.is_monotonic_increasing
            }
        except Exception:
            return {}

    def _generate_working_sql(self, datasets: List[Dict[str, Any]]) -> str:
        """Generate working SQL that uses actual detected column names."""
        
        if not datasets:
            return "-- No datasets available for SQL generation"
        
        # Find the primary dataset (usually the one with most rows and useful columns)
        primary_dataset = self._find_primary_dataset(datasets)
        
        if not primary_dataset or primary_dataset.get('data') is None:
            return "-- No suitable dataset found for SQL generation"
        
        data = primary_dataset['data']
        columns = list(data.columns)
        
        # Find actual column names for key elements
        case_id_col = self._find_best_column_match(columns, ['id', 'number', 'key', 'event id', 'case', 'incident'])
        activity_col = self._find_best_column_match(columns, ['status', 'state', 'activity', 'event', 'action'])
        timestamp_col = self._find_best_column_match(columns, ['date', 'time', 'timestamp', 'created', 'updated'])
        
        # Generate clean table name
        table_name = self._generate_clean_table_name(primary_dataset.get('file_path', 'data_table'))
        
        # Build SQL components
        sql_parts = []
        sql_parts.append("-- Process Mining Event Log SQL")
        sql_parts.append(f"-- Generated from: {self._get_clean_filename(primary_dataset.get('file_path', 'unknown'))}")
        sql_parts.append(f"-- Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sql_parts.append(f"-- Rows in source: {len(data):,}")
        sql_parts.append("")
        
        # Add data quality notes
        if hasattr(data, 'isnull'):
            missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            sql_parts.append(f"-- Data quality: {100-missing_pct:.1f}% complete ({missing_pct:.1f}% missing)")
        
        sql_parts.append("")
        sql_parts.append("SELECT")
        
        # Case ID column
        if case_id_col:
            sql_parts.append(f"    [{case_id_col}] as case_id,")
            sql_parts.append(f"    -- Business meaning: Unique identifier for each process instance")
        else:
            sql_parts.append("    -- ⚠️  No clear case ID found - manual review required")
            sql_parts.append("    [COLUMN_NAME] as case_id,  -- TODO: Replace with actual case ID column")
        
        # Activity column
        if activity_col:
            sql_parts.append(f"    [{activity_col}] as activity,")
            sql_parts.append(f"    -- Business meaning: What happened in the process")
        else:
            sql_parts.append("    -- ⚠️  No clear activity found - manual review required")
            sql_parts.append("    [COLUMN_NAME] as activity,  -- TODO: Replace with actual activity column")
        
        # Timestamp column
        if timestamp_col:
            sql_parts.append(f"    [{timestamp_col}] as timestamp,")
            sql_parts.append(f"    -- Business meaning: When the activity occurred")
        else:
            sql_parts.append("    -- ❌ CRITICAL: No timestamp found - process mining requires temporal data")
            sql_parts.append("    ROW_NUMBER() OVER (PARTITION BY case_id ORDER BY case_id) as event_sequence,")
            sql_parts.append("    -- TODO: Request timestamp data from system administrators")
        
        # Additional useful columns (attributes)
        other_useful_cols = self._find_useful_attribute_columns(columns, [case_id_col, activity_col, timestamp_col])
        if other_useful_cols:
            sql_parts.append("    ")
            sql_parts.append("    -- Additional attributes for analysis:")
            for col in other_useful_cols[:5]:  # Limit to top 5
                sql_parts.append(f"    [{col}],")
        
        # Remove last comma
        if sql_parts[-1].endswith(','):
            sql_parts[-1] = sql_parts[-1][:-1]
        
        sql_parts.append("")
        sql_parts.append(f"FROM [{table_name}]")
        
        # Add WHERE conditions
        where_conditions = []
        if case_id_col:
            where_conditions.append(f"[{case_id_col}] IS NOT NULL")
        
        if timestamp_col:
            where_conditions.append(f"[{timestamp_col}] IS NOT NULL")
        
        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Add ORDER BY
        if timestamp_col:
            sql_parts.append(f"ORDER BY case_id, [{timestamp_col}];")
        else:
            sql_parts.append("ORDER BY case_id, event_sequence;")
        
        # Add implementation notes
        sql_parts.extend([
            "",
            "-- Implementation Notes:",
            "-- 1. Validate column mappings with business stakeholders",
            "-- 2. Test query with small sample first", 
            "-- 3. Add data quality filters as needed",
            "-- 4. Consider case and event attribute selection",
            ""
        ])
        
        # Add business context
        if case_id_col and activity_col:
            sql_parts.extend([
                "-- Expected Business Value:",
                "-- • Process flow visualization and analysis",
                "-- • Bottleneck identification and cycle time analysis", 
                "-- • Compliance and variance detection",
                "-- • Performance improvement opportunities"
            ])
        else:
            sql_parts.extend([
                "-- Next Steps Required:",
                "-- • Identify proper case ID and activity columns",
                "-- • Obtain timestamp data for temporal analysis",
                "-- • Validate business meaning with domain experts"
            ])
        
        return "\n".join(sql_parts)

    def _find_primary_dataset(self, datasets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the most suitable dataset for SQL generation."""
        
        scored_datasets = []
        
        for dataset in datasets:
            if dataset.get('data') is None:
                continue
            
            score = 0
            data = dataset['data']
            columns = [str(col).lower() for col in data.columns]
            
            # Score based on useful columns
            useful_indicators = ['id', 'status', 'date', 'event', 'activity', 'time']
            for indicator in useful_indicators:
                if any(indicator in col for col in columns):
                    score += 1
            
            # Score based on data size (more rows = more useful)
            score += min(len(data) / 1000, 5)  # Cap at 5 points for size
            
            # Penalty for too many unnamed columns
            unnamed_count = sum(1 for col in data.columns if 'unnamed' in str(col).lower())
            score -= unnamed_count * 0.5
            
            scored_datasets.append((score, dataset))
        
        if not scored_datasets:
            return None
        
        # Return highest scoring dataset
        scored_datasets.sort(reverse=True, key=lambda x: x[0])
        return scored_datasets[0][1]

    def _find_best_column_match(self, columns: List[str], indicators: List[str]) -> Optional[str]:
        """Find the best column matching the given indicators."""
        
        # First pass: exact matches
        for indicator in indicators:
            for col in columns:
                if indicator.lower() == col.lower():
                    return col
        
        # Second pass: partial matches
        for indicator in indicators:
            for col in columns:
                if indicator.lower() in col.lower():
                    return col
        
        return None

    def _generate_clean_table_name(self, file_path: str) -> str:
        """Generate a clean table name from file path."""
        
        if '#' in file_path:
            # Handle Excel tab reference
            parts = file_path.split('#')
            base_name = parts[0].split('/')[-1] if '/' in parts[0] else parts[0]
            tab_name = parts[1] if len(parts) > 1 else ''
            
            # Clean base name
            base_clean = base_name.split('.')[0].replace(' ', '_').replace('-', '_')
            tab_clean = tab_name.replace(' ', '_').replace('-', '_')
            
            return f"{base_clean}_{tab_clean}" if tab_clean else base_clean
        else:
            # Regular file
            file_name = file_path.split('/')[-1] if '/' in file_path else file_path
            table_name = file_name.split('.')[0]  # Remove extension
            return table_name.replace(' ', '_').replace('-', '_')

    def _find_useful_attribute_columns(self, all_columns: List[str], exclude_columns: List[str]) -> List[str]:
        """Find columns that would be useful as case or event attributes."""
        
        # Remove None values from exclude list
        exclude_set = {col for col in exclude_columns if col is not None}
        
        useful_columns = []
        
        for col in all_columns:
            if col in exclude_set:
                continue
            
            col_lower = str(col).lower()
            
            # Skip obviously useless columns
            if any(skip in col_lower for skip in ['unnamed', 'index', 'level_']):
                continue
            
            # Prefer columns with business meaning
            business_indicators = [
                'user', 'person', 'owner', 'assigned', 'responsible',
                'priority', 'category', 'type', 'department', 'team',
                'customer', 'product', 'service', 'location', 'region',
                'amount', 'value', 'cost', 'price', 'quantity'
            ]
            
            is_business_relevant = any(indicator in col_lower for indicator in business_indicators)
            
            if is_business_relevant or len(useful_columns) < 3:  # Always include some columns
                useful_columns.append(col)
        
        return useful_columns[:5]  # Limit to 5 most useful
