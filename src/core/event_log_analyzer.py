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
        """Generate streamlined process mining assessment focused on core elements.
        
        Args:
            datasets: List of dataset information
            schema_info: Parsed schema information
            ai_insights: AI-generated insights
            business_context: Business context description
            
        Returns:
            Streamlined assessment focused on UIDs, activities, and attributes
        """
        logger.info("Generating streamlined process mining assessment")
        
        # CORE VALUE 1: Identify potential UIDs (Case IDs)
        case_id_candidates = self._compile_case_id_candidates(datasets, schema_info, ai_insights)
        
        # CORE VALUE 2: Identify activities/events with timestamps
        activity_analysis = self._compile_activity_analysis(datasets, schema_info, ai_insights)
        timestamp_analysis = self._compile_timestamp_analysis(datasets, schema_info, ai_insights)
        
        # CORE VALUE 3: Identify attributes for context
        case_attributes, event_attributes = self._compile_attributes(datasets, case_id_candidates, 
                                                                   activity_analysis.get('activity_candidates', []), 
                                                                   timestamp_analysis.get('timestamp_candidates', []),
                                                                   schema_info)
        
        # Generate SQL for immediate use
        suggested_sql = self._generate_suggested_sql(
            case_id_candidates,
            activity_analysis.get('activity_candidates', []),
            timestamp_analysis.get('timestamp_candidates', []),
            datasets
        )
        
        # Simple readiness check
        readiness = self._assess_readiness(datasets, ai_insights)
        data_quality = self._assess_data_quality(datasets)
        
        # Streamlined assessment output
        assessment = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'files_analyzed': [d.get('file_path') for d in datasets]
            },
            
            # Core findings - what the user needs
            'case_id_candidates': case_id_candidates[:5],  # Top 5 potential UIDs
            'activity_candidates': activity_analysis.get('activity_candidates', [])[:5],  # Top 5 activities
            'timestamp_candidates': timestamp_analysis.get('timestamp_candidates', [])[:5],  # Top 5 timestamps
            
            'case_attributes': case_attributes[:10],  # Case-level context attributes
            'event_attributes': event_attributes[:10],  # Event-level context attributes
            
            # Actionable output
            'suggested_sql': suggested_sql,
            
            # Simple status
            'readiness': readiness,
            'data_issues': data_quality.get('issues', []),
            
            # Optional AI insights
            'ai_insights': ai_insights.get('next_steps', []) if ai_insights.get('ai_analysis') else []
        }
        
        logger.info("Streamlined assessment completed")
        return _convert_numpy_types(assessment)
    

    
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
