"""Event log analyzer for generating comprehensive process mining assessments."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class EventLogAnalyzer:
    """Generates comprehensive assessments for process mining event log creation."""
    
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
        """Generate comprehensive process mining assessment.
        
        Args:
            datasets: List of dataset information
            schema_info: Parsed schema information
            ai_insights: AI-generated insights
            business_context: Business context description
            
        Returns:
            Complete assessment report
        """
        logger.info("Generating comprehensive process mining assessment")
        
        assessment = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'assessment_version': '1.0',
                'analyzer': 'Process Mining Event Log Assessment Assistant'
            },
            'business_context': business_context,
            'data_sources': self._analyze_data_sources(datasets),
            'schema_analysis': self._analyze_schema(schema_info) if schema_info else None,
            'case_id_candidates': self._compile_case_id_candidates(datasets, schema_info, ai_insights),
            'activity_analysis': self._compile_activity_analysis(datasets, schema_info, ai_insights),
            'timestamp_analysis': self._compile_timestamp_analysis(datasets, schema_info, ai_insights),
            'attribute_mapping': self._compile_attribute_mapping(datasets, schema_info, ai_insights),
            'data_quality': self._assess_data_quality(datasets),
            'process_mining_readiness': self._assess_readiness(datasets, ai_insights),
            'recommendations': self._compile_recommendations(datasets, schema_info, ai_insights),
            'transformation_plan': self._generate_transformation_plan(datasets, ai_insights),
            'next_steps': self._generate_next_steps(ai_insights),
            'ai_insights': ai_insights if ai_insights.get('ai_analysis', False) else None
        }
        
        logger.info("Assessment generation completed")
        return assessment
    
    def _analyze_data_sources(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the available data sources."""
        
        analysis = {
            'total_files': len(datasets),
            'file_details': [],
            'overall_statistics': {
                'total_rows': 0,
                'total_columns': 0,
                'unique_columns': set()
            }
        }
        
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            file_info = {
                'file_path': dataset['file_path'],
                'format': dataset['file_path'].split('.')[-1].upper(),
                'shape': metadata.get('shape', {}),
                'data_types': metadata.get('data_types', {}),
                'missing_data': metadata.get('missing_data', {}),
                'columns': list(metadata.get('columns', {}).keys()),
                'data_quality_score': self._calculate_file_quality_score(metadata)
            }
            
            analysis['file_details'].append(file_info)
            
            # Update overall statistics
            shape = metadata.get('shape', {})
            analysis['overall_statistics']['total_rows'] += shape.get('rows', 0)
            analysis['overall_statistics']['total_columns'] += shape.get('columns', 0)
            analysis['overall_statistics']['unique_columns'].update(metadata.get('columns', {}).keys())
        
        analysis['overall_statistics']['unique_columns'] = list(analysis['overall_statistics']['unique_columns'])
        
        return analysis
    
    def _analyze_schema(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema information for process mining relevance."""
        
        analysis = {
            'schema_type': schema_info.get('type'),
            'summary': {},
            'process_mining_elements': {}
        }
        
        if schema_info.get('type') == 'sql':
            tables = schema_info.get('tables', {})
            analysis['summary'] = {
                'total_tables': len(tables),
                'tables_with_relationships': len([t for t in tables.values() if t.get('foreign_keys')]),
                'total_columns': sum(len(t.get('columns', {})) for t in tables.values())
            }
            
            # Get process mining candidates from schema analyzer
            if hasattr(schema_info, 'process_mining_candidates'):
                analysis['process_mining_elements'] = schema_info['process_mining_candidates']
        
        return analysis
    
    def _compile_case_id_candidates(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        ai_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compile case ID candidates from all sources."""
        
        candidates = []
        
        # From AI analysis
        if ai_insights.get('ai_analysis') and 'case_id_analysis' in ai_insights:
            case_analysis = ai_insights['case_id_analysis']
            
            if case_analysis.get('primary_recommendation'):
                primary = case_analysis['primary_recommendation']
                primary['source'] = 'ai_primary'
                candidates.append(primary)
            
            for alt in case_analysis.get('alternatives', []):
                alt['source'] = 'ai_alternative'
                candidates.append(alt)
        
        # From data analysis
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            for col in metadata.get('potential_identifiers', []):
                col_info = metadata.get('columns', {}).get(col, {})
                candidates.append({
                    'table': dataset['file_path'],
                    'column': col,
                    'confidence': self._calculate_case_id_confidence(col, col_info),
                    'reasoning': 'Data structure analysis',
                    'source': 'data_analysis'
                })
        
        # Remove duplicates and sort by confidence
        unique_candidates = {}
        for candidate in candidates:
            key = f"{candidate['table']}::{candidate['column']}"
            if key not in unique_candidates or candidate['confidence'] > unique_candidates[key]['confidence']:
                unique_candidates[key] = candidate
        
        sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x['confidence'], reverse=True)
        
        return sorted_candidates[:10]  # Top 10 candidates
    
    def _compile_activity_analysis(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile activity analysis from all sources."""
        
        analysis = {
            'activity_candidates': [],
            'recommended_activities': [],
            'activities_to_aggregate': [],
            'activity_patterns': []
        }
        
        # From AI analysis
        if ai_insights.get('ai_analysis') and 'activity_analysis' in ai_insights:
            activity_ai = ai_insights['activity_analysis']
            
            if activity_ai.get('primary_recommendation'):
                primary = activity_ai['primary_recommendation']
                primary['source'] = 'ai_primary'
                analysis['activity_candidates'].append(primary)
            
            for alt in activity_ai.get('alternatives', []):
                alt['source'] = 'ai_alternative'
                analysis['activity_candidates'].append(alt)
            
            analysis['activities_to_aggregate'] = activity_ai.get('aggregation_suggestions', [])
        
        # From data analysis
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
        
        analysis['activity_candidates'] = sorted(unique_candidates.values(), key=lambda x: x['confidence'], reverse=True)
        
        return analysis
    
    def _compile_timestamp_analysis(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile timestamp analysis from all sources."""
        
        analysis = {
            'timestamp_candidates': [],
            'temporal_coverage': {},
            'timestamp_quality': {}
        }
        
        # From AI analysis
        if ai_insights.get('ai_analysis') and 'timestamp_analysis' in ai_insights:
            timestamp_ai = ai_insights['timestamp_analysis']
            
            if timestamp_ai.get('primary_recommendation'):
                primary = timestamp_ai['primary_recommendation']
                primary['source'] = 'ai_primary'
                analysis['timestamp_candidates'].append(primary)
            
            for alt in timestamp_ai.get('alternatives', []):
                alt['source'] = 'ai_alternative'
                analysis['timestamp_candidates'].append(alt)
        
        # From data analysis
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
                
                # Analyze temporal coverage
                if 'data' in dataset and col in dataset['data'].columns:
                    coverage = self._analyze_temporal_coverage(dataset['data'], col)
                    if coverage:
                        analysis['temporal_coverage'][f"{dataset['file_path']}::{col}"] = coverage
        
        # Remove duplicates
        unique_candidates = {}
        for candidate in analysis['timestamp_candidates']:
            key = f"{candidate['table']}::{candidate['column']}"
            if key not in unique_candidates or candidate['confidence'] > unique_candidates[key]['confidence']:
                unique_candidates[key] = candidate
        
        analysis['timestamp_candidates'] = sorted(unique_candidates.values(), key=lambda x: x['confidence'], reverse=True)
        
        return analysis
    
    def _compile_attribute_mapping(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile attribute mapping recommendations."""
        
        mapping = {
            'case_attributes': [],
            'event_attributes': [],
            'derived_attributes': []
        }
        
        # From AI analysis
        if ai_insights.get('ai_analysis') and 'attribute_recommendations' in ai_insights:
            attr_rec = ai_insights['attribute_recommendations']
            mapping['case_attributes'] = attr_rec.get('case_attributes', [])
            mapping['event_attributes'] = attr_rec.get('event_attributes', [])
        
        # From data analysis - identify potential attributes
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            
            for col_name, col_info in metadata.get('columns', {}).items():
                # Skip columns that are already identified as case_id, activity, or timestamp
                if col_name in (metadata.get('potential_identifiers', []) + 
                              metadata.get('potential_activities', []) + 
                              metadata.get('potential_timestamps', [])):
                    continue
                
                attribute_info = {
                    'table': dataset['file_path'],
                    'column': col_name,
                    'data_type': col_info.get('dtype'),
                    'unique_percentage': col_info.get('unique_percentage', 0),
                    'missing_percentage': col_info.get('missing_percentage', 0),
                    'reasoning': 'Data analysis'
                }
                
                # Classify as case or event attribute based on characteristics
                if col_info.get('unique_percentage', 0) < 50:  # Low variability suggests case attribute
                    mapping['case_attributes'].append(attribute_info)
                else:  # High variability suggests event attribute
                    mapping['event_attributes'].append(attribute_info)
        
        return mapping
    
    def _assess_data_quality(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall data quality for process mining."""
        
        quality_assessment = {
            'overall_score': 0.0,
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'validity_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        total_completeness = 0
        total_files = len(datasets)
        
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            missing_data = metadata.get('missing_data', {})
            
            # Completeness score (inverse of missing data percentage)
            missing_pct = missing_data.get('missing_percentage', 0)
            completeness = 100 - missing_pct
            total_completeness += completeness
            
            # Check for data quality issues
            if missing_pct > 20:
                quality_assessment['issues'].append(
                    f"High missing data in {dataset['file_path']}: {missing_pct:.1f}%"
                )
            
            quality_indicators = metadata.get('data_quality_indicators', {})
            if quality_indicators.get('duplicate_percentage', 0) > 5:
                quality_assessment['issues'].append(
                    f"High duplicate rate in {dataset['file_path']}: {quality_indicators['duplicate_percentage']:.1f}%"
                )
        
        # Calculate scores
        quality_assessment['completeness_score'] = total_completeness / total_files if total_files > 0 else 0
        quality_assessment['consistency_score'] = 85.0  # Placeholder - would need more complex analysis
        quality_assessment['validity_score'] = 90.0   # Placeholder - would need domain validation
        
        # Overall score (weighted average)
        weights = {'completeness': 0.4, 'consistency': 0.3, 'validity': 0.3}
        quality_assessment['overall_score'] = (
            quality_assessment['completeness_score'] * weights['completeness'] +
            quality_assessment['consistency_score'] * weights['consistency'] +
            quality_assessment['validity_score'] * weights['validity']
        )
        
        # Generate recommendations
        if quality_assessment['overall_score'] < 70:
            quality_assessment['recommendations'].append("Improve data quality before process mining analysis")
        
        if quality_assessment['completeness_score'] < 80:
            quality_assessment['recommendations'].append("Address missing data issues")
        
        if len(quality_assessment['issues']) > 0:
            quality_assessment['recommendations'].append("Review and resolve identified data quality issues")
        
        return quality_assessment
    
    def _assess_readiness(self, datasets: List[Dict[str, Any]], ai_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readiness for process mining."""
        
        readiness = {
            'score': 0.0,
            'level': 'Not Ready',
            'criteria': {
                'has_case_id': False,
                'has_activity': False,
                'has_timestamp': False,
                'sufficient_data': False,
                'data_quality_acceptable': False
            },
            'missing_elements': [],
            'recommendations': []
        }
        
        # Check for essential elements
        has_case_candidates = bool(ai_insights.get('case_id_analysis', {}).get('primary_recommendation'))
        has_activity_candidates = bool(ai_insights.get('activity_analysis', {}).get('primary_recommendation'))
        has_timestamp_candidates = bool(ai_insights.get('timestamp_analysis', {}).get('primary_recommendation'))
        
        # Check data volume
        total_rows = sum(dataset.get('metadata', {}).get('shape', {}).get('rows', 0) for dataset in datasets)
        sufficient_data = total_rows >= 100  # Minimum threshold
        
        # Check data quality
        data_quality_score = ai_insights.get('process_mining_readiness', {}).get('score', 0.5)
        data_quality_acceptable = data_quality_score >= 0.7
        
        # Update criteria
        readiness['criteria'].update({
            'has_case_id': has_case_candidates,
            'has_activity': has_activity_candidates,
            'has_timestamp': has_timestamp_candidates,
            'sufficient_data': sufficient_data,
            'data_quality_acceptable': data_quality_acceptable
        })
        
        # Calculate score
        criteria_met = sum(readiness['criteria'].values())
        readiness['score'] = criteria_met / len(readiness['criteria'])
        
        # Determine readiness level
        if readiness['score'] >= 0.8:
            readiness['level'] = 'Ready'
        elif readiness['score'] >= 0.6:
            readiness['level'] = 'Nearly Ready'
        elif readiness['score'] >= 0.4:
            readiness['level'] = 'Requires Work'
        else:
            readiness['level'] = 'Not Ready'
        
        # Identify missing elements
        if not has_case_candidates:
            readiness['missing_elements'].append('Case ID')
        if not has_activity_candidates:
            readiness['missing_elements'].append('Activity information')
        if not has_timestamp_candidates:
            readiness['missing_elements'].append('Timestamp data')
        if not sufficient_data:
            readiness['missing_elements'].append('Sufficient data volume')
        if not data_quality_acceptable:
            readiness['missing_elements'].append('Acceptable data quality')
        
        return readiness
    
    def _compile_recommendations(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        ai_insights: Dict[str, Any]
    ) -> List[str]:
        """Compile actionable recommendations."""
        
        recommendations = []
        
        # From AI insights
        if ai_insights.get('ai_analysis'):
            ai_recs = ai_insights.get('transformation_recommendations', [])
            recommendations.extend(ai_recs)
        
        # Data quality recommendations
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            missing_pct = metadata.get('missing_data', {}).get('missing_percentage', 0)
            
            if missing_pct > 20:
                recommendations.append(f"Address missing data in {dataset['file_path']} ({missing_pct:.1f}% missing)")
        
        # Essential element recommendations
        case_id_found = bool(ai_insights.get('case_id_analysis', {}).get('primary_recommendation'))
        activity_found = bool(ai_insights.get('activity_analysis', {}).get('primary_recommendation'))
        timestamp_found = bool(ai_insights.get('timestamp_analysis', {}).get('primary_recommendation'))
        
        if not case_id_found:
            recommendations.append("Identify or create a unique case identifier for process instances")
        
        if not activity_found:
            recommendations.append("Identify columns that represent process activities or events")
        
        if not timestamp_found:
            recommendations.append("Ensure timestamp information is available for all events")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:15]  # Limit to top 15 recommendations
    
    def _generate_transformation_plan(self, datasets: List[Dict[str, Any]], ai_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a data transformation plan."""
        
        plan = {
            'overview': 'Step-by-step plan to transform source data into process mining event log',
            'steps': [],
            'estimated_effort': 'Medium',
            'prerequisites': []
        }
        
        # Add transformation steps based on analysis
        step_counter = 1
        
        # Step 1: Data preparation
        plan['steps'].append({
            'step': step_counter,
            'title': 'Data Preparation',
            'description': 'Clean and prepare source data files',
            'actions': [
                'Remove duplicate records',
                'Handle missing values',
                'Standardize data formats',
                'Validate data integrity'
            ]
        })
        step_counter += 1
        
        # Step 2: Case ID identification/creation
        case_id_rec = ai_insights.get('case_id_analysis', {}).get('primary_recommendation')
        if case_id_rec:
            plan['steps'].append({
                'step': step_counter,
                'title': 'Case ID Mapping',
                'description': f"Use {case_id_rec['column']} from {case_id_rec['table']} as Case ID",
                'actions': [
                    f"Extract {case_id_rec['column']} as case_id",
                    'Validate uniqueness and consistency',
                    'Handle any missing case IDs'
                ]
            })
        else:
            plan['steps'].append({
                'step': step_counter,
                'title': 'Case ID Creation',
                'description': 'Create case identifiers from available data',
                'actions': [
                    'Analyze data patterns to identify process instances',
                    'Create composite case IDs if necessary',
                    'Validate case boundaries'
                ]
            })
        step_counter += 1
        
        # Step 3: Activity mapping
        activity_rec = ai_insights.get('activity_analysis', {}).get('primary_recommendation')
        if activity_rec:
            plan['steps'].append({
                'step': step_counter,
                'title': 'Activity Mapping',
                'description': f"Map activities from {activity_rec['column']} in {activity_rec['table']}",
                'actions': [
                    f"Extract and standardize activity names from {activity_rec['column']}",
                    'Create activity hierarchy if needed',
                    'Validate activity completeness'
                ]
            })
        step_counter += 1
        
        # Step 4: Timestamp processing
        timestamp_rec = ai_insights.get('timestamp_analysis', {}).get('primary_recommendation')
        if timestamp_rec:
            plan['steps'].append({
                'step': step_counter,
                'title': 'Timestamp Processing',
                'description': f"Process timestamps from {timestamp_rec['column']}",
                'actions': [
                    f"Convert {timestamp_rec['column']} to standard datetime format",
                    'Validate temporal ordering',
                    'Handle timezone considerations'
                ]
            })
        step_counter += 1
        
        # Step 5: Event log creation
        plan['steps'].append({
            'step': step_counter,
            'title': 'Event Log Assembly',
            'description': 'Combine all elements into final event log',
            'actions': [
                'Merge case IDs, activities, and timestamps',
                'Add case and event attributes',
                'Validate event log structure',
                'Export in standard format (CSV/XES)'
            ]
        })
        
        return plan
    
    def _generate_next_steps(self, ai_insights: Dict[str, Any]) -> List[str]:
        """Generate immediate next steps."""
        
        next_steps = []
        
        # From AI insights
        if ai_insights.get('ai_analysis') and 'next_steps' in ai_insights:
            next_steps.extend(ai_insights['next_steps'])
        
        # Default next steps
        default_steps = [
            "Review and validate the recommended case ID, activity, and timestamp mappings",
            "Prepare data transformation scripts based on the transformation plan",
            "Test the event log creation process with a sample of data",
            "Validate the resulting event log with process mining tools",
            "Iterate and refine the mapping based on initial results"
        ]
        
        # Combine and deduplicate
        all_steps = next_steps + default_steps
        seen = set()
        unique_steps = []
        for step in all_steps:
            if step not in seen:
                seen.add(step)
                unique_steps.append(step)
        
        return unique_steps[:10]  # Limit to top 10 steps
    
    # Helper methods for confidence calculations
    
    def _calculate_file_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate quality score for a file."""
        missing_pct = metadata.get('missing_data', {}).get('missing_percentage', 0)
        completeness = 100 - missing_pct
        
        quality_indicators = metadata.get('data_quality_indicators', {})
        duplicate_pct = quality_indicators.get('duplicate_percentage', 0)
        
        # Simple quality score calculation
        score = (completeness - duplicate_pct) / 100
        return max(0.0, min(1.0, score))
    
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
                'most_common': activity_counts.head().to_dict(),
                'activity_balance': float(activity_counts.std() / activity_counts.mean()) if activity_counts.mean() > 0 else 0
            })
        
        return patterns
    
    def _analyze_temporal_coverage(self, data: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze temporal coverage of timestamp data."""
        if timestamp_col not in data.columns:
            return {}
        
        try:
            # Try to convert to datetime
            timestamps = pd.to_datetime(data[timestamp_col], errors='coerce')
            valid_timestamps = timestamps.dropna()
            
            if len(valid_timestamps) == 0:
                return {'error': 'No valid timestamps found'}
            
            coverage = {
                'start_date': valid_timestamps.min().isoformat(),
                'end_date': valid_timestamps.max().isoformat(),
                'date_range_days': (valid_timestamps.max() - valid_timestamps.min()).days,
                'valid_percentage': (len(valid_timestamps) / len(data)) * 100,
                'temporal_gaps': self._find_temporal_gaps(valid_timestamps)
            }
            
            return coverage
            
        except Exception:
            return {'error': 'Could not parse timestamps'}
    
    def _find_temporal_gaps(self, timestamps: pd.Series) -> List[str]:
        """Find potential gaps in temporal data."""
        if len(timestamps) < 2:
            return []
        
        gaps = []
        sorted_timestamps = timestamps.sort_values()
        
        # Look for large gaps (more than 7 days)
        time_diffs = sorted_timestamps.diff()
        large_gaps = time_diffs[time_diffs > pd.Timedelta(days=7)]
        
        for gap in large_gaps.head(5):  # Report up to 5 largest gaps
            gaps.append(f"Gap of {gap.days} days found")
        
        return gaps
