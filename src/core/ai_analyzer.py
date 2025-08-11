"""AI analyzer for leveraging Azure OpenAI to analyze data structures and business context."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI


logger = logging.getLogger(__name__)


class AIIntegration:
    """Handles Azure OpenAI interactions."""
    
    def __init__(self):
        """Initialize the AIIntegration with Azure OpenAI client."""
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client with enhanced error reporting."""
        try:
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
            
            if not endpoint or not api_key:
                logger.warning("Azure OpenAI credentials not found. AI analysis will be disabled.")
                logger.warning(f"Missing: endpoint={not endpoint}, api_key={not api_key}")
                return
            
            logger.info(f"Initializing Azure OpenAI client with endpoint: {endpoint}")
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
            
            self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')
            logger.info(f"Azure OpenAI client initialized successfully with deployment: {self.deployment_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            self.client = None
    
    def call_openai(self, prompt: str) -> str:
        """Call Azure OpenAI with the given prompt."""
        if not self.client:
            logger.warning("AI analysis skipped - client not available")
            return ""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=3000  # Increased for detailed business analysis
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Azure OpenAI API call failed: {e}")
            logger.error(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
            logger.error(f"Deployment: {self.deployment_name}")
            return ""

class EventAnalyzer:
    """Handles event identification and readiness assessment."""
    
    def identify_event_candidates(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify complete event structures (activity + timestamp pairs)."""
        event_candidates = []
        
        for dataset in datasets:
            if dataset.get('data') is None:
                continue
                
            columns = list(dataset['data'].columns)
            table_name = dataset.get('file_path', 'unknown_table')
            
            # Find potential case IDs
            case_id_candidates = self._find_case_id_columns(columns)
            
            # Find activity-timestamp pairs
            activity_columns = self._find_activity_columns(columns)
            timestamp_columns = self._find_timestamp_columns(columns)
            
            # Create event candidates by pairing activities with timestamps
            for activity_col in activity_columns:
                for timestamp_col in timestamp_columns:
                    for case_id_col in case_id_candidates:
                        
                        # Calculate confidence based on column quality and data coverage
                        confidence = self._calculate_event_confidence(
                            dataset['data'], case_id_col, activity_col, timestamp_col
                        )
                        
                        if confidence > 0.3:  # Only include viable event candidates
                            event_candidate = {
                                'table': table_name,
                                'case_id_column': case_id_col,
                                'activity_column': activity_col,
                                'timestamp_column': timestamp_col,
                                'confidence': confidence,
                                'event_description': f"{activity_col} changes tracked with {timestamp_col}",
                                'business_meaning': self._infer_event_meaning(activity_col, timestamp_col),
                                'sample_events': self._extract_sample_events(
                                    dataset['data'], case_id_col, activity_col, timestamp_col
                                )
                            }
                            event_candidates.append(event_candidate)
        
        # Sort by confidence and return top candidates
        return sorted(event_candidates, key=lambda x: x['confidence'], reverse=True)[:10]

    def _find_case_id_columns(self, columns: List[str]) -> List[str]:
        """Find potential case ID columns."""
        case_id_indicators = ['id', 'number', 'key', 'reference', 'case', 'ticket', 'order', 'event']
        candidates = []
        
        for col in columns:
            col_lower = str(col).lower()
            for indicator in case_id_indicators:
                if indicator in col_lower and 'name' not in col_lower and 'description' not in col_lower:
                    candidates.append(col)
                    break
        
        return candidates

    def _find_activity_columns(self, columns: List[str]) -> List[str]:
        """Find potential activity columns."""
        activity_indicators = ['status', 'state', 'activity', 'event', 'action', 'step', 'phase', 'stage']
        candidates = []
        
        for col in columns:
            col_lower = str(col).lower()
            for indicator in activity_indicators:
                if indicator in col_lower:
                    candidates.append(col)
                    break
        
        return candidates

    def _find_timestamp_columns(self, columns: List[str]) -> List[str]:
        """Find potential timestamp columns."""
        timestamp_indicators = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 'changed']
        candidates = []
        
        for col in columns:
            col_lower = str(col).lower()
            for indicator in timestamp_indicators:
                if indicator in col_lower:
                    candidates.append(col)
                    break
        
        return candidates

    def _calculate_event_confidence(self, data, case_id_col: str, activity_col: str, timestamp_col: str) -> float:
        """Calculate confidence score for an event candidate."""
        try:
            # Check data availability
            case_coverage = data[case_id_col].notna().sum() / len(data)
            activity_coverage = data[activity_col].notna().sum() / len(data)
            timestamp_coverage = data[timestamp_col].notna().sum() / len(data)
            
            # Check uniqueness and variety
            case_uniqueness = data[case_id_col].nunique() / len(data) if len(data) > 0 else 0
            activity_variety = data[activity_col].nunique() / len(data) if len(data) > 0 else 0
            
            # Calculate weighted confidence
            confidence = (
                case_coverage * 0.3 +           # Case ID coverage
                activity_coverage * 0.3 +       # Activity coverage  
                timestamp_coverage * 0.3 +      # Timestamp coverage
                min(case_uniqueness * 2, 1) * 0.1  # Case ID uniqueness (capped at 1)
            )
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.1

    def _extract_sample_events(self, data, case_id_col: str, activity_col: str, timestamp_col: str) -> List[Dict[str, str]]:
        """Extract sample events to demonstrate the event structure."""
        try:
            # Get complete records only
            complete_data = data.dropna(subset=[case_id_col, activity_col, timestamp_col])
            
            if complete_data.empty:
                return []
            
            # Take first few complete records
            samples = []
            for _, row in complete_data.head(3).iterrows():
                samples.append({
                    'case_id': str(row[case_id_col]),
                    'activity': str(row[activity_col]),
                    'timestamp': str(row[timestamp_col])
                })
            
            return samples
            
        except Exception:
            return []

    def _infer_event_meaning(self, activity_col: str, timestamp_col: str) -> str:
        """Infer business meaning of the event structure."""
        activity_lower = activity_col.lower()
        timestamp_lower = timestamp_col.lower()
        
        if 'status' in activity_lower:
            return f"Process status transitions tracked by {timestamp_col}"
        elif 'event' in activity_lower:
            return f"Business events logged with {timestamp_col}"
        elif 'activity' in activity_lower:
            return f"Process activities executed at {timestamp_col}"
        else:
            return f"Process steps in {activity_col} with timing from {timestamp_col}"

    def identify_event_candidates(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify complete event structures (activity + timestamp pairs)."""
        event_candidates = []
        
        for dataset in datasets:
            if dataset.get('data') is None:
                continue
                
            columns = list(dataset['data'].columns)
            table_name = dataset.get('file_path', 'unknown_table')
            
            # Find potential case IDs
            case_id_candidates = self._find_case_id_columns(columns)
            
            # Find activity-timestamp pairs
            activity_columns = self._find_activity_columns(columns)
            timestamp_columns = self._find_timestamp_columns(columns)
            
            # Create event candidates by pairing activities with timestamps
            for activity_col in activity_columns:
                for timestamp_col in timestamp_columns:
                    for case_id_col in case_id_candidates:
                        
                        # Calculate confidence based on column quality and data coverage
                        confidence = self._calculate_event_confidence(
                            dataset['data'], case_id_col, activity_col, timestamp_col
                        )
                        
                        if confidence > 0.3:  # Only include viable event candidates
                            event_candidate = {
                                'table': table_name,
                                'case_id_column': case_id_col,
                                'activity_column': activity_col,
                                'timestamp_column': timestamp_col,
                                'confidence': confidence,
                                'event_description': f"{activity_col} changes tracked with {timestamp_col}",
                                'business_meaning': self._infer_event_meaning(activity_col, timestamp_col),
                                'sample_events': self._extract_sample_events(
                                    dataset['data'], case_id_col, activity_col, timestamp_col
                                )
                            }
                            event_candidates.append(event_candidate)
        
        # Sort by confidence and return top candidates
        return sorted(event_candidates, key=lambda x: x['confidence'], reverse=True)[:10]

    def _find_case_id_columns(self, columns: List[str]) -> List[str]:
        """Find potential case ID columns."""
        case_id_indicators = ['id', 'number', 'key', 'reference', 'case', 'ticket', 'order', 'event']
        candidates = []
        
        for col in columns:
            col_lower = str(col).lower()
            for indicator in case_id_indicators:
                if indicator in col_lower and 'name' not in col_lower and 'description' not in col_lower:
                    candidates.append(col)
                    break
        
        return candidates

    def _find_activity_columns(self, columns: List[str]) -> List[str]:
        """Find potential activity columns."""
        activity_indicators = ['status', 'state', 'activity', 'event', 'action', 'step', 'phase', 'stage']
        candidates = []
        
        for col in columns:
            col_lower = str(col).lower()
            for indicator in activity_indicators:
                if indicator in col_lower:
                    candidates.append(col)
                    break
        
        return candidates

    def _find_timestamp_columns(self, columns: List[str]) -> List[str]:
        """Find potential timestamp columns."""
        timestamp_indicators = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 'changed']
        candidates = []
        
        for col in columns:
            col_lower = str(col).lower()
            for indicator in timestamp_indicators:
                if indicator in col_lower:
                    candidates.append(col)
                    break
        
        return candidates

    def _calculate_event_confidence(self, data, case_id_col: str, activity_col: str, timestamp_col: str) -> float:
        """Calculate confidence score for an event candidate."""
        try:
            # Check data availability
            case_coverage = data[case_id_col].notna().sum() / len(data)
            activity_coverage = data[activity_col].notna().sum() / len(data)
            timestamp_coverage = data[timestamp_col].notna().sum() / len(data)
            
            # Check uniqueness and variety
            case_uniqueness = data[case_id_col].nunique() / len(data) if len(data) > 0 else 0
            activity_variety = data[activity_col].nunique() / len(data) if len(data) > 0 else 0
            
            # Calculate weighted confidence
            confidence = (
                case_coverage * 0.3 +           # Case ID coverage
                activity_coverage * 0.3 +       # Activity coverage  
                timestamp_coverage * 0.3 +      # Timestamp coverage
                min(case_uniqueness * 2, 1) * 0.1  # Case ID uniqueness (capped at 1)
            )
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.1

    def _extract_sample_events(self, data, case_id_col: str, activity_col: str, timestamp_col: str) -> List[Dict[str, str]]:
        """Extract sample events to demonstrate the event structure."""
        try:
            # Get complete records only
            complete_data = data.dropna(subset=[case_id_col, activity_col, timestamp_col])
            
            if complete_data.empty:
                return []
            
            # Take first few complete records
            samples = []
            for _, row in complete_data.head(3).iterrows():
                samples.append({
                    'case_id': str(row[case_id_col]),
                    'activity': str(row[activity_col]),
                    'timestamp': str(row[timestamp_col])
                })
            
            return samples
            
        except Exception:
            return []

    def _infer_event_meaning(self, activity_col: str, timestamp_col: str) -> str:
        """Infer business meaning of the event structure."""
        activity_lower = activity_col.lower()
        timestamp_lower = timestamp_col.lower()
        
        if 'status' in activity_lower:
            return f"Process status transitions tracked by {timestamp_col}"
        elif 'event' in activity_lower:
            return f"Business events logged with {timestamp_col}"
        elif 'activity' in activity_lower:
            return f"Process activities executed at {timestamp_col}"
        else:
            return f"Process steps in {activity_col} with timing from {timestamp_col}"

    def assess_readiness(self, event_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess readiness based on complete event availability."""
        
        if not event_candidates:
            return {
                'overall_score': 0,
                'event_completeness': 0,
                'temporal_coverage': 0,
                'case_id_quality': 0,
                'reasoning': 'No complete events (activity+timestamp pairs) found'
            }
        
        # Calculate scores based on best event candidate
        best_event = event_candidates[0]
        
        event_completeness = best_event['confidence'] * 10
        temporal_coverage = len([e for e in event_candidates if e['confidence'] > 0.5]) * 2
        case_id_quality = min(len([e for e in event_candidates if 'id' in e['case_id_column'].lower()]) * 3, 10)
        
        overall_score = (event_completeness + temporal_coverage + case_id_quality) / 3
        
        return {
            'overall_score': round(overall_score, 1),
            'event_completeness': round(event_completeness, 1),
            'temporal_coverage': round(min(temporal_coverage, 10), 1),
            'case_id_quality': round(case_id_quality, 1),
            'reasoning': f'Based on {len(event_candidates)} complete event structures found'
        }

class SQLGenerator:
    """Handles SQL generation."""
    
    @staticmethod
    def generate_event_sql(event_candidate: Dict[str, Any]) -> str:
        """Generate SQL for the best event candidate."""
        if not event_candidate:
            return "-- No complete events found - cannot generate event log SQL\n-- CRITICAL: Activities must have associated timestamps for process mining"
        
        table_name = event_candidate['table'].split('/')[-1].split('.')[0].replace(' ', '_')
        
        sql = f"""-- Event Log SQL - Process Mining Ready
-- Generated from: {event_candidate['table']}
-- Event Structure: {event_candidate['event_description']}

SELECT 
    [{event_candidate['case_id_column']}] as case_id,
    [{event_candidate['activity_column']}] as activity,
    [{event_candidate['timestamp_column']}] as timestamp,
    -- Add additional attributes as needed
    ROW_NUMBER() OVER (
        PARTITION BY [{event_candidate['case_id_column']}] 
        ORDER BY [{event_candidate['timestamp_column']}]
    ) as event_sequence
FROM [{table_name}]
WHERE [{event_candidate['case_id_column']}] IS NOT NULL
  AND [{event_candidate['activity_column']}] IS NOT NULL  
  AND [{event_candidate['timestamp_column']}] IS NOT NULL
ORDER BY case_id, timestamp;

-- Sample Events Found:
{chr(10).join([f"-- {event['case_id']} | {event['activity']} | {event['timestamp']}" for event in event_candidate.get('sample_events', [])])}

-- Confidence: {event_candidate['confidence']:.1%}
-- Business Meaning: {event_candidate['business_meaning']}"""
        
        return sql

class ProcessInference:
    """Handles process type detection."""
    
    @staticmethod
    def infer_process_type(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer business process type from data patterns."""
        process_indicators = {
            'incident_management': ['event', 'incident', 'ticket', 'status', 'priority', 'assigned', 'mis', 'service'],
            'order_management': ['order', 'customer', 'product', 'quantity', 'ship', 'invoice', 'delivery'],
            'procurement': ['purchase', 'vendor', 'supplier', 'contract', 'approval', 'procurement'],
            'hr_process': ['employee', 'hire', 'performance', 'leave', 'payroll', 'training'],
            'finance': ['payment', 'invoice', 'account', 'transaction', 'budget', 'billing'],
            'manufacturing': ['production', 'machine', 'quality', 'batch', 'manufacturing'],
            'logistics': ['shipment', 'warehouse', 'transport', 'delivery', 'logistics']
        }
        
        scores = {}
        for process_type, keywords in process_indicators.items():
            score = 0
            evidence = []
            
            for dataset in datasets:
                # Check column names
                if dataset.get('data') is not None:
                    columns = [str(col).lower() for col in dataset['data'].columns]
                    for keyword in keywords:
                        matching_cols = [col for col in columns if keyword in col]
                        if matching_cols:
                            score += len(matching_cols)
                            evidence.extend(matching_cols)
                
                # Check file path
                file_path = dataset.get('file_path', '').lower()
                for keyword in keywords:
                    if keyword in file_path:
                        score += 2  # File name matches are weighted higher
                        evidence.append(f"filename: {keyword}")
        
            scores[process_type] = {'score': score, 'evidence': evidence }
        
        best_match = max(scores, key=lambda k: scores[k]['score']) if scores else 'unknown'
        best_score = scores[best_match]['score'] if best_match in scores else 0
        confidence = min(best_score / 8.0, 1.0) if best_score > 0 else 0.1
        
        process_name = best_match.replace('_', ' ').title()
        evidence_text = ', '.join(scores[best_match]['evidence'][:5]) if best_match in scores else 'no clear indicators'
        
        return {
            'process_type': process_name,
            'confidence': confidence,
            'reasoning': f"Detected {best_score} indicators for {process_name.lower()} process. Evidence: {evidence_text}"
        }

class AIAnalyzer:
    """Main class orchestrating AI analysis and fallback logic."""
    
    def __init__(self):
        """Initialize the AIAnalyzer components."""
        self.ai_integration = AIIntegration()
        self.event_analyzer = EventAnalyzer()
        self.sql_generator = SQLGenerator()
        self.process_inference = ProcessInference()

    def analyze_for_process_mining(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]] = None,
        business_context: str = ""
    ) -> Dict[str, Any]:
        """Analyze datasets and schema for process mining insights using AI.
        
        Args:
            datasets: List of dataset information
            schema_info: Parsed schema information
            business_context: Business context description
            
        Returns:
            AI-generated insights and recommendations
        """
        if not self.ai_integration.client:
            logger.warning("AI analysis skipped - client not available")
            return self.generate_fallback_analysis(datasets, schema_info, business_context)
        
        try:
            # Prepare context for AI analysis
            analysis_context = self._prepare_analysis_context(datasets, schema_info, business_context)
            
            # Generate AI insights
            ai_response = self.ai_integration.call_openai(analysis_context)
            
            # Parse and structure the response
            insights = self._parse_ai_response(ai_response)
            
            logger.info("AI analysis completed successfully")
            return insights
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self.generate_fallback_analysis(datasets, schema_info, business_context)
    
    def _prepare_analysis_context(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        business_context: str
    ) -> str:
        """Prepare context string for AI analysis."""
        
        context_parts = []
        
        # Add business context
        if business_context:
            context_parts.append(f"BUSINESS CONTEXT:\n{business_context}\n")
        
        # Add dataset information
        context_parts.append("AVAILABLE DATASETS:")
        for i, dataset in enumerate(datasets, 1):
            metadata = dataset.get('metadata', {})
            context_parts.append(f"\nDataset {i}: {dataset['file_path']}")
            context_parts.append(f"  - Shape: {metadata.get('shape', {})}")
            context_parts.append(f"  - Columns: {list(metadata.get('columns', {}).keys())}")
            context_parts.append(f"  - Potential identifiers: {metadata.get('potential_identifiers', [])}")
            context_parts.append(f"  - Potential timestamps: {metadata.get('potential_timestamps', [])}")
            context_parts.append(f"  - Potential activities: {metadata.get('potential_activities', [])}")
            
            # Add sample data
            if 'data' in dataset and not dataset['data'].empty:
                sample_data = dataset['data'].head(3).to_string()
                context_parts.append(f"  - Sample data:\n{sample_data}")
        
        # Add schema information
        if schema_info:
            context_parts.append(f"\nDATABASE SCHEMA:")
            context_parts.append(f"  - Type: {schema_info.get('type', 'unknown')}")
            
            if 'tables' in schema_info:
                context_parts.append(f"  - Tables: {list(schema_info['tables'].keys())}")
                for table_name, table_info in schema_info['tables'].items():
                    context_parts.append(f"    * {table_name}: {list(table_info.get('columns', {}).keys())}")
        
        return "\n".join(context_parts)
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured insights."""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response[json_start:json_end]
                parsed_response = json.loads(json_content)
                
                # Add metadata
                parsed_response['ai_analysis'] = True
                parsed_response['raw_response'] = response
                
                return parsed_response
            else:
                # Fallback if JSON parsing fails
                return {
                    'ai_analysis': True,
                    'raw_response': response,
                    'parsing_error': 'Could not extract JSON from response',
                    'recommendations': [response]  # Use raw response as recommendation
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            return {
                'ai_analysis': True,
                'raw_response': response,
                'parsing_error': str(e),
                'recommendations': [response]
            }
    
    def generate_fallback_analysis(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        business_context: str
    ) -> Dict[str, Any]:
        """Generate event-centric fallback analysis when AI unavailable."""
        
        logger.info("Generating event-centric fallback analysis with business intelligence")
        
        event_candidates = self.event_analyzer.identify_event_candidates(datasets)
        readiness = self.event_analyzer.assess_readiness(event_candidates)
        process_type = self.process_inference.infer_process_type(datasets)
        sql = self.sql_generator.generate_event_sql(event_candidates[0]) if event_candidates else ""
        
        fallback_insights = {
            'ai_analysis': False,
            'fallback_analysis': True,
            'business_process_analysis': {
                'process_type': process_type['process_type'],
                'confidence': process_type['confidence'],
                'reasoning': process_type['reasoning']
            },
            'event_candidates': event_candidates,
            'readiness_assessment': {
                'overall_score': readiness['overall_score'],
                'event_completeness': readiness['event_completeness'],
                'temporal_coverage': readiness['temporal_coverage'],
                'case_id_quality': readiness['case_id_quality'],
                'reasoning': readiness['reasoning']
            },
            'actionable_recommendations': self._generate_event_recommendations(event_candidates, readiness),
            'event_log_sql': sql,
            'business_value': f"Process mining these {process_type['process_type'].lower()} events will provide insights into process efficiency and bottlenecks.",
            'data_quality_concerns': self._identify_event_quality_issues(event_candidates, datasets),
            'process_mining_potential': self._assess_mining_potential_events(readiness)
        }
        
        return fallback_insights

    def _generate_event_recommendations(self, event_candidates: List[Dict[str, Any]], readiness: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on event analysis."""
        recommendations = []
        
        if not event_candidates:
            recommendations.extend([
                "CRITICAL: No complete events found - activities must have associated timestamps",
                "Identify data sources that contain both process activities and their execution times",
                "Request event logging enhancement from system administrators"
            ])
        else:
            best_event = event_candidates[0]
            
            # Event-specific recommendations
            if best_event['confidence'] < 0.7:
                recommendations.append(f"Improve data quality for {best_event['activity_column']} and {best_event['timestamp_column']} pairing")
            
            recommendations.append(f"Validate that {best_event['activity_column']} represents meaningful business activities")
            recommendations.append(f"Confirm {best_event['timestamp_column']} captures actual event timing")
            
            # Add sample events validation
            if best_event.get('sample_events'):
                recommendations.append("Review sample events with business stakeholders for accuracy")
        
        recommendations.extend([
            "Ensure complete case lifecycle coverage from start to end",
            "Consider data privacy requirements for process mining analysis"
        ])
        
        return recommendations[:8]

    def _identify_event_quality_issues(self, event_candidates: List[Dict[str, Any]], datasets: List[Dict[str, Any]]) -> List[str]:
        """Identify specific data quality concerns for events."""
        issues = []
        
        if not event_candidates:
            issues.append("CRITICAL: No complete events (activity+timestamp pairs) found")
            issues.append("Cannot perform process mining without temporal activity data")
        else:
            for event in event_candidates[:3]:  # Check top 3 candidates
                if event['confidence'] < 0.5:
                    issues.append(f"Low confidence event structure: {event['event_description']} ({event['confidence']:.1%})")
                
                if not event.get('sample_events'):
                    issues.append(f"No sample events found for {event['activity_column']} + {event['timestamp_column']}")
        
        # Add general data quality issues
        for dataset in datasets:
            if dataset.get('data') is None:
                continue
                
            file_name = dataset.get('file_path', 'dataset').split('/')[-1] if '/' in dataset.get('file_path', '') else 'dataset'
            
            # Check for high missing data
            if hasattr(dataset['data'], 'isnull'):
                missing_pct = (dataset['data'].isnull().sum().sum() / (dataset['data'].shape[0] * dataset['data'].shape[1])) * 100
                if missing_pct > 30:
                    issues.append(f"High missing data in {file_name}: {missing_pct:.1f}%")
        
        return issues

    def _assess_mining_potential_events(self, readiness: Dict[str, Any]) -> str:
        """Assess overall process mining potential based on events."""
        score = readiness['overall_score']
        
        if score >= 7:
            return "High - Complete event structures found, ready for process mining"
        elif score >= 4:
            return "Medium - Event structures identified but need quality improvements"
        else:
            return "Low - No viable event structures found, fundamental data collection needed"

    def _infer_business_process(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer business process type from data patterns."""
        process_indicators = {
            'incident_management': ['event', 'incident', 'ticket', 'status', 'priority', 'assigned', 'mis', 'service'],
            'order_management': ['order', 'customer', 'product', 'quantity', 'ship', 'invoice', 'delivery'],
            'procurement': ['purchase', 'vendor', 'supplier', 'contract', 'approval', 'procurement'],
            'hr_process': ['employee', 'hire', 'performance', 'leave', 'payroll', 'training'],
            'finance': ['payment', 'invoice', 'account', 'transaction', 'budget', 'billing'],
            'manufacturing': ['production', 'machine', 'quality', 'batch', 'manufacturing'],
            'logistics': ['shipment', 'warehouse', 'transport', 'delivery', 'logistics']
        }
        
        scores = {}
        for process_type, keywords in process_indicators.items():
            score = 0
            evidence = []
            
            for dataset in datasets:
                # Check column names
                if dataset.get('data') is not None:
                    columns = [str(col).lower() for col in dataset['data'].columns]
                    for keyword in keywords:
                        matching_cols = [col for col in columns if keyword in col]
                        if matching_cols:
                            score += len(matching_cols)
                            evidence.extend(matching_cols)
                
                # Check file path
                file_path = dataset.get('file_path', '').lower()
                for keyword in keywords:
                    if keyword in file_path:
                        score += 2  # File name matches are weighted higher
                        evidence.append(f"filename: {keyword}")
        
            scores[process_type] = {'score': score, 'evidence': evidence }
        
        best_match = max(scores, key=lambda k: scores[k]['score']) if scores else 'unknown'
        best_score = scores[best_match]['score'] if best_match in scores else 0
        confidence = min(best_score / 8.0, 1.0) if best_score > 0 else 0.1
        
        process_name = best_match.replace('_', ' ').title()
        evidence_text = ', '.join(scores[best_match]['evidence'][:5]) if best_match in scores else 'no clear indicators'
        
        return {
            'process_type': process_name,
            'confidence': confidence,
            'reasoning': f"Detected {best_score} indicators for {process_name.lower()} process. Evidence: {evidence_text}"
        }

    def _assess_business_readiness(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess process mining readiness from business perspective."""
        
        case_id_score = 0
        activity_score = 0
        timestamp_score = 0
        data_completeness_score = 0
        
        total_datasets = len(datasets)
        if total_datasets == 0:
            return {'overall_score': 0, 'case_id_score': 0, 'activity_score': 0, 'timestamp_score': 0, 'data_completeness': 0, 'breakdown': 'No datasets available'}
        
        case_id_candidates = 0
        activity_candidates = 0
        timestamp_candidates = 0
        total_completeness = 0
        
        for dataset in datasets:
            if dataset.get('data') is None:
                continue
                
            columns = [str(col).lower() for col in dataset['data'].columns]
            
            # Check for case ID indicators
            case_id_indicators = ['id', 'number', 'key', 'reference', 'case']
            if any(indicator in col for col in columns for indicator in case_id_indicators):
                case_id_candidates += 1
            
            # Check for activity indicators  
            activity_indicators = ['status', 'state', 'activity', 'event', 'action', 'step']
            if any(indicator in col for col in columns for indicator in activity_indicators):
                activity_candidates += 1
            
            # Check for timestamp indicators
            timestamp_indicators = ['date', 'time', 'timestamp', 'created', 'updated', 'modified']
            if any(indicator in col for col in columns for indicator in timestamp_indicators):
                timestamp_candidates += 1
            
            # Calculate data completeness
            completeness = dataset['data'].notnull().mean().mean()
            total_completeness += completeness
        
        # Calculate individual scores
        case_id_score = min(case_id_candidates / total_datasets, 1.0) * 10
        activity_score = min(activity_candidates / total_datasets, 1.0) * 10
        timestamp_score = min(timestamp_candidates / total_datasets, 1.0) * 10
        data_completeness_score = min(total_completeness / total_datasets, 1.0) * 10
        
        overall_score = (case_id_score + activity_score + timestamp_score + data_completeness_score) / 4
        
        return {
            'overall_score': round(overall_score, 1),
            'case_id_score': round(case_id_score, 1),
            'activity_score': round(activity_score, 1),
            'timestamp_score': round(timestamp_score, 1),
            'data_completeness': round(data_completeness_score, 1),
            'breakdown': f"Case ID: {case_id_score}, Activity: {activity_score}, Timestamp: {timestamp_score}, Completeness: {data_completeness_score}",
            'reasoning': 'Based on analysis of available datasets and identified indicators'
        }

def filter_columns_by_indicators(columns, indicators):
    """Filter columns that match any of the given indicators."""
    return [col for col in columns if any(indicator in col.lower() for indicator in indicators)]
