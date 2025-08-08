"""AI analyzer for leveraging Azure OpenAI to analyze data structures and business context."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI


logger = logging.getLogger(__name__)


class AIAnalyzer:
    """Leverages Azure OpenAI to analyze data for process mining insights."""
    
    def __init__(self):
        """Initialize the AIAnalyzer with Azure OpenAI client."""
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
        if not self.client:
            logger.warning("AI analysis skipped - client not available")
            return self._generate_fallback_analysis(datasets, schema_info, business_context)
        
        try:
            # Prepare context for AI analysis
            analysis_context = self._prepare_analysis_context(datasets, schema_info, business_context)
            
            # Generate AI insights
            ai_response = self._call_azure_openai(analysis_context)
            
            # Parse and structure the response
            insights = self._parse_ai_response(ai_response)
            
            logger.info("AI analysis completed successfully")
            return insights
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._generate_fallback_analysis(datasets, schema_info, business_context)
    
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
    
    def _call_azure_openai(self, context: str) -> str:
        """Call Azure OpenAI API for analysis."""
        
        system_prompt = """You are a world-class process mining consultant with 15+ years of experience helping enterprises analyze their business processes. Your expertise includes:

- Business process identification (Order-to-Cash, Procure-to-Pay, Incident Management, HR Processes, Finance, etc.)
- Event log structure requirements (Case ID, Activity, Timestamp, Attributes)
- Data quality assessment for process mining readiness
- SQL generation for event log creation
- Business value articulation for executives

When analyzing data sources, provide insights that help business consultants understand:
1. What business process this data represents (with confidence level)
2. How ready this data is for process mining (scored 0-10 with clear reasoning)
3. Specific actionable steps to improve the data
4. Working SQL code that actually uses the detected column names
5. Business impact and value proposition

Always respond in JSON format with the following structure:
{
  "business_process_analysis": {
    "process_type": "e.g., Incident Management System",
    "confidence": 0.85,
    "reasoning": "Detailed explanation based on column names, file names, and data patterns"
  },
  "readiness_assessment": {
    "overall_score": 6,
    "case_id_quality": 8,
    "activity_quality": 7,
    "timestamp_quality": 2,
    "data_completeness": 5,
    "reasoning": "Clear explanation of each score with specific evidence"
  },
  "actionable_recommendations": [
    "Immediate action 1: Request timestamp data from system administrator",
    "Data quality fix 2: Address high missing data percentages",
    "Quick win 3: Use specific column for activity names"
  ],
  "working_sql": "-- Actual SQL using real column names\\nSELECT [actual_column_name] as case_id, [status_column] as activity FROM [table_name]...",
  "business_value": "Expected business outcomes from process mining this data",
  "next_steps": ["Concrete step 1", "Concrete step 2", "Concrete step 3"],
  "data_quality_concerns": ["Specific issue 1", "Specific issue 2"],
  "process_mining_potential": "High/Medium/Low with reasoning"
}

Be specific, actionable, and focus on business value rather than technical details. Use the actual column names and table references found in the data."""

        user_prompt = f"""Analyze the following data sources for process mining event log creation:

{context}

The user is a process mining consultant who needs to assess this data for their client. Provide practical, business-focused recommendations that they can immediately act upon. Focus on identifying the business process type and providing actionable next steps."""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=3000  # Increased for detailed business analysis
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Azure OpenAI API call failed: {e}")
            logger.error(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
            logger.error(f"Deployment: {self.deployment_name}")
            raise
    
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
    
    def _generate_fallback_analysis(
        self,
        datasets: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]],
        business_context: str
    ) -> Dict[str, Any]:
        """Generate enhanced fallback analysis with business intelligence when AI unavailable."""
        
        logger.info("Generating enhanced fallback analysis with business intelligence")
        
        # Analyze data patterns to infer business process type
        process_inference = self._infer_business_process(datasets)
        
        # Generate business-focused readiness assessment
        readiness_assessment = self._assess_business_readiness(datasets)
        
        # Create actionable recommendations
        recommendations = self._generate_business_recommendations(datasets, process_inference)
        
        fallback_insights = {
            'ai_analysis': False,
            'fallback_analysis': True,
            'business_process_analysis': {
                'process_type': process_inference['process_type'],
                'confidence': process_inference['confidence'],
                'reasoning': process_inference['reasoning']
            },
            'readiness_assessment': {
                'overall_score': readiness_assessment['overall_score'],
                'case_id_quality': readiness_assessment['case_id_score'],
                'activity_quality': readiness_assessment['activity_score'], 
                'timestamp_quality': readiness_assessment['timestamp_score'],
                'data_completeness': readiness_assessment['data_completeness'],
                'reasoning': readiness_assessment['breakdown']
            },
            'actionable_recommendations': recommendations,
            'working_sql': self._generate_working_sql_fallback(datasets),
            'business_value': f"Process mining this {process_inference['process_type'].lower()} data could provide insights into process efficiency, bottlenecks, and improvement opportunities.",
            'next_steps': self._generate_next_steps(datasets, readiness_assessment),
            'data_quality_concerns': self._identify_data_quality_issues(datasets),
            'process_mining_potential': self._assess_mining_potential(readiness_assessment)
        }
        
        return fallback_insights

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
        
            scores[process_type] = {'score': score, 'evidence': evidence}
        
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
            if hasattr(dataset['data'], 'isnull'):
                completeness = (1 - dataset['data'].isnull().sum().sum() / (dataset['data'].shape[0] * dataset['data'].shape[1])) * 100
                total_completeness += completeness
        
        # Calculate scores (0-10 scale)
        case_id_score = min((case_id_candidates / total_datasets) * 10, 10)
        activity_score = min((activity_candidates / total_datasets) * 10, 10)
        timestamp_score = min((timestamp_candidates / total_datasets) * 10, 10)
        data_completeness_score = min((total_completeness / total_datasets) / 10, 10) if total_datasets > 0 else 0
        
        overall_score = (case_id_score + activity_score + timestamp_score + data_completeness_score) / 4
        
        breakdown = f"Case ID: {case_id_score:.1f}/10, Activity: {activity_score:.1f}/10, Timestamp: {timestamp_score:.1f}/10, Completeness: {data_completeness_score:.1f}/10"
        
        return {
            'overall_score': round(overall_score, 1),
            'case_id_score': round(case_id_score, 1),
            'activity_score': round(activity_score, 1),
            'timestamp_score': round(timestamp_score, 1),
            'data_completeness': round(data_completeness_score, 1),
            'breakdown': breakdown
        }

    def _generate_business_recommendations(self, datasets: List[Dict[str, Any]], process_inference: Dict[str, Any]) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        # Process-specific recommendations
        if 'incident' in process_inference['process_type'].lower():
            recommendations.append("Focus on incident lifecycle: Created → Assigned → In Progress → Resolved → Closed")
            recommendations.append("Identify key performance indicators: Resolution time, escalation patterns, customer impact")
        elif 'order' in process_inference['process_type'].lower():
            recommendations.append("Map order-to-cash flow: Order → Payment → Fulfillment → Delivery → Invoice")
            recommendations.append("Track order processing efficiency and customer satisfaction metrics")
        
        # Data quality recommendations
        has_timestamps = any('date' in str(col).lower() or 'time' in str(col).lower() 
                           for dataset in datasets if dataset.get('data') is not None 
                           for col in dataset['data'].columns)
        
        if not has_timestamps:
            recommendations.append("CRITICAL: Request timestamp data from system administrator - process mining requires temporal information")
        
        # Missing data recommendations
        for dataset in datasets:
            if dataset.get('data') is not None and hasattr(dataset['data'], 'isnull'):
                missing_pct = (dataset['data'].isnull().sum().sum() / (dataset['data'].shape[0] * dataset['data'].shape[1])) * 100
                if missing_pct > 50:
                    file_name = dataset.get('file_path', 'dataset').split('/')[-1] if '/' in dataset.get('file_path', '') else dataset.get('file_path', 'dataset')
                    recommendations.append(f"Address high missing data in {file_name} ({missing_pct:.1f}% missing)")
        
        # Always include these general recommendations
        recommendations.extend([
            "Validate business meaning of identified Case IDs with domain experts",
            "Confirm activity names represent meaningful business events",
            "Consider data privacy and compliance requirements for process mining"
        ])
        
        return recommendations[:8]  # Limit to top 8 recommendations

    def _generate_working_sql_fallback(self, datasets: List[Dict[str, Any]]) -> str:
        """Generate working SQL based on actual detected columns."""
        if not datasets:
            return "-- No datasets available for SQL generation"
        
        # Find the dataset with the most promising structure
        best_dataset = None
        best_score = 0
        
        for dataset in datasets:
            if dataset.get('data') is None:
                continue
                
            columns = [str(col).lower() for col in dataset['data'].columns]
            score = 0
            
            # Score based on useful columns
            useful_indicators = ['id', 'status', 'date', 'event', 'activity']
            for indicator in useful_indicators:
                if any(indicator in col for col in columns):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_dataset = dataset
        
        if not best_dataset:
            return "-- No suitable dataset found for SQL generation"
        
        columns = list(best_dataset['data'].columns)
        
        # Find best columns for each purpose
        case_id_col = self._find_best_column(columns, ['id', 'number', 'key', 'event id'])
        activity_col = self._find_best_column(columns, ['status', 'state', 'activity'])
        timestamp_col = self._find_best_column(columns, ['date', 'time', 'created', 'updated'])
        
        # Generate table name from file path
        file_path = best_dataset.get('file_path', 'data_table')
        table_name = file_path.split('/')[-1].split('.')[0] if '/' in file_path else file_path.split('.')[0]
        table_name = table_name.replace(' ', '_').replace('-', '_')
        
        sql_parts = []
        sql_parts.append("-- Process Mining Event Log SQL")
        sql_parts.append(f"-- Generated from: {best_dataset.get('file_path', 'unknown')}")
        sql_parts.append(f"-- Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sql_parts.append("")
        sql_parts.append("SELECT")
        
        if case_id_col:
            sql_parts.append(f"    [{case_id_col}] as case_id,")
        else:
            sql_parts.append("    -- No clear case ID found - manual review required")
            sql_parts.append("    [column_name] as case_id,  -- TODO: Replace with actual case ID column")
        
        if activity_col:
            sql_parts.append(f"    [{activity_col}] as activity,")
        else:
            sql_parts.append("    -- No clear activity found - manual review required")
            sql_parts.append("    [column_name] as activity,  -- TODO: Replace with actual activity column")
        
        if timestamp_col:
            sql_parts.append(f"    [{timestamp_col}] as timestamp,")
        else:
            sql_parts.append("    -- No timestamp found - consider adding sequence number")
            sql_parts.append("    ROW_NUMBER() OVER (PARTITION BY case_id ORDER BY case_id) as event_sequence,")
        
        # Add other potentially useful columns
        other_cols = [col for col in columns[:5] if col not in [case_id_col, activity_col, timestamp_col]]
        for col in other_cols:
            sql_parts.append(f"    [{col}],")
        
        # Remove last comma
        if sql_parts[-1].endswith(','):
            sql_parts[-1] = sql_parts[-1][:-1]
        
        sql_parts.append(f"FROM [{table_name}]")
        
        if case_id_col:
            sql_parts.append(f"WHERE [{case_id_col}] IS NOT NULL")
        
        if timestamp_col:
            sql_parts.append(f"ORDER BY case_id, [{timestamp_col}];")
        else:
            sql_parts.append("ORDER BY case_id;")
        
        sql_parts.append("")
        sql_parts.append("-- Next Steps:")
        sql_parts.append("-- 1. Validate column mappings with business users")
        sql_parts.append("-- 2. Add data quality filters as needed")
        sql_parts.append("-- 3. Consider additional case/event attributes")
        
        return "\n".join(sql_parts)

    def _find_best_column(self, columns: List[str], indicators: List[str]) -> Optional[str]:
        """Find the best column matching the given indicators."""
        for indicator in indicators:
            for col in columns:
                if indicator.lower() in col.lower():
                    return col
        return None

    def _generate_next_steps(self, datasets: List[Dict[str, Any]], readiness: Dict[str, Any]) -> List[str]:
        """Generate concrete next steps based on readiness assessment."""
        steps = []
        
        if readiness['timestamp_score'] < 3:
            steps.append("Request historical timestamp data from system administrators")
        
        if readiness['case_id_score'] < 5:
            steps.append("Clarify business definition of case/process instances with domain experts")
        
        if readiness['activity_score'] < 5:
            steps.append("Map business activities to data fields with process owners")
        
        if readiness['data_completeness'] < 5:
            steps.append("Investigate and address data quality issues with IT team")
        
        steps.extend([
            "Validate initial findings with business stakeholders",
            "Plan proof-of-concept process mining analysis",
            "Define success metrics and business value targets"
        ])
        
        return steps[:5]  # Limit to top 5 steps

    def _identify_data_quality_issues(self, datasets: List[Dict[str, Any]]) -> List[str]:
        """Identify specific data quality concerns."""
        issues = []
        
        for dataset in datasets:
            if dataset.get('data') is None:
                continue
                
            file_name = dataset.get('file_path', 'dataset').split('/')[-1] if '/' in dataset.get('file_path', '') else 'dataset'
            
            # Check for high missing data
            if hasattr(dataset['data'], 'isnull'):
                missing_pct = (dataset['data'].isnull().sum().sum() / (dataset['data'].shape[0] * dataset['data'].shape[1])) * 100
                if missing_pct > 30:
                    issues.append(f"High missing data in {file_name}: {missing_pct:.1f}%")
            
            # Check for unnamed columns
            unnamed_cols = [col for col in dataset['data'].columns if 'unnamed' in str(col).lower()]
            if unnamed_cols:
                issues.append(f"Unnamed columns in {file_name}: {len(unnamed_cols)} columns need proper names")
        
        return issues

    def _assess_mining_potential(self, readiness: Dict[str, Any]) -> str:
        """Assess overall process mining potential."""
        score = readiness['overall_score']
        
        if score >= 7:
            return "High - Good foundation for immediate process mining analysis"
        elif score >= 4:
            return "Medium - Requires data improvements but viable with effort"
        else:
            return "Low - Significant data quality issues need resolution first"
