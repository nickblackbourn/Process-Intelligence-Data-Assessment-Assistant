"""AI analyzer for leveraging Azure OpenAI to analyze data structures and business context."""

import json
import logging
import os
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
        """Initialize Azure OpenAI client."""
        try:
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
            
            if not endpoint or not api_key:
                logger.warning("Azure OpenAI credentials not found. AI analysis will be disabled.")
                return
            
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
            
            self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')
            logger.info("Azure OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
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
        
        system_prompt = """You are an expert process mining consultant. Your task is to analyze data sources and provide recommendations for creating high-quality event logs for process mining.

Focus on:
1. Identifying the best Case ID candidates (unique process instance identifiers)
2. Identifying Activity columns (what activities/events occurred)
3. Identifying Timestamp columns (when activities occurred)
4. Recommending key attributes for case and event data
5. Suggesting data transformations or aggregations needed
6. Identifying potential data quality issues
7. Providing specific recommendations for process mining success

Respond in JSON format with the following structure:
{
  "case_id_analysis": {
    "primary_recommendation": {"table": "...", "column": "...", "confidence": 0.9, "reasoning": "..."},
    "alternatives": [{"table": "...", "column": "...", "confidence": 0.7, "reasoning": "..."}]
  },
  "activity_analysis": {
    "primary_recommendation": {"table": "...", "column": "...", "confidence": 0.9, "reasoning": "..."},
    "alternatives": [...],
    "aggregation_suggestions": ["suggestion1", "suggestion2"]
  },
  "timestamp_analysis": {
    "primary_recommendation": {"table": "...", "column": "...", "confidence": 0.9, "reasoning": "..."},
    "alternatives": [...]
  },
  "attribute_recommendations": {
    "case_attributes": [{"table": "...", "column": "...", "value": "...", "reasoning": "..."}],
    "event_attributes": [{"table": "...", "column": "...", "value": "...", "reasoning": "..."}]
  },
  "data_quality_concerns": ["concern1", "concern2"],
  "transformation_recommendations": ["transformation1", "transformation2"],
  "process_mining_readiness": {"score": 0.8, "reasoning": "..."},
  "next_steps": ["step1", "step2", "step3"]
}"""

        user_prompt = f"""Analyze the following data sources for process mining event log creation:

{context}

Please provide detailed recommendations for creating a high-quality event log from this data."""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Azure OpenAI API call failed: {e}")
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
        """Generate basic analysis when AI is not available."""
        
        logger.info("Generating fallback analysis without AI")
        
        fallback_insights = {
            'ai_analysis': False,
            'case_id_analysis': {
                'primary_recommendation': None,
                'alternatives': []
            },
            'activity_analysis': {
                'primary_recommendation': None,
                'alternatives': [],
                'aggregation_suggestions': []
            },
            'timestamp_analysis': {
                'primary_recommendation': None,
                'alternatives': []
            },
            'attribute_recommendations': {
                'case_attributes': [],
                'event_attributes': []
            },
            'data_quality_concerns': [],
            'transformation_recommendations': [],
            'process_mining_readiness': {'score': 0.5, 'reasoning': 'Basic analysis without AI'},
            'next_steps': []
        }
        
        # Basic rule-based analysis
        all_identifiers = []
        all_timestamps = []
        all_activities = []
        
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            file_path = dataset['file_path']
            
            # Collect potential identifiers
            for col in metadata.get('potential_identifiers', []):
                all_identifiers.append({
                    'table': file_path,
                    'column': col,
                    'confidence': 0.6,
                    'reasoning': 'Rule-based identification'
                })
            
            # Collect potential timestamps
            for col in metadata.get('potential_timestamps', []):
                all_timestamps.append({
                    'table': file_path,
                    'column': col,
                    'confidence': 0.6,
                    'reasoning': 'Rule-based identification'
                })
            
            # Collect potential activities
            for col in metadata.get('potential_activities', []):
                all_activities.append({
                    'table': file_path,
                    'column': col,
                    'confidence': 0.6,
                    'reasoning': 'Rule-based identification'
                })
        
        # Set primary recommendations (highest confidence)
        if all_identifiers:
            fallback_insights['case_id_analysis']['primary_recommendation'] = all_identifiers[0]
            fallback_insights['case_id_analysis']['alternatives'] = all_identifiers[1:3]
        
        if all_timestamps:
            fallback_insights['timestamp_analysis']['primary_recommendation'] = all_timestamps[0]
            fallback_insights['timestamp_analysis']['alternatives'] = all_timestamps[1:3]
        
        if all_activities:
            fallback_insights['activity_analysis']['primary_recommendation'] = all_activities[0]
            fallback_insights['activity_analysis']['alternatives'] = all_activities[1:3]
        
        # Basic recommendations
        fallback_insights['next_steps'] = [
            "Configure Azure OpenAI for enhanced AI-powered analysis",
            "Review and validate the identified case ID, activity, and timestamp columns",
            "Check data quality and completeness",
            "Consider data transformations if needed"
        ]
        
        if not all_identifiers:
            fallback_insights['data_quality_concerns'].append("No clear case identifier found")
        
        if not all_timestamps:
            fallback_insights['data_quality_concerns'].append("No timestamp columns identified")
        
        if not all_activities:
            fallback_insights['data_quality_concerns'].append("No activity columns identified")
        
        return fallback_insights
    
    def enhance_recommendations(
        self,
        basic_analysis: Dict[str, Any],
        business_context: str
    ) -> Dict[str, Any]:
        """Enhance basic analysis with AI insights."""
        
        if not self.client:
            return basic_analysis
        
        try:
            enhancement_prompt = f"""Based on the following process mining analysis, provide enhanced recommendations:

CURRENT ANALYSIS:
{json.dumps(basic_analysis, indent=2)}

BUSINESS CONTEXT:
{business_context}

Please provide:
1. Validation of the current recommendations
2. Additional insights or corrections
3. Domain-specific best practices
4. Specific implementation guidance

Respond with enhanced recommendations in JSON format."""
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a process mining expert providing enhanced analysis."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            enhanced_insights = self._parse_ai_response(response.choices[0].message.content)
            
            # Merge with original analysis
            basic_analysis['ai_enhancements'] = enhanced_insights
            
            return basic_analysis
            
        except Exception as e:
            logger.error(f"Failed to enhance recommendations: {e}")
            return basic_analysis
