import sys
import os
sys.path.append('src')
from core.schema_analyzer import SchemaAnalyzer

analyzer = SchemaAnalyzer()
# Test with one of the XSD files
schema_info = analyzer.parse_schema_file('data/Order.xsd')
print('Schema type:', schema_info.get('type'))
print('Total elements:', len(schema_info.get('elements', {})))

# Look at process mining elements
if 'process_mining_candidates' in schema_info:
    pm_candidates = schema_info['process_mining_candidates']
    print('\nProcess Mining Candidates:')
    print('Case ID candidates:', len(pm_candidates.get('case_id_candidates', [])))
    print('Activity candidates:', len(pm_candidates.get('activity_candidates', [])))
    print('Timestamp candidates:', len(pm_candidates.get('timestamp_candidates', [])))
    
    # Show a few examples
    if pm_candidates.get('case_id_candidates'):
        print('\nExample Case ID candidates:')
        for candidate in pm_candidates['case_id_candidates'][:3]:
            print(f'  - {candidate["name"]} (confidence: {candidate["confidence"]})')
            
    if pm_candidates.get('activity_candidates'):
        print('\nExample Activity candidates:')
        for candidate in pm_candidates['activity_candidates'][:3]:
            print(f'  - {candidate["name"]} (confidence: {candidate["confidence"]})')
            
    if pm_candidates.get('timestamp_candidates'):
        print('\nExample Timestamp candidates:')
        for candidate in pm_candidates['timestamp_candidates'][:3]:
            print(f'  - {candidate["name"]} (confidence: {candidate["confidence"]})')
