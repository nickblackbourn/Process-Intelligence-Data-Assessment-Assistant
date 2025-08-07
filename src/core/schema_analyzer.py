"""Schema analyzer for parsing database schemas and understanding data relationships."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from xml.etree import ElementTree as ET

import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, Name


logger = logging.getLogger(__name__)


class SchemaAnalyzer:
    """Analyzes database schemas to understand table structures and relationships."""
    
    def __init__(self):
        """Initialize the SchemaAnalyzer."""
        self.supported_formats = ['.sql', '.ddl', '.xml', '.xsd']
        logger.info("SchemaAnalyzer initialized")
    
    def parse_schema_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a schema file and extract table structures.
        
        Args:
            file_path: Path to the schema file
            
        Returns:
            Dictionary containing parsed schema information
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported schema format: {extension}")
        
        logger.info(f"Parsing schema from {file_path}")
        
        try:
            if extension in ['.sql', '.ddl']:
                return self._parse_sql_schema(file_path)
            elif extension in ['.xml', '.xsd']:
                return self._parse_xml_schema(file_path)
            else:
                raise ValueError(f"Parser not implemented for {extension}")
                
        except Exception as e:
            logger.error(f"Error parsing schema: {e}")
            raise
    
    def _parse_sql_schema(self, file_path: str) -> Dict[str, Any]:
        """Parse SQL DDL schema file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Parse SQL statements
        statements = sqlparse.split(sql_content)
        
        schema_info = {
            'type': 'sql',
            'tables': {},
            'relationships': [],
            'indexes': [],
            'constraints': [],
            'raw_statements': statements
        }
        
        for statement in statements:
            if statement.strip():
                parsed = sqlparse.parse(statement)[0]
                self._analyze_sql_statement(parsed, schema_info)
        
        # Infer relationships based on column names and constraints
        self._infer_relationships(schema_info)
        
        logger.info(f"Parsed {len(schema_info['tables'])} tables from SQL schema")
        return schema_info
    
    def _analyze_sql_statement(self, statement: Statement, schema_info: Dict[str, Any]):
        """Analyze individual SQL statement."""
        statement_str = str(statement).upper().strip()
        
        if statement_str.startswith('CREATE TABLE'):
            self._parse_create_table(statement, schema_info)
        elif statement_str.startswith('ALTER TABLE'):
            self._parse_alter_table(statement, schema_info)
        elif statement_str.startswith('CREATE INDEX'):
            self._parse_create_index(statement, schema_info)
    
    def _parse_create_table(self, statement: Statement, schema_info: Dict[str, Any]):
        """Parse CREATE TABLE statement."""
        statement_str = str(statement)
        
        # Extract table name
        table_name_match = re.search(r'CREATE\s+TABLE\s+(?:\w+\.)?(\w+)', statement_str, re.IGNORECASE)
        if not table_name_match:
            return
        
        table_name = table_name_match.group(1)
        
        # Extract column definitions
        columns = {}
        constraints = []
        
        # Find the column definition section
        paren_match = re.search(r'\((.*)\)', statement_str, re.DOTALL)
        if paren_match:
            column_section = paren_match.group(1)
            
            # Split by commas, but handle nested parentheses
            column_defs = self._split_column_definitions(column_section)
            
            for col_def in column_defs:
                col_def = col_def.strip()
                if col_def.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK')):
                    constraints.append(col_def)
                else:
                    # Parse column definition
                    col_info = self._parse_column_definition(col_def)
                    if col_info:
                        columns[col_info['name']] = col_info
        
        schema_info['tables'][table_name] = {
            'name': table_name,
            'columns': columns,
            'constraints': constraints,
            'primary_keys': self._extract_primary_keys(constraints),
            'foreign_keys': self._extract_foreign_keys(constraints)
        }
    
    def _split_column_definitions(self, column_section: str) -> List[str]:
        """Split column definitions handling nested parentheses."""
        definitions = []
        current_def = ""
        paren_count = 0
        
        for char in column_section:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                if current_def.strip():
                    definitions.append(current_def.strip())
                current_def = ""
                continue
            
            current_def += char
        
        if current_def.strip():
            definitions.append(current_def.strip())
        
        return definitions
    
    def _parse_column_definition(self, col_def: str) -> Optional[Dict[str, Any]]:
        """Parse individual column definition."""
        # Basic column pattern: name type [constraints]
        parts = col_def.split()
        if len(parts) < 2:
            return None
        
        name = parts[0].strip('"`')
        data_type = parts[1]
        
        # Extract additional constraints and properties
        col_info = {
            'name': name,
            'data_type': data_type,
            'nullable': 'NOT NULL' not in col_def.upper(),
            'primary_key': 'PRIMARY KEY' in col_def.upper(),
            'unique': 'UNIQUE' in col_def.upper(),
            'auto_increment': any(keyword in col_def.upper() 
                                for keyword in ['AUTO_INCREMENT', 'IDENTITY', 'SERIAL']),
            'default_value': self._extract_default_value(col_def)
        }
        
        return col_info
    
    def _extract_default_value(self, col_def: str) -> Optional[str]:
        """Extract default value from column definition."""
        default_match = re.search(r'DEFAULT\s+([^\s,]+)', col_def, re.IGNORECASE)
        return default_match.group(1) if default_match else None
    
    def _extract_primary_keys(self, constraints: List[str]) -> List[str]:
        """Extract primary key columns from constraints."""
        primary_keys = []
        for constraint in constraints:
            if 'PRIMARY KEY' in constraint.upper():
                # Extract column names from PRIMARY KEY constraint
                paren_match = re.search(r'\((.*?)\)', constraint)
                if paren_match:
                    cols = [col.strip().strip('"`') for col in paren_match.group(1).split(',')]
                    primary_keys.extend(cols)
        return primary_keys
    
    def _extract_foreign_keys(self, constraints: List[str]) -> List[Dict[str, str]]:
        """Extract foreign key relationships from constraints."""
        foreign_keys = []
        for constraint in constraints:
            if 'FOREIGN KEY' in constraint.upper():
                # Parse FOREIGN KEY (col) REFERENCES table(col)
                fk_match = re.search(
                    r'FOREIGN\s+KEY\s*\(\s*(\w+)\s*\)\s*REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)',
                    constraint, re.IGNORECASE
                )
                if fk_match:
                    foreign_keys.append({
                        'column': fk_match.group(1),
                        'referenced_table': fk_match.group(2),
                        'referenced_column': fk_match.group(3)
                    })
        return foreign_keys
    
    def _parse_alter_table(self, statement: Statement, schema_info: Dict[str, Any]):
        """Parse ALTER TABLE statement."""
        # For now, just log that we found an ALTER TABLE
        logger.debug(f"Found ALTER TABLE statement: {str(statement)[:100]}...")
    
    def _parse_create_index(self, statement: Statement, schema_info: Dict[str, Any]):
        """Parse CREATE INDEX statement."""
        statement_str = str(statement)
        index_match = re.search(
            r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+)\s*\((.*?)\)',
            statement_str, re.IGNORECASE
        )
        if index_match:
            schema_info['indexes'].append({
                'name': index_match.group(1),
                'table': index_match.group(2),
                'columns': [col.strip() for col in index_match.group(3).split(',')],
                'unique': 'UNIQUE' in statement_str.upper()
            })
    
    def _infer_relationships(self, schema_info: Dict[str, Any]):
        """Infer additional relationships based on column names and patterns."""
        tables = schema_info['tables']
        
        for table_name, table_info in tables.items():
            for col_name, col_info in table_info['columns'].items():
                # Look for foreign key patterns (e.g., table_id, tableId)
                if col_name.lower().endswith('_id') or col_name.lower().endswith('id'):
                    potential_table = col_name.lower().replace('_id', '').replace('id', '')
                    
                    # Check if there's a table with similar name
                    for other_table in tables.keys():
                        if other_table.lower() == potential_table:
                            # Potential relationship found
                            schema_info['relationships'].append({
                                'type': 'inferred_foreign_key',
                                'from_table': table_name,
                                'from_column': col_name,
                                'to_table': other_table,
                                'confidence': 0.7
                            })
    
    def _parse_xml_schema(self, file_path: str) -> Dict[str, Any]:
        """Parse XML schema file (XSD)."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        schema_info = {
            'type': 'xml',
            'elements': {},
            'complex_types': {},
            'simple_types': {},
            'namespaces': {}
        }
        
        # Extract namespace information
        for prefix, uri in root.nsmap.items() if hasattr(root, 'nsmap') else {}:
            schema_info['namespaces'][prefix] = uri
        
        # Parse elements and types
        for element in root.iter():
            if element.tag.endswith('element'):
                self._parse_xml_element(element, schema_info)
            elif element.tag.endswith('complexType'):
                self._parse_xml_complex_type(element, schema_info)
        
        logger.info(f"Parsed XML schema with {len(schema_info['elements'])} elements")
        return schema_info
    
    def _parse_xml_element(self, element: ET.Element, schema_info: Dict[str, Any]):
        """Parse XML element definition."""
        name = element.get('name')
        if name:
            schema_info['elements'][name] = {
                'name': name,
                'type': element.get('type'),
                'min_occurs': element.get('minOccurs', '1'),
                'max_occurs': element.get('maxOccurs', '1')
            }
    
    def _parse_xml_complex_type(self, element: ET.Element, schema_info: Dict[str, Any]):
        """Parse XML complex type definition."""
        name = element.get('name')
        if name:
            schema_info['complex_types'][name] = {
                'name': name,
                'elements': []
            }
            
            # Find child elements
            for child in element.iter():
                if child.tag.endswith('element'):
                    child_name = child.get('name')
                    if child_name:
                        schema_info['complex_types'][name]['elements'].append(child_name)
    
    def identify_process_mining_candidates(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential process mining elements in the schema.
        
        Args:
            schema_info: Parsed schema information
            
        Returns:
            Dictionary with process mining candidates
        """
        candidates = {
            'case_id_candidates': [],
            'activity_candidates': [],
            'timestamp_candidates': [],
            'event_tables': [],
            'case_tables': []
        }
        
        if schema_info['type'] == 'sql':
            tables = schema_info['tables']
            
            for table_name, table_info in tables.items():
                table_analysis = self._analyze_table_for_process_mining(table_name, table_info)
                
                if table_analysis['likely_event_table']:
                    candidates['event_tables'].append({
                        'table': table_name,
                        'confidence': table_analysis['event_confidence'],
                        'reasons': table_analysis['event_reasons']
                    })
                
                if table_analysis['likely_case_table']:
                    candidates['case_tables'].append({
                        'table': table_name,
                        'confidence': table_analysis['case_confidence'],
                        'reasons': table_analysis['case_reasons']
                    })
                
                # Collect column-level candidates
                for col_name, col_info in table_info['columns'].items():
                    if self._could_be_case_id(col_name, col_info):
                        candidates['case_id_candidates'].append({
                            'table': table_name,
                            'column': col_name,
                            'confidence': self._calculate_case_id_confidence(col_name, col_info)
                        })
                    
                    if self._could_be_timestamp(col_name, col_info):
                        candidates['timestamp_candidates'].append({
                            'table': table_name,
                            'column': col_name,
                            'confidence': self._calculate_timestamp_confidence(col_name, col_info)
                        })
                    
                    if self._could_be_activity(col_name, col_info):
                        candidates['activity_candidates'].append({
                            'table': table_name,
                            'column': col_name,
                            'confidence': self._calculate_activity_confidence(col_name, col_info)
                        })
        
        # Sort candidates by confidence
        for key in ['case_id_candidates', 'activity_candidates', 'timestamp_candidates']:
            candidates[key].sort(key=lambda x: x['confidence'], reverse=True)
        
        return candidates
    
    def _analyze_table_for_process_mining(self, table_name: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if table is likely an event or case table."""
        analysis = {
            'likely_event_table': False,
            'likely_case_table': False,
            'event_confidence': 0.0,
            'case_confidence': 0.0,
            'event_reasons': [],
            'case_reasons': []
        }
        
        columns = table_info['columns']
        column_names = [col.lower() for col in columns.keys()]
        
        # Event table indicators
        event_indicators = [
            ('has_timestamp', any('time' in name or 'date' in name for name in column_names)),
            ('has_activity', any('activity' in name or 'event' in name or 'action' in name for name in column_names)),
            ('has_case_ref', any('case' in name or 'order' in name or 'ticket' in name for name in column_names)),
            ('table_name_suggests_events', any(keyword in table_name.lower() for keyword in ['log', 'event', 'activity', 'transaction']))
        ]
        
        event_score = sum(1 for _, condition in event_indicators if condition)
        analysis['event_confidence'] = event_score / len(event_indicators)
        analysis['event_reasons'] = [reason for reason, condition in event_indicators if condition]
        analysis['likely_event_table'] = analysis['event_confidence'] > 0.5
        
        # Case table indicators
        case_indicators = [
            ('has_case_id', any('id' in name and ('case' in name or 'order' in name or 'ticket' in name) for name in column_names)),
            ('has_case_attributes', len(columns) > 3),  # Case tables usually have multiple attributes
            ('primary_key_suggests_case', len(table_info.get('primary_keys', [])) == 1),
            ('table_name_suggests_cases', any(keyword in table_name.lower() for keyword in ['case', 'order', 'ticket', 'customer', 'application']))
        ]
        
        case_score = sum(1 for _, condition in case_indicators if condition)
        analysis['case_confidence'] = case_score / len(case_indicators)
        analysis['case_reasons'] = [reason for reason, condition in case_indicators if condition]
        analysis['likely_case_table'] = analysis['case_confidence'] > 0.5
        
        return analysis
    
    def _could_be_case_id(self, col_name: str, col_info: Dict[str, Any]) -> bool:
        """Check if column could be a case identifier."""
        name_lower = col_name.lower()
        
        # Strong indicators
        if any(pattern in name_lower for pattern in ['case_id', 'caseid', 'order_id', 'orderid', 'ticket_id', 'ticketid']):
            return True
        
        # Weaker indicators
        if (col_info.get('primary_key', False) or 
            col_info.get('unique', False) or
            'id' in name_lower):
            return True
        
        return False
    
    def _could_be_timestamp(self, col_name: str, col_info: Dict[str, Any]) -> bool:
        """Check if column could be a timestamp."""
        name_lower = col_name.lower()
        data_type = col_info.get('data_type', '').lower()
        
        # Strong indicators
        timestamp_keywords = ['timestamp', 'datetime', 'created_at', 'updated_at', 'event_time', 'time', 'date']
        if any(keyword in name_lower for keyword in timestamp_keywords):
            return True
        
        # Data type indicators
        if any(dt in data_type for dt in ['timestamp', 'datetime', 'date']):
            return True
        
        return False
    
    def _could_be_activity(self, col_name: str, col_info: Dict[str, Any]) -> bool:
        """Check if column could be an activity."""
        name_lower = col_name.lower()
        
        activity_keywords = ['activity', 'event', 'action', 'status', 'state', 'step', 'phase']
        return any(keyword in name_lower for keyword in activity_keywords)
    
    def _calculate_case_id_confidence(self, col_name: str, col_info: Dict[str, Any]) -> float:
        """Calculate confidence score for case ID candidate."""
        score = 0.0
        name_lower = col_name.lower()
        
        if 'case' in name_lower and 'id' in name_lower:
            score += 0.9
        elif any(pattern in name_lower for pattern in ['order_id', 'ticket_id']):
            score += 0.8
        elif col_info.get('primary_key', False):
            score += 0.7
        elif col_info.get('unique', False):
            score += 0.6
        elif 'id' in name_lower:
            score += 0.4
        
        return min(1.0, score)
    
    def _calculate_timestamp_confidence(self, col_name: str, col_info: Dict[str, Any]) -> float:
        """Calculate confidence score for timestamp candidate."""
        score = 0.0
        name_lower = col_name.lower()
        data_type = col_info.get('data_type', '').lower()
        
        if 'timestamp' in name_lower or 'datetime' in name_lower:
            score += 0.9
        elif any(keyword in name_lower for keyword in ['created_at', 'updated_at', 'event_time']):
            score += 0.8
        elif 'time' in name_lower or 'date' in name_lower:
            score += 0.6
        
        if any(dt in data_type for dt in ['timestamp', 'datetime']):
            score += 0.5
        elif 'date' in data_type:
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_activity_confidence(self, col_name: str, col_info: Dict[str, Any]) -> float:
        """Calculate confidence score for activity candidate."""
        score = 0.0
        name_lower = col_name.lower()
        
        if 'activity' in name_lower:
            score += 0.9
        elif 'event' in name_lower:
            score += 0.8
        elif 'action' in name_lower:
            score += 0.7
        elif any(keyword in name_lower for keyword in ['status', 'state', 'step']):
            score += 0.6
        
        return min(1.0, score)
