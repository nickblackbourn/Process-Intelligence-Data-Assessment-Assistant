"""
Multi-Tab Excel Test Creator
===========================

Creates test Excel files with multiple tabs and embedded schemas to demonstrate
the enhanced multi-tab processing capabilities.
"""

import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

def create_multi_tab_test_file():
    """Create a comprehensive multi-tab Excel file for testing."""
    
    with pd.ExcelWriter('test_data/multi_tab_process_data.xlsx', engine='openpyxl') as writer:
        
        # Tab 1: Order Data (main transactional data)
        orders_data = {
            'order_id': [f'ORD-{1000+i}' for i in range(100)],
            'customer_id': [f'CUST-{random.randint(1, 20)}' for _ in range(100)],
            'order_date': [(datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d') for _ in range(100)],
            'status': [random.choice(['created', 'confirmed', 'shipped', 'delivered', 'cancelled']) for _ in range(100)],
            'total_amount': [round(random.uniform(50, 5000), 2) for _ in range(100)],
            'sales_rep': [random.choice(['Alice Smith', 'Bob Jones', 'Carol Davis', 'David Wilson']) for _ in range(100)]
        }
        pd.DataFrame(orders_data).to_excel(writer, sheet_name='Orders', index=False)
        
        # Tab 2: Order Events (activity-level data)
        events_data = []
        for order_id in orders_data['order_id'][:50]:  # Events for first 50 orders
            base_date = datetime.strptime(random.choice(orders_data['order_date']), '%Y-%m-%d')
            
            events = [
                {'order_id': order_id, 'activity': 'Order Created', 'timestamp': base_date, 'user': 'System'},
                {'order_id': order_id, 'activity': 'Payment Verified', 'timestamp': base_date + timedelta(minutes=30), 'user': 'Payment System'},
                {'order_id': order_id, 'activity': 'Order Confirmed', 'timestamp': base_date + timedelta(hours=2), 'user': random.choice(['Alice Smith', 'Bob Jones'])},
                {'order_id': order_id, 'activity': 'Shipped', 'timestamp': base_date + timedelta(days=1), 'user': 'Warehouse'},
                {'order_id': order_id, 'activity': 'Delivered', 'timestamp': base_date + timedelta(days=3), 'user': 'Courier'}
            ]
            
            # Add random subset of events for variety
            num_events = random.randint(3, 5)
            events_data.extend(events[:num_events])
        
        events_df = pd.DataFrame(events_data)
        events_df['timestamp'] = events_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        events_df.to_excel(writer, sheet_name='OrderEvents', index=False)
        
        # Tab 3: Customer Master Data
        customers_data = {
            'customer_id': [f'CUST-{i}' for i in range(1, 21)],
            'customer_name': [f'Customer {i}' for i in range(1, 21)],
            'customer_type': [random.choice(['Premium', 'Standard', 'Basic']) for _ in range(20)],
            'region': [random.choice(['North', 'South', 'East', 'West']) for _ in range(20)],
            'registration_date': [(datetime.now() - timedelta(days=random.randint(30, 1000))).strftime('%Y-%m-%d') for _ in range(20)]
        }
        pd.DataFrame(customers_data).to_excel(writer, sheet_name='Customers', index=False)
        
        # Tab 4: Data Dictionary (Schema Definition)
        schema_data = {
            'table_name': [
                'Orders', 'Orders', 'Orders', 'Orders', 'Orders', 'Orders',
                'OrderEvents', 'OrderEvents', 'OrderEvents', 'OrderEvents',
                'Customers', 'Customers', 'Customers', 'Customers', 'Customers'
            ],
            'column_name': [
                'order_id', 'customer_id', 'order_date', 'status', 'total_amount', 'sales_rep',
                'order_id', 'activity', 'timestamp', 'user',
                'customer_id', 'customer_name', 'customer_type', 'region', 'registration_date'
            ],
            'data_type': [
                'varchar(20)', 'varchar(20)', 'date', 'varchar(20)', 'decimal(10,2)', 'varchar(100)',
                'varchar(20)', 'varchar(100)', 'timestamp', 'varchar(100)',
                'varchar(20)', 'varchar(200)', 'varchar(50)', 'varchar(50)', 'date'
            ],
            'description': [
                'Unique order identifier', 'Foreign key to customer', 'Date order was placed', 'Current order status', 'Total order value', 'Sales representative',
                'Foreign key to order', 'Process activity name', 'When activity occurred', 'User who performed activity',
                'Unique customer identifier', 'Customer display name', 'Customer tier level', 'Geographic region', 'When customer registered'
            ],
            'nullable': [
                'NO', 'NO', 'NO', 'YES', 'YES', 'YES',
                'NO', 'NO', 'NO', 'YES',
                'NO', 'NO', 'YES', 'YES', 'YES'
            ]
        }
        pd.DataFrame(schema_data).to_excel(writer, sheet_name='DataDictionary', index=False)
        
        # Tab 5: Process Mapping
        process_data = {
            'process': ['Order Management'] * 8,
            'activity': [
                'Order Created', 'Payment Verified', 'Order Confirmed', 'Inventory Check',
                'Shipped', 'In Transit', 'Delivered', 'Order Closed'
            ],
            'step': list(range(1, 9)),
            'input': [
                'Customer order', 'Order details', 'Payment confirmation', 'Confirmed order',
                'Packed order', 'Shipping confirmation', 'Delivery confirmation', 'Delivered order'
            ],
            'output': [
                'Order record', 'Payment status', 'Order confirmation', 'Inventory status',
                'Tracking number', 'Transit update', 'Delivery status', 'Closed order'
            ],
            'role': [
                'Customer', 'Payment System', 'Sales Rep', 'Warehouse',
                'Warehouse', 'Courier', 'Courier', 'System'
            ]
        }
        pd.DataFrame(process_data).to_excel(writer, sheet_name='ProcessMapping', index=False)
        
        # Tab 6: Status Codes (Lookup Table)
        status_codes = {
            'code': ['CR', 'CF', 'SH', 'DL', 'CN'],
            'value': ['created', 'confirmed', 'shipped', 'delivered', 'cancelled'],
            'description': [
                'Order has been created but not confirmed',
                'Order confirmed and ready for processing',
                'Order has been shipped to customer',
                'Order successfully delivered',
                'Order was cancelled before completion'
            ],
            'category': ['Initial', 'Processing', 'Fulfillment', 'Complete', 'Terminal']
        }
        pd.DataFrame(status_codes).to_excel(writer, sheet_name='StatusCodes', index=False)

if __name__ == "__main__":
    import os
    os.makedirs('test_data', exist_ok=True)
    create_multi_tab_test_file()
    print("✅ Created multi_tab_process_data.xlsx with 6 tabs:")
    print("  📊 Orders - Main transactional data")
    print("  📊 OrderEvents - Activity-level events")
    print("  📊 Customers - Master data")
    print("  📋 DataDictionary - Schema definitions")
    print("  📋 ProcessMapping - Process flow definitions")
    print("  📋 StatusCodes - Lookup table")
