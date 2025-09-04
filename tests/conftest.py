"""
Pytest configuration and fixtures for AI Pipeline tests.
"""

import pytest
import pandas as pd
import os
import tempfile
from typing import Dict, Any


@pytest.fixture
def sample_customer_data():
    """Sample customer dataset for testing."""
    return pd.DataFrame({
        'customer_id': [1001, 1002, 1003, 1004, 1005],
        'email': ['john@example.com', 'jane@test.org', 'bob@company.net', 'alice@domain.com', 'charlie@site.gov'],
        'phone': ['+1-555-0123', '555.987.6543', '(555) 123-4567', '555-111-2222', '+1 555 999 8888'],
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'last_name': ['Doe', 'Smith', 'Johnson', 'Williams', 'Brown'],
        'age': [25, 34, 42, 28, 55],
        'is_premium': [True, False, True, False, True],
        'signup_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12'],
        'status': ['active', 'inactive', 'active', 'pending', 'active'],
        'total_orders': [5, 0, 12, 1, 8],
        'account_balance': [150.50, 0.00, 1250.75, 25.00, 890.25]
    })


@pytest.fixture
def sample_messy_data():
    """Messy dataset with various data quality issues."""
    return pd.DataFrame({
        'ID_FIELD': ['ABC123', 'DEF456', 'GHI789', 'JKL012', 'MNO345'],
        'mixed_case_Email': ['USER@DOMAIN.COM', 'test@site.org', '', 'bad.email', 'valid@test.com'],
        'Phone_Numbers': ['555-1234', '(555) 987-6543', 'not-a-phone', '555.123.4567', ''],
        'dates_various_formats': ['2023-12-01', '12/25/2023', 'Jan 15, 2024', '2024/03/10', ''],
        'numeric_with_nulls': [100, '', 250.50, '500', None],
        'boolean_mixed': ['true', 'FALSE', 1, 0, 'yes'],
        'free_text': ['This is a description', 'Another text field', '', 'Mixed content 123', 'Final entry'],
        'category_codes': ['A1', 'B2', 'A1', 'C3', 'B2'],
        'uuid_field': [
            '550e8400-e29b-41d4-a716-446655440000',
            '6ba7b810-9dad-11d1-80b4-00c04fd430c8', 
            '6ba7b811-9dad-11d1-80b4-00c04fd430c8',
            '550e8401-e29b-41d4-a716-446655440001',
            '6ba7b812-9dad-11d1-80b4-00c04fd430c9'
        ]
    })


@pytest.fixture
def empty_dataframe():
    """Empty dataframe for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def single_column_dataframe():
    """Single column dataframe."""
    return pd.DataFrame({'test_column': ['value1', 'value2', 'value3']})


@pytest.fixture
def temp_csv_file(sample_customer_data):
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_customer_data.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    return {
        'ANTHROPIC_API_KEY': 'test_claude_key',
        'GOOGLE_AI_API_KEY': 'test_gemini_key',
        'OPENAI_API_KEY': 'test_openai_key'
    }


@pytest.fixture
def classification_test_cases():
    """Comprehensive test cases for classification."""
    return [
        {
            'column_name': 'customer_id',
            'data': [1001, 1002, 1003, 1004, 1005],
            'expected_type': 'identifier',
            'expected_primary_key': True
        },
        {
            'column_name': 'email_address',
            'data': ['user@domain.com', 'test@example.org', 'admin@company.net'],
            'expected_type': 'email',
            'expected_pii': 'high'
        },
        {
            'column_name': 'phone_number',
            'data': ['555-1234', '(555) 987-6543', '+1-555-0123'],
            'expected_type': 'phone',
            'expected_pii': 'high'
        },
        {
            'column_name': 'order_date',
            'data': ['2023-01-15', '2023-02-20', '2023-03-10'],
            'expected_type': 'date'
        },
        {
            'column_name': 'is_active',
            'data': [True, False, True, False],
            'expected_type': 'boolean'
        },
        {
            'column_name': 'status_code',
            'data': ['ACTIVE', 'INACTIVE', 'PENDING', 'ACTIVE'],
            'expected_type': 'business_key',
            'expected_business_key': True
        }
    ]