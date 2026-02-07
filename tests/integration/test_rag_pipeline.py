import pytest
import os
import json
from unittest.mock import MagicMock, patch
from src.query.query_engine import QueryEngine

@pytest.fixture
def mock_openai():
    """Mocks the global OpenAI client with high-fidelity structured responses."""
    with patch('openai.OpenAI') as mock_client:
        mock_instance = mock_client.return_value
        
        # Create a mock response object that looks like OpenAI's ChatCompletion
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "merchants": ["Target"],
            "date_range": None,
            "aggregation": None
        })
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        mock_instance.chat.completions.create.return_value = mock_response
        yield mock_client

@pytest.fixture
def mock_vector_manager():
    vm = MagicMock()
    vm.hybrid_search.return_value = [
        {
            'id': 'c1',
            'score': 0.95,
            'metadata': {
                'receipt_id': 'r1',
                'merchant_name': 'Target',
                'total_amount': 25.0,
                'transaction_date': '2024-01-01',
                'chunk_type': 'receipt_summary'
            }
        }
    ]
    return vm

def test_query_engine_orchestration(mock_openai, mock_vector_manager):
    """Verifies the orchestrator correctly parses, retrieves, and synthesizes."""
    # Ensure environment variables for testing are present
    os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"
    
    engine = QueryEngine(mock_vector_manager)
    engine.generator.generate = MagicMock(return_value="Target receipt for $25 found.")
    
    result = engine.query("Show me Target receipts")
    
    assert "Target" in result.answer
    assert result.receipts[0]['merchant_name'] == "Target"
    mock_vector_manager.hybrid_search.assert_called_once()
    
    # Verify filters
    filters = mock_vector_manager.hybrid_search.call_args[1]['filters']
    assert filters['merchant_name_norm'] == 'target'
