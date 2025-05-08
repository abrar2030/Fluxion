import pytest
from flask import Flask
import os

# Ensure the app is imported correctly from its location
# Adjust the path as necessary based on your project structure
from backend.app import app as flask_app  # Assuming app.py is in a 'backend' directory relative to tests

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    # Mock environment variables if your app relies on them during testing
    os.environ['PORT'] = '5000' # Example, adjust as needed
    os.environ['FLASK_ENV'] = 'development' # Example, adjust as needed

    with flask_app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the /health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {"status": "healthy"}

def test_home_endpoint(client):
    """Test the / (home) endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.json == {
        "message": "Welcome to Fluxion API",
        "version": "1.0.0",
        "status": "operational"
    }

def test_404_not_found(client):
    """Test the 404 error handler for a non-existent route."""
    response = client.get('/nonexistentroute')
    assert response.status_code == 404
    assert response.json == {"error": "Resource not found"}

def test_500_error(client):
    """Test 500 error handler"""
    # Force a 500 error by accessing an invalid route with POST
    response = client.post('/')
    assert response.status_code == 500
    data = response.get_json()
    assert data['error'] == 'Internal server error'

# To test the 500 error handler, we need a way to trigger an internal server error.
# One way is to add a temporary route that intentionally raises an exception.
# Or, if you have a specific function that might fail, you can mock it to raise an error.

# Example of adding a temporary route for testing 500 error (ensure this is not in production app.py):
# @flask_app.route('/trigger_error')
# def trigger_error():
#     raise Exception("Intentional Test Error")

# def test_500_internal_error(client):
#     """Test the 500 error handler."""
#     # This test assumes you have a way to trigger an internal server error, 
#     # for example, by adding a temporary route that raises an exception.
#     # If not, you might need to mock a function within your app to simulate an error.
#     response = client.get('/trigger_error') # Make sure this route exists for testing purposes
#     assert response.status_code == 500
#     assert response.json == {"error": "Internal server error"}

# If you prefer mocking, you might do something like this (conceptual):
# from unittest.mock import patch
# @patch('your_app_module.some_function_that_might_fail')
# def test_500_with_mock(mock_failing_function, client):
#     mock_failing_function.side_effect = Exception("Simulated error")
#     response = client.get('/some_route_that_uses_the_function')
#     assert response.status_code == 500
#     assert response.json == {"error": "Internal server error"}

