import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json

from main import app
from models import CustomerMessage
from classifier import MessageClassifier
from response_generator import ResponseGenerator

client = TestClient(app)

# Test cases from requirements
TEST_CASE_1 = "I can't log in to the web portal. When I enter my password and click login, the button just spins and nothing happens."
TEST_CASE_2 = "It would be really useful if the app could send me a notification 15 minutes before a scheduled workout instead of just 5 minutes before."
TEST_CASE_3 = "Hello, I just signed up yesterday. Can you tell me how billing works and if there's a way to switch between monthly and annual plans?"

def test_root_endpoint():
    """Test the root endpoint returns the correct status."""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "running" in response.json()["status"]

@pytest.mark.parametrize("test_message,expected_type", [
    (TEST_CASE_1, "bug_report"),
    (TEST_CASE_2, "feature_request"),
    (TEST_CASE_3, "general_inquiry"),
])
def test_process_message_test_cases(test_message, expected_type):
    """Test that the test cases are correctly classified."""
    response = client.post(
        "/process-customer-message",
        json={
            "customer_id": "test_user",
            "message": test_message,
            "product": "1440 Mobile App"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message_type"] == expected_type
    assert isinstance(data["confidence_score"], float)
    assert 0 <= data["confidence_score"] <= 1
    assert "response_data" in data
    assert "customer_response" in data
    assert data["customer_response"]  # Not empty

def test_bug_report_response_format():
    """Test that bug report responses have the correct format."""
    response = client.post(
        "/process-customer-message",
        json={
            "customer_id": "test_user",
            "message": TEST_CASE_1,
            "product": "1440 Mobile App"
        }
    )
    data = response.json()
    assert "ticket" in data["response_data"]
    ticket = data["response_data"]["ticket"]
    assert "id" in ticket
    assert ticket["id"].startswith("BUG-")
    assert "title" in ticket
    assert "severity" in ticket
    assert "affected_component" in ticket
    assert "reproduction_steps" in ticket
    assert isinstance(ticket["reproduction_steps"], list)
    assert "priority" in ticket
    assert "assigned_team" in ticket

def test_feature_request_response_format():
    """Test that feature request responses have the correct format."""
    response = client.post(
        "/process-customer-message",
        json={
            "customer_id": "test_user",
            "message": TEST_CASE_2,
            "product": "1440 Mobile App"
        }
    )
    data = response.json()
    assert "product_requirement" in data["response_data"]
    req = data["response_data"]["product_requirement"]
    assert "id" in req
    assert req["id"].startswith("FR-")
    assert "title" in req
    assert "description" in req
    assert "user_story" in req
    assert "business_value" in req
    assert "complexity_estimate" in req
    assert "affected_components" in req
    assert isinstance(req["affected_components"], list)
    assert "status" in req
    assert req["status"] == "Under Review"

def test_general_inquiry_response_format():
    """Test that general inquiry responses have the correct format."""
    response = client.post(
        "/process-customer-message",
        json={
            "customer_id": "test_user",
            "message": TEST_CASE_3,
            "product": "1440 Mobile App"
        }
    )
    data = response.json()
    rd = data["response_data"]
    assert "inquiry_category" in rd
    assert rd["inquiry_category"] in ["Account Management", "Billing", "Usage Question", "Other"]
    assert "requires_human_review" in rd
    assert isinstance(rd["requires_human_review"], bool)
    assert "suggested_resources" in rd
    assert isinstance(rd["suggested_resources"], list)
    for resource in rd["suggested_resources"]:
        assert "title" in resource
        assert "url" in resource

@patch("classifier.MessageClassifier.classify_message")
@patch("response_generator.ResponseGenerator.generate_customer_response")
async def test_error_handling(mock_generate, mock_classify):
    """Test that errors are properly handled."""
    # Setup mock to raise an exception
    mock_classify.side_effect = Exception("Test error")

    response = client.post(
        "/process-customer-message",
        json={
            "customer_id": "test_user",
            "message": "This will cause an error",
            "product": "1440 Mobile App"
        }
    )
    assert response.status_code == 500
    assert "detail" in response.json()