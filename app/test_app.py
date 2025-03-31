import pytest
from fastapi.testclient import TestClient
import json
import logging

# Configure logging to show test output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_app")

# Import the app and dependencies
try:
    from main import app
    from models import CustomerMessage
    from classifier import MessageClassifier
    from response_generator import ResponseGenerator
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

# Create test client
client = TestClient(app)

# Test cases from requirements
TEST_CASE_1 = "I can't log in to the web portal. When I enter my password and click login, the button just spins and nothing happens."
TEST_CASE_2 = "It would be really useful if the app could send me a notification 15 minutes before a scheduled workout instead of just 5 minutes before."
TEST_CASE_3 = "Hello, I just signed up yesterday. Can you tell me how billing works and if there's a way to switch between monthly and annual plans?"

def test_root_endpoint():
    """Test the root endpoint returns the correct status."""
    logger.info("Testing root endpoint")
    response = client.get("/")
    logger.info(f"Response status: {response.status_code}")
    logger.info(f"Response body: {response.json()}")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "running" in response.json()["status"]

@pytest.mark.parametrize("test_case,test_message,expected_type", [
    (1, TEST_CASE_1, "bug_report"),
    (2, TEST_CASE_2, "feature_request"),
    (3, TEST_CASE_3, "general_inquiry"),
])
def test_process_message_test_cases(test_case, test_message, expected_type):
    """Test that the test cases are correctly classified."""
    logger.info(f"Testing test case {test_case}: {expected_type}")

    request_data = {
        "customer_id": f"test_user_{test_case}",
        "message": test_message,
        "product": "1440 Mobile App"
    }

    logger.info(f"Request data: {json.dumps(request_data)}")

    try:
        response = client.post(
            "/process-customer-message",
            json=request_data
        )

        logger.info(f"Response status: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Response error: {response.json()}")
            assert False, f"Expected status code 200, got {response.status_code}"

        data = response.json()
        logger.info(f"Response data: {json.dumps(data)}")

        assert data["message_type"] == expected_type, f"Expected {expected_type}, got {data['message_type']}"
        assert isinstance(data["confidence_score"], float), "confidence_score is not a float"
        assert 0 <= data["confidence_score"] <= 1, "confidence_score out of range"
        assert "response_data" in data, "Missing response_data"
        assert "customer_response" in data, "Missing customer_response"

        logger.info(f"Test case {test_case} passed")
    except Exception as e:
        logger.error(f"Exception during test: {str(e)}", exc_info=True)
        raise

# Add more specific test cases with enhanced logging
def test_bug_report_format():
    """Test bug report format in detail"""
    logger.info("Testing bug report format")
    try:
        response = client.post(
            "/process-customer-message",
            json={
                "customer_id": "test_user_bug",
                "message": TEST_CASE_1,
                "product": "1440 Mobile App"
            }
        )

        data = response.json()
        logger.info(f"Response: {json.dumps(data)}")

        # Validate structure
        assert "ticket" in data["response_data"], "Missing ticket object"
        ticket = data["response_data"]["ticket"]

        # Log each field for debugging
        for field in ["id", "title", "severity", "affected_component",
                    "reproduction_steps", "priority", "assigned_team"]:
            logger.info(f"Ticket {field}: {ticket.get(field, 'MISSING')}")
            assert field in ticket, f"Missing {field} in ticket"

        # Validate specific formats
        assert ticket["id"].startswith("BUG-"), "Invalid ticket ID format"
        assert isinstance(ticket["reproduction_steps"], list), "reproduction_steps is not a list"

        logger.info("Bug report format test passed")
    except Exception as e:
        logger.error(f"Exception during test: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Running tests directly")
    test_root_endpoint()
    logger.info("Root endpoint test completed")

    for i, (test_case, expected) in enumerate([
        (TEST_CASE_1, "bug_report"),
        (TEST_CASE_2, "feature_request"),
        (TEST_CASE_3, "general_inquiry")
    ]):
        test_process_message_test_cases(i+1, test_case, expected)
        logger.info(f"Test case {i+1} completed")

    test_bug_report_format()
    logger.info("Bug report format test completed")

    logger.info("All tests completed")