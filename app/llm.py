import os
import json
import random
import logging
from typing import Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MessageProcessor:
    """Process customer messages using LLM."""

    @staticmethod
    def process_message(message: str, product: str) -> Dict[str, Any]:
        """
        Process a customer message using the OpenAI API.

        Args:
            message: Customer message text
            product: Product name

        Returns:
            Dictionary with message_type, confidence_score, response_data, and customer_response
        """
        # Generate random IDs for tickets and feature requests
        bug_id = f"BUG-{random.randint(1000, 9999)}"
        feature_id = f"FR-{random.randint(1000, 9999)}"

        # Create the prompt
        prompt = MessageProcessor._create_prompt(message, product, bug_id, feature_id)

        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            # Parse response
            result = json.loads(response.choices[0].message.content)

            # Validate response structure
            if "message_type" not in result or "confidence_score" not in result or "response_data" not in result:
                logger.warning("Invalid response format from LLM")
                return MessageProcessor._generate_fallback(message, product)

            logger.info(f"Message classified as {result['message_type']} with confidence {result['confidence_score']}")
            return result

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return MessageProcessor._generate_fallback(message, product)

    @staticmethod
    def _create_prompt(message: str, product: str, bug_id: str, feature_id: str) -> str:
        """Create the prompt for the LLM."""
        return f"""
        You are an AI expert at analyzing customer support messages for {product}.

        TASK: Analyze this customer message and generate a detailed, structured response:

        Message: "{message}"

        INSTRUCTIONS:
        1. First, determine if this is a:
           - bug_report: User reporting an issue with existing functionality
           - feature_request: User suggesting new functionality or improvements
           - general_inquiry: User asking questions about usage, billing, etc.

        2. Generate an appropriate response with the following structure:

        For bug_report:
        ```
        {{
          "message_type": "bug_report",
          "confidence_score": (float between 0.0-1.0),
          "response_data": {{
            "ticket": {{
              "id": "{bug_id}",
              "title": (clear, concise issue title),
              "severity": (must be exactly "Low", "Medium", "High", or "Critical"),
              "affected_component": (component name),
              "reproduction_steps": [(step 1), (step 2), ...],
              "priority": (must be exactly "Low", "Medium", or "High"),
              "assigned_team": (team name)
            }}
          }},
          "customer_response": (helpful response acknowledging the issue)
        }}
        ```

        For feature_request:
        ```
        {{
          "message_type": "feature_request",
          "confidence_score": (float between 0.0-1.0),
          "response_data": {{
            "product_requirement": {{
              "id": "{feature_id}",
              "title": (clear, concise feature title),
              "description": (detailed feature description),
              "user_story": (format: "As a user, I want to... so that..."),
              "business_value": (value explanation with rationale),
              "complexity_estimate": (must be exactly "Low", "Medium", or "High"),
              "affected_components": [(component 1), (component 2), ...],
              "status": "Under Review"
            }}
          }},
          "customer_response": (friendly response thanking the user for their suggestion)
        }}
        ```

        For general_inquiry:
        ```
        {{
          "message_type": "general_inquiry",
          "confidence_score": (float between 0.0-1.0),
          "response_data": {{
            "inquiry_category": (must be exactly "Account Management", "Billing", "Usage Question", or "Other"),
            "requires_human_review": (boolean: true or false),
            "suggested_resources": [
              {{"title": (resource name), "url": (resource URL)}},
              ...
            ]
          }},
          "customer_response": (helpful response addressing their question)
        }}
        ```

        IMPORTANT:
        - Use EXACTLY the field names shown above
        - Include all required fields for the message type
        - Ensure values match format requirements
        - Make the customer_response friendly, helpful, and professional

        Return ONLY a valid JSON object matching the format above.
        """

    @staticmethod
    def _generate_fallback(message: str, product: str) -> Dict[str, Any]:
        """Generate fallback response when processing fails."""
        return {
            "message_type": "general_inquiry",
            "confidence_score": 0.3,
            "response_data": {
                "inquiry_category": "Other",
                "requires_human_review": True,
                "suggested_resources": [
                    {"title": "Help Center", "url": "https://example.com/help"},
                    {"title": "Contact Support", "url": "https://example.com/support"}
                ]
            },
            "customer_response": f"Thank you for your message about {product}. We'll have a support agent review your request and get back to you soon."
        }