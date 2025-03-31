import logging
from typing import Dict, Any
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates customer-friendly responses based on message type and data."""

    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key."""
        openai.api_key = openai_api_key

    async def generate_customer_response(
        self, message_type: str, response_data: Dict[str, Any],
        customer_message: str, product: str
    ) -> str:
        """
        Generate a customer-friendly response based on message type and data.

        Args:
            message_type: Type of message (bug_report, feature_request, general_inquiry)
            response_data: Structured data for the message type
            customer_message: Original customer message
            product: Product name

        Returns:
            A human-friendly response text
        """
        try:
            if message_type == "bug_report":
                return await self._generate_bug_report_response(response_data, customer_message, product)
            elif message_type == "feature_request":
                return await self._generate_feature_request_response(response_data, customer_message, product)
            else:  # general_inquiry
                return await self._generate_general_inquiry_response(response_data, customer_message, product)
        except Exception as e:
            logger.error(f"Error generating customer response: {str(e)}")
            # Fallback response
            return f"Thank you for contacting us about {product}. Our team will review your message and get back to you soon."

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def _generate_bug_report_response(
        self, response_data: Dict[str, Any], customer_message: str, product: str
    ) -> str:
        """Generate a response for a bug report."""
        ticket = response_data.get("ticket", {})

        system_message = """
        You are a helpful and empathetic customer support specialist.
        Your goal is to acknowledge the customer's issue, provide them with information about their ticket,
        and set expectations about next steps. Keep your response under 4 sentences.
        """

        user_message = f"""
        Product: {product}
        Customer bug report: "{customer_message}"

        Ticket details:
        ID: {ticket.get('id')}
        Title: {ticket.get('title')}
        Severity: {ticket.get('severity')}
        Priority: {ticket.get('priority')}

        Write a helpful response to the customer that:
        1. Acknowledges their issue
        2. Thanks them for reporting the problem
        3. Informs them that a ticket has been created (include the ID)
        4. Sets expectations about next steps

        The tone should be professional, helpful, and understanding.
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=200,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating bug report response: {str(e)}")
            # Fallback response with ticket information
            return (f"Thank you for reporting this issue with {product}. We've created ticket "
                    f"{ticket.get('id', 'BUG-0000')} with {ticket.get('priority', 'High')} priority "
                    f"and our team is investigating. We'll update you once we have more information.")

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def _generate_feature_request_response(
        self, response_data: Dict[str, Any], customer_message: str, product: str
    ) -> str:
        """Generate a response for a feature request."""
        req = response_data.get("product_requirement", {})

        system_message = """
        You are a helpful and appreciative product specialist.
        Your goal is to thank the customer for their feature suggestion and
        provide information about how their feedback will be considered.
        Keep your response under 4 sentences.
        """

        user_message = f"""
        Product: {product}
        Customer feature request: "{customer_message}"

        Feature details:
        ID: {req.get('id')}
        Title: {req.get('title')}
        Status: {req.get('status')}
        Business Value: {req.get('business_value')}

        Write a helpful response to the customer that:
        1. Thanks them for their suggestion
        2. Acknowledges the feature they've requested
        3. Informs them that their suggestion has been logged (include the ID)
        4. Sets expectations about the review process

        The tone should be appreciative, enthusiastic, and professional.
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=200,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating feature request response: {str(e)}")
            # Fallback response with feature request information
            return (f"Thank you for your suggestion about {req.get('title', 'this feature')}! "
                    f"We've logged it as {req.get('id', 'FR-0000')} and our product team will review it. "
                    f"We appreciate your feedback as it helps us improve {product}.")

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def _generate_general_inquiry_response(
        self, response_data: Dict[str, Any], customer_message: str, product: str
    ) -> str:
        """Generate a response for a general inquiry."""
        category = response_data.get("inquiry_category", "Other")
        requires_human = response_data.get("requires_human_review", False)
        resources = response_data.get("suggested_resources", [])

        system_message = """
        You are a knowledgeable and helpful customer support specialist.
        Your goal is to provide clear information to the customer's inquiry
        and direct them to relevant resources. Keep your response under 4 sentences.
        """

        resource_list = "\n".join([f"- {r.get('title')}: {r.get('url')}" for r in resources])

        user_message = f"""
        Product: {product}
        Customer inquiry: "{customer_message}"

        Inquiry details:
        Category: {category}
        Requires human review: {'Yes' if requires_human else 'No'}

        Suggested resources:
        {resource_list}

        Write a helpful response to the customer that:
        1. Acknowledges their inquiry
        2. Provides a direct answer if possible
        3. References relevant resources that might help them
        4. Informs them {'that a support specialist will reach out' if requires_human else 'of any next steps they should take'}

        The tone should be helpful, informative, and friendly.
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=200,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating general inquiry response: {str(e)}")
            # Fallback response with resources
            resource_text = ""
            for resource in resources[:2]:  # Limit to first 2 resources
                title = resource.get("title", "Resource")
                url = resource.get("url", "https://example.com")
                resource_text += f" {title} at {url},"

            if resource_text:
                resource_text = f" For more information, check out{resource_text[:-1]}."

            return f"Thank you for your inquiry about {product}.{resource_text}"