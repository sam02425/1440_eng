from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import logging

from models import CustomerMessage, CustomerResponse
from llm import MessageProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Customer Message Processor",
    description="Process customer messages and classify them as bug reports, feature requests, or general inquiries",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint that confirms the API is running."""
    return {"status": "Customer Message Processor API is running"}


@app.post("/process-customer-message", response_model=CustomerResponse)
async def process_customer_message(message: CustomerMessage):
    """
    Process a customer message and return a structured response.

    The message is classified as one of:
    - bug_report: Issues with existing functionality
    - feature_request: Suggestions for new functionality
    - general_inquiry: Questions about product, billing, etc.
    """
    try:
        logger.info(f"Processing message from customer {message.customer_id}")

        # Process message using LLM
        result = MessageProcessor.process_message(message.message, message.product)

        # Return response
        return CustomerResponse(
            message_type=result["message_type"],
            confidence_score=result["confidence_score"],
            response_data=result["response_data"],
            customer_response=result["customer_response"]
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)