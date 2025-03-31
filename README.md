# 1440 Engineering Test Task

## Expected Output
Please deliver your files and 5 - 15 minute video overview of your work. Please record this overview with Loom or a similar service that allows you to record yourself and your screen. If it needs to be broken up into multiple videos, that is okay. You will not be evaluated on the length of your overview. Rather, we want you to talk to us through your code, how you approached the test, what considerations you made, and show your understanding of your completed work.

## Task Overview
Build a FastAPI server with a single endpoint that processes customer messages and classifies them as bug reports, feature requests, or general inquiries, then generates appropriate structured responses.

We encourage you to use any coding/AI tools you would normally use in your day-to-day work.

## Requirements
1. Create a FastAPI endpoint: POST /process-customer-message
2. Input format:
```json
{
  "customer_id": "user_123",
  "message": "Your customer message here",
  "product": "1440 Mobile App"
}
```
3. Return a standardized response:
```json
{
  "message_type": "bug_report" | "feature_request" | "general_inquiry",
  "confidence_score": 0.85,
  "response_data": { ... },
  "customer_response": "Plain text response to the customer"
}
```

## Expected Response Formats

### For bug reports:
```json
{
  "ticket": {
    "id": "BUG-1234",
    "title": "Issue Title",
    "severity": "Medium",
    "affected_component": "Component Name",
    "reproduction_steps": ["Step 1", "Step 2"],
    "priority": "High",
    "assigned_team": "Team Name"
  }
}
```

### For feature requests:
```json
{
  "product_requirement": {
    "id": "FR-5678",
    "title": "Feature Title",
    "description": "Feature description",
    "user_story": "As a user...",
    "business_value": "High/Medium/Low with rationale",
    "complexity_estimate": "Medium",
    "affected_components": ["Component 1", "Component 2"],
    "status": "Under Review"
  }
}
```

### For general inquiries:
```json
{
  "inquiry_category": "Account Management" | "Billing" | "Usage Question" | "Other",
  "requires_human_review": true | false,
  "suggested_resources": [
    {"title": "Resource Name", "url": "https://example.com"}
  ]
}
```

## Test Cases
1. Bug report: "I can't log in to the web portal. When I enter my password and click login, the button just spins and nothing happens."
2. Feature request: "It would be really useful if the app could send me a notification 15 minutes before a scheduled workout instead of just 5 minutes before."

- FastAPI
- PYDANTIC
- OPENAI
- TENACITY