from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel

# Input models
class CustomerMessage(BaseModel):
    """Customer message input model."""
    customer_id: str
    message: str
    product: str

# Structured data models
class SuggestedResource(BaseModel):
    """Model for suggested resources in general inquiries."""
    title: str
    url: str

class BugTicket(BaseModel):
    """Model for bug ticket data."""
    id: str
    title: str
    severity: Literal["Low", "Medium", "High", "Critical"]
    affected_component: str
    reproduction_steps: List[str]
    priority: Literal["Low", "Medium", "High"]
    assigned_team: str

class FeatureRequest(BaseModel):
    """Model for feature request data."""
    id: str
    title: str
    description: str
    user_story: str
    business_value: str
    complexity_estimate: Literal["Low", "Medium", "High"]
    affected_components: List[str]
    status: str = "Under Review"

class GeneralInquiry(BaseModel):
    """Model for general inquiry data."""
    inquiry_category: Literal["Account Management", "Billing", "Usage Question", "Other"]
    requires_human_review: bool
    suggested_resources: List[SuggestedResource]

# Response model
class CustomerResponse(BaseModel):
    """Response model for processed customer messages."""
    message_type: Literal["bug_report", "feature_request", "general_inquiry"]
    confidence_score: float
    response_data: Dict[str, Any]
    customer_response: str