import operator

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict, Annotated

Category = Literal["technology", "finance", "None"]


class ResearchState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    search_queries: list[str]
    findings: Annotated[list[str], operator.add]
    analysis: str
    report: str
    quality_score: float
    quality_feedback: str
    iteration: int


# State For Individual Search Tasks (used with Send API)
class SearchTaskState(TypedDict):
    search_query: str
    findings: Annotated[list[str], operator.add]


class AgentDecisionState(BaseModel):
    category: Category = Field(
        description="Category of the user's topic. Use none if unrelated to Finance or Technology."
    )
    reasoning: str = Field(description="Specific reasons for the topic classification")


class QualityReviewState(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Quality Score of report.")
    feedback: str = Field(description="Specific feedback for improvement")
    approved: bool = Field(description="Whether the report meets the quality standards.")
