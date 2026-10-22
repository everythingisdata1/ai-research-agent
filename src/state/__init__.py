from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict, Annotated

Category = Literal["technology", "finance", "None"]


class ResearchState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    category: Category
    search_queries: list
    findings: list[str]
    reasoning: str
    analysis: str
    report: str
    quality_score: str
    quality_feedback: str
    iteration: int


class AgentDecisionState(BaseModel):
    category: Category = Field(
        description="Category of the user's topic. Use none if unrelated to Finance or Technology."
    )
    reasoning: str = Field(description="Specific reasons for the topic classifcation")


class QualityCheckState(BaseModel):
    score: int = Field(ge=1, le=10, description="Quality score from 1 to 10")
    reasoning: str = Field(description="Specific reasons for the score")
