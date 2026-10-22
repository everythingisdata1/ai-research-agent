from typing import Dict, Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.graph.llm_client import SEP, load_model
from src.graph.logger import get_logger
from src.graph.save_graph import save_graph
from src.state import AgentDecisionState, ResearchState, QualityCheckState

load_dotenv(override=True)
log = get_logger(__name__)

log.info(f"Research Supervisor Agent")

log.info(SEP)
log.info("[RESEARCHER STATE]")

llm_model = load_model()


def create_supervisor_agent():
    """ Create Supervisor Agent to Research on Agent."""

    def supervisor_agent(state: ResearchState) -> Dict:
        """ Analysis User topic type and call the agent accordingly """
        # Get most recent message
        message = next((msg for msg in reversed(state.get("messages", [])) if isinstance(msg, HumanMessage)), None)
        log.info(f"Human message::\n {message}")

        if not message:
            return {"category": "None", "reasoning": "No research topic was provided."}

        prompt = """
           Classify the user's research question.\n
           technology: software, programming, AI, cloud, data engineering.
           finance: banking, investments, financial markets, economics.
           None: unrelated to both categories.\n\n Return the category in lowercase. Do not assume.
           """

        llm_structure = llm_model.with_structured_output(AgentDecisionState)
        resp = llm_structure.invoke([SystemMessage(content=prompt), HumanMessage(content=message.content)])

        log.info(f"[SUPERVISOR AGENT]: Category: {resp.category} reason: {resp.reasoning}")

        return {
            "topic": resp.category,
            "category": [resp.category],
            "reasoning": resp.reasoning
        }

    def route_to_agent(state: ResearchState) -> Literal["technology", "finance", "end"]:
        agent = state.get("category", None)
        if agent in ["technology", "finance"]:
            return agent
        else:
            return "end"

    def finance_agent(state: ResearchState) -> dict:
        prompt = """You are finance specialized agent.
        please review the given topic and write 2-3 line on topic
        precisely  """

        message = next((msg for msg in reversed(state.get("messages", [])) if isinstance(msg, HumanMessage)), None)

        resp = llm_model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"[TOPIC]:: {message}")
        ])
        log.info(f"[FINANCE AGENT]: {resp.content}")
        return {
            "findings": [resp.content],
        }

    def tech_agent(state: ResearchState) -> dict:
        prompt = """You are Technology Specialized Agent.
        please review the given topic and write 2-3 line on topic in list [line1, line2, lin3]
        precisely  """

        message = next((msg for msg in reversed(state.get("messages", [])) if isinstance(msg, HumanMessage)), None)

        resp = llm_model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"[TOPIC]:: {message}")
        ])
        return {
            "findings": [resp.content],
        }

    def analyst_agent(state: ResearchState) -> dict:
        prompt = """You are a AI Analyst Agent. You job is to validate findings regarding  topic and message, and precisely 
        help to the research accuracy. in line 2-3   """

        message = next((msg for msg in reversed(state.get("messages", [])) if isinstance(msg, HumanMessage)), None)

        findings = "\n\n".join(state.get("findings"))

        resp = llm_model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=(
                f"Question: {message.content}\n"
                f"Category: {state.get('category')}\n"
                f"Findings:\n{findings}"
            )),
        ])

        return {
            "analysis": resp.content,
            "iteration": state.get("iteration", 0) + 1,
        }

    def report_writer_agent(state: ResearchState) -> dict:
        prompt = """You are a AI Report Reviewer Agent.  
        Review the findings and generate report for the  given user message . in line 2-3   """

        # message = next((msg for msg in reversed(state.get("messages", [])) if isinstance(msg, HumanMessage)), None)
        #
        # findings = "\n\n".join(state.get("findings"))
        resp = llm_model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=(
                f"Question: {state.get('topic', '')}\n"
                f"Category: {state.get('category', '')}\n"
                f"Findings: {state.get('findings', [])}\n"
                f"Analysis: {state.get('analysis', '')}"
            )),
        ])
        return {"report": resp.content}

    def quality_checker_agent(state: ResearchState) -> dict:
        prompt = """You are a AI Report Reviewer Agent  Agent. You job is to validate finding regarding  topic and message, and pricelessly 
        help with quality score between (1-10). in line 2-3 """

        message = next((msg for msg in reversed(state.get("messages", [])) if isinstance(msg, HumanMessage)), None)

        findings = "\n\n".join(state.get("findings"))
        llm_output = llm_model.with_structured_output(QualityCheckState)

        resp = llm_output.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"[MESSAGE]:\n {message}, [Topic]:\n {state.get("topic")} [Findings]:\n {findings}")
        ])
        log.info(SEP)
        log.info(f"Feedback:\n {resp.score}")
        log.info(f"Reasoning:\n {resp.reasoning}")
        return {
            "quality_score": resp.score,
            "quality_feedback": resp.reasoning,
        }

    def quality_check_route(state: ResearchState) -> Literal["SupervisorAgent", "end"]:
        score = state.get("quality_score", 0)
        quality_feedback = state.get("quality_feedback", 0)
        iteration = state.get("iteration")
        if score <= 7 and iteration < 2:
            return "SupervisorAgent"
        return "end"

    def revision_agent(state: ResearchState) -> dict:
        """ Revise the existing report by  using the quality check feedback"""
        prompt = """ 
        
        
        """

        message = next((msg for msg in reversed(state.get("messages", [])) if isinstance(msg, HumanMessage)), None)

        findings = "\n\n".join(state.get("findings"))
        llm_output = llm_model.with_structured_output(QualityCheckState)

        resp = llm_output.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"[MESSAGE]:\n {message}, [Topic]:\n {state.get("topic")} [Findings]:\n {findings}")
        ])
        log.info(SEP)
        log.info(f"Feedback:\n {resp.score}")
        log.info(f"Reasoning:\n {resp.reasoning}")
        return {
            "quality_score": resp.score,
            "quality_feedback": resp.reasoning,
        }

    researcher_build = StateGraph(ResearchState)

    researcher_build.add_node("SupervisorAgent", supervisor_agent)
    researcher_build.add_node("TechAgent", tech_agent)
    researcher_build.add_node("FinAgent", finance_agent)
    researcher_build.add_node("AnalystAgent", analyst_agent)
    researcher_build.add_node("ReportWriterAgent", report_writer_agent)
    researcher_build.add_node("CheckerAgent", quality_checker_agent)

    researcher_build.add_edge(START, "SupervisorAgent")
    researcher_build.add_conditional_edges("SupervisorAgent", route_to_agent, {
        "technology": "TechAgent",
        "finance": "FinAgent",
        "end": END,
    })
    researcher_build.add_edge("TechAgent", "AnalystAgent")
    researcher_build.add_edge("FinAgent", "AnalystAgent")
    researcher_build.add_edge("AnalystAgent", "ReportWriterAgent")
    researcher_build.add_edge("ReportWriterAgent", "CheckerAgent")

    researcher_build.add_conditional_edges("CheckerAgent", quality_check_route, {
        "SupervisorAgent": "SupervisorAgent",
        "end": END
    })

    app = researcher_build.compile()

    save_graph(app, indx=2)
    return app


if __name__ == '__main__':
    log.info("Test Supervisor Agent")
    app = create_supervisor_agent()
    questions = [
        "Explain Retrieval-Augmented Generation (RAG).",
        "How do interest rates affect bond prices?",
        "Explain Kubernetes deployments.",
    ]

    for question in questions:
        result = app.invoke({
            "messages": [HumanMessage(content=question)],
            "iteration": 0,
            "findings": [],
        })

        print("Query:", question)
        print("Topic:", result.get("topic"))
        print("Category:", result.get("category"))
        print("Analysis:", result.get("analysis"))
        print("Report:", result.get("report"))
        print("Quality feedback:", result.get("quality_feedback"))
        print("Quality score:", result.get("quality_score"))
