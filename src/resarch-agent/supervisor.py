import json
from typing import Dict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send

from search_agent import search_agent
from src.graph.llm_client import SEP, load_model
from src.graph.logger import get_logger
from src.graph.save_graph import save_graph
from src.state import ResearchState
from state import QualityReviewState

load_dotenv(override=True)
log = get_logger(__name__)

log.info(f"Research Supervisor Agent")

log.info(SEP)
log.info("[RESEARCHER STATE]")

llm_model = load_model()
creative_llm = load_model(temperature=0.7)


def create_supervisor_agent():
    """ Plans research by generating   targeted search queries .
    FLOW:
    1- Supervisor plan search queries.
    2- Search Agent will run in parallel by SEND API
    3- Analyst synthesized the findings
    4- Report Writer produce repor
    5- Quality checker review - Loop back if needed"""

    def supervisor_agent(state: ResearchState) -> Dict:
        """ Analysis User topic type and call the agent accordingly """
        # Get most recent message

        prompt = """ You are a Research Supervisor. Given a topic, Generate exactly 3 search query 
        that will cover different angle of the topic. 
        Return only a JSON array of strings. No Markdown formating. 
           """
        resp = llm_model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f'Research Topic: {state.get("topic", "")}')
        ])

        log.info(f"[SUPERVISOR AGENT]: {resp.content}")
        try:
            queries = json.loads(resp.content)
        except json.JSONDecodeError:
            # Fallback: split by newlines
            queries = [
                f"{state.get("topic")} overview",
                f"{state.get("topic")} latest development",
                f"{state.get("topic")} practical applications",

            ]
        print(f"Supervisor :: {queries}")
        return {
            "search_queries": queries[:3],
            "messages": [AIMessage(
                content=f"[SUPERVISOR]: Planned {len(queries)} research queries: {queries}",
                name="supervisor"
            )
            ]
        }

    def despatch_searches(state: ResearchState) -> list[Send]:
        """ Dynamically create parallel search tasks using Send API"""
        return [Send("searchAgent", {"search_query": q, 'findings': []}) for q in state.get("search_queries")]

    # ============================================================
    # Node Analyst: Synthesize all the findings
    # ============================================================

    def report_analyst(state: ResearchState) -> dict:
        """ Read all findings from the blackboard and synthesize..."""
        findings_txt = json.dumps(state.get("findings"), indent=2)
        system_prompt = """
        You are a Research analyst.  Synthesize the collected findings into a clear analysis.
        Identify:
        1- Key theme across the findings.\n
        2- Any contradiction or gaps.\n
        3- The most important finding.. \n\n
        
        Write 2-3 paragraphs 
        """
        human_prompt = f"""Research Topic:: {state.get("topic")}\n\n Collected Findings:: {findings_txt}"""

        resp = llm_model.invoke([
            SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)
        ])

        return {
            "analysis": resp.content,
            "messages": [AIMessage(content=f"[ANALYST]: {resp.content}", name="analyst")]
        }

    # =================================================================
    # Node Report Writer: Produce final Report
    # =================================================================
    def report_writer_agent(state: ResearchState) -> dict:
        """ Writes a structured research report from the analysis """

        # Include quality feedback if this a revision
        revision_note = ""
        if state.get('iteration') > 0 and state.get("quality_feedback"):
            revision_note = f"""IMPORTANT - This is revision :: {state.get("iteration")}.
                                Address this feedback. {state.get('quality_feedback')}"""

        prompt = f"""You are a AI Report Writer Agent. Produce a well-structured research report with the below sections.
        1- Executive Summery (2-3 sentences).\n
        2- Key findings (bullet points) \n
        3- Analysis (1-2 paragraphs) \n
        4- Recommendations (3 actionable items)\n
        Use Markdown formatting. Be specific and activable  
        {revision_note}"""

        human_prompt = f"""
        TOPIC:: \n{state.get("topic")}\n\n
        Analysis:: \n {state.get("analysis")}\n\n
        RAW Findings:  \n{json.dumps(state.get("findings")[:6], indent=2)}                """
        resp = creative_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=human_prompt),
        ])
        return {
            "report": resp.content,
            "messages": [
                AIMessage(content=f"[REPORT] : Report  {'revised' if state.get('iteration') > 0 else 'drafted'}")
            ]}

    def quality_checker_agent(state: ResearchState) -> dict:
        """Review the report and either approved or send back fos for revision """

        llm_output = llm_model.with_structured_output(QualityReviewState)

        prompt = """
        You are a AI Quality Reviewer Agent. Score the report on 
        - Completeness: Does it cover the topic well..\n 
        - Clarify: Is it well-written and understandable?\n\n
        - Actionability: Are recommendations specific ?
        Score from 0.0 to 1.0. And Approve if score>=0.7
        If the iteration 2 or higher, be more lenient.  """

        human_prompt = f"""
        Topic: \n{state.get('topic')}\n\n 
        Iteration:: {state.get('iteration')}\n\n 
        Report:: \n {state.get("report")}\n\n
        """

        resp = llm_output.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=human_prompt)
        ])

        # force approve after 2 iteration
        approved = resp.approved or state.get('iteration') >= 2

        return {
            "quality_score": resp.score,
            "quality_feedback": resp.feedback,
            'iteration': state.get('iteration') + 1,
            'messages': [
                AIMessage(
                    content=f"[QUALITY CHECK]:: Score {resp.score:.1f} - "
                            f" {'APPROVED' if approved else 'REVISION NEEDED'} : {resp.feedback}",
                    name="quality_check"
                )
            ]
        }

    MAX_REVISIONS = 2
    QUALITY_THRESHOLD = 0.7

    def quality_gate(state: ResearchState) -> str:
        """Approve the report or request another revision."""

        if state.get("quality_score", 0.0) >= QUALITY_THRESHOLD:
            return "end"

        if state.get("iteration", 0) >= MAX_REVISIONS:
            return "end"

        return "ReportWriterAgent"

    researcher_build = StateGraph(ResearchState)

    researcher_build.add_node("SupervisorAgent", supervisor_agent)
    researcher_build.add_node("searchAgent", search_agent)
    researcher_build.add_node("AnalystAgent", report_analyst)
    researcher_build.add_node("ReportWriterAgent", report_writer_agent)
    researcher_build.add_node("qualityAgent", quality_checker_agent)

    # Graph EDGES
    researcher_build.add_edge(START, "SupervisorAgent")

    # Supervisor-Parallel search agent (Dynamicaly  Fan-out)
    researcher_build.add_conditional_edges("SupervisorAgent", despatch_searches, ["searchAgent"])

    # All search agents - analyst (fan-in)
    researcher_build.add_edge("searchAgent", "AnalystAgent")

    # Analyst - Report Writer
    researcher_build.add_edge("AnalystAgent", "ReportWriterAgent")

    # Reprot Writer to  Quality check
    researcher_build.add_edge("ReportWriterAgent", "qualityAgent")

    researcher_build.add_conditional_edges("qualityAgent",
                                           quality_gate,
                                           {
                                               "ReportWriterAgent": "ReportWriterAgent",
                                               "end": END,
                                           },
                                           )

    app = researcher_build.compile()

    save_graph(app, indx=2)
    return app


def test_research_with_stream():
    """ Run Research System  with Step-by-step streaming output"""

    system = create_supervisor_agent()
    topic = "Best practices to build multi agent ai systems"

    log.info(SEP)
    log.info(f"Streaming Topic:: {topic}")
    initial_state = ResearchState(
        messages=[],
        topic=topic,
        search_queries=[],
        findings=[],
        iteration=0,
        analysis="",
        quality_score=0.0,
        quality_feedback="",
        report="")

    for step in system.stream(initial_state, stream_mode="updates"):
        for node_name, update in step.items():
            print(SEP)
            print(f"[{node_name}] completed")

            # Show interesting state changes
            if "search_queries" in update and update["search_queries"]:
                print(f"  Planned queries: {update['search_queries']}")
            if "findings" in update and update["findings"]:
                print(f"  Found {len(update['findings'])} results")
            if "quality_score" in update:
                print(f"  Quality score: {update['quality_score']:.1f}")
            if "report" in update and update["report"]:
                print(f"  Report length: {len(update['report'])} chars")

        print()


if __name__ == '__main__':
    log.info("Test Supervisor Agent")
    test_research_with_stream()
    # questions = [
    #     "Explain Retrieval-Augmented Generation (RAG).",
    #     "How do interest rates affect bond prices?",
    #     "Explain Kubernetes deployments.",
    # ]
    #
    # for question in questions:
    #     result = app.invoke({
    #         "messages": [HumanMessage(content=question)],
    #         "iteration": 0,
    #         "findings": [],
    #     })
    #
    #     print("Query:", question)
    #     print("Topic:", result.get("topic"))
    #     print("Category:", result.get("category"))
    #     print("Analysis:", result.get("analysis"))
    #     print("Report:", result.get("report"))
    #     print("Quality feedback:", result.get("quality_feedback"))
    #     print("Quality score:", result.get("quality_score"))
