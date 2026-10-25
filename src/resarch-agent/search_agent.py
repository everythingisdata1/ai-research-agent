import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.graph.llm_client import SEP, load_model
from src.graph.logger import get_logger
from src.state import ResearchState

load_dotenv(override=True)
log = get_logger(__name__)

log.info(SEP)
log.info("[SEARCH AGENT]")

llm_model = load_model()


def search_agent(state: ResearchState):
    """
    Executes one search query and return findings.
    Each agent runs in parallel via SEND API
    """
    query = state.get("search_query")
    prompt = """You are a web research agent. 
    For the given search query. provide 2-3 key findings.
    Each findings should have 'title' and 'details' field.
    Return a JSON array not Markdown reason
    """
    print(prompt)
    response = llm_model.invoke([
        HumanMessage(content=prompt),
        AIMessage(content=f"Search Query:  {query}"),
    ])

    try:
        results = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: split by newlines
        results = [{'title': query, 'details': resp.content}]

    # Tag each finding with the query it came from
    for r in results:
        r['search_query'] = query
    return {"findings": results}


if __name__ == '__main__':
    """Demo jsut seach Agent"""

    resp = search_agent({
        'search_query': 'Langgraph multi-agent patterns',
        "findings": []
    })
    print(f"Findings from seach ")
    for f in resp.get('findings'):
        print(f)
