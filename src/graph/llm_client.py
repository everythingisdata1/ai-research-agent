from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.graph.logger import get_logger

load_dotenv(override=True)
log = get_logger(__name__)


def load_model():
    log.info("OpenAI Model loaded")
    return ChatOpenAI(temperature=0.0)


SEP = "=" * 40
