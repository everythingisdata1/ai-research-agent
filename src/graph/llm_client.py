import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.graph.logger import get_logger

load_dotenv(override=True)
log = get_logger(__name__)


def load_model(temperature=0.0):
    log.info(f"OpenAI Model loaded temperature :{temperature}")
    return ChatOpenAI(temperature=temperature, model=os.getenv("MODEL"))


SEP = "#" * 40
