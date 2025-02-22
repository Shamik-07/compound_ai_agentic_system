import json
from pathlib import Path
from shutil import rmtree
from textwrap import dedent
from typing import Any

import wikipedia
from dotenv import load_dotenv
from openai import OpenAI

_ = load_dotenv()
openai_client = OpenAI()


def moderate_content(text: str) -> str:
    """Use this function to moderate messages.

    Args:
        text (str): Messages sent by user or responses from model.

    Returns:
        bool: JSON string of True or False.
    """
    response = openai_client.moderations.create(
        model="omni-moderation-latest", input=text
    )
    return json.dumps(response.results[0].flagged)


def create_finance_reports_dir() -> Path:
    reports_dir = Path(__file__).parent.joinpath("finance_agent", "reports")
    if reports_dir.exists():
        rmtree(path=reports_dir, ignore_errors=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def search_on_wikipedia(query: str) -> Any | str | Any:
    try:
        content = wikipedia.page(title=query, auto_suggest=False).summary
        return content
    except wikipedia.DisambiguationError as err:
        return (
            f"Your query resulted to the following topics: {err.options}. "
            + "Which one do you want to know about?"
        )
    except wikipedia.PageError:
        if len(search_result := wikipedia.search(query)):
            return dedent(
                f"The query didn't match an exact page but\
            these are the closest search results: {search_result}"
            )
        else:
            return (
                f"No search results for: {query}. " + "Please try and be more specific."
            )


def delete_exisiting_chat_history() -> None:
    filepath = Path(__file__).parent.joinpath("agent_storage.db")
    if filepath.exists():
        filepath.unlink()
    return
