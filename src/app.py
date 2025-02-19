import time
from collections.abc import Generator
from typing import Any

import gradio as gr
from agno.models.message import Message
from dotenv import load_dotenv

from agents import planning_agent

_ = load_dotenv()


def predict(message: str, history: list[dict[str, str]]) -> Generator[Any, Any, None]:
    history_format: list[Message] = []
    for msg in history:
        if msg["role"] == "user":
            history_format.append(
                Message(
                    role="user",
                    content=msg["content"],
                )
            )
        elif msg["role"] == "assistant":
            history_format.append(
                Message(
                    role="assistant",
                    content=msg["content"],
                )
            )

    history_format.append(
        Message(
            role="user",
            content=message,
        )
    )
    gpt_response = planning_agent.run(messages=history_format)
    # return gpt_response.content
    for i in range(len(gpt_response.content)):
        time.sleep(0.01)
        yield gpt_response.content[: i + 1]


TITLE = """
Compound AI System with versatile capabilities
"""
DESCRIPTION = """
This is application allows you to **search for top news, search for tech specific news
from hackernews, teach you C, C++, Rust, and Python, research a particular equity and
give you a guidance to a personal finance, search for a Wikipedia article.**
If you aren't still satisfied with these capabilities,
then you can use the **ASK ME ANYTHING(AMA)** feature.

**Note:** After the first input, there will be a `trash icon` on the top right hand corner
of the chatbox, to clear the entire chat, and below each agent response
there's an `Undo` and `Retry` icon.
"""

if __name__ == "__main__":
    gr.ChatInterface(
        predict,
        title=TITLE,
        description=DESCRIPTION,
        type="messages",
        textbox=gr.Textbox(
            placeholder="Type in a message and press enter...",
            submit_btn=True,
            stop_btn=True,
        ),
    ).launch()
