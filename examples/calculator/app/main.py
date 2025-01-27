import json
from typing import Annotated, Any

from fastapi import FastAPI
from ironhide import BaseAgent, tool
from pydantic import BaseModel

app = FastAPI()


class Request(BaseModel):
    """User Message to Agent."""

    content: str


class Response(BaseModel):
    """Agent Message to User."""

    result: int


class Calculator(BaseAgent):
    instructions = """You are a function-calling agent designed to calculate expressions through a chain of reasoning. You will receive a mathematical expression, and your task will be to identify and execute the correct functions in the proper order, passing the return values of previously executed functions to subsequent ones that depend on those results to resolve the expression. You are not an agent that performs calculations directly, only one that executes functions to calculate. You are not allowed to infer the result of any operation."""

    chain_of_thought = (
        "Lets think step by step and define the sequence of tools needs to be executed to solve the problem.",
        "Evaluate the previous reasoning to ensure that everything is correct and no operation result is being inferred. If find any issue, explain how to fix it.",
    )

    feedback_loop = "Evaluate the previous steps and oly approve it if the function calls in the proper order, passing the return values of previously executed functions to subsequent ones that depend on those results to resolve the expression and without infer any result value. Otherwise, reject it and explain how to fix it. You must don't evaluate the correctness of the result."

    def __init__(self, value: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value


agent = Calculator(value=999)


@app.post("/")
async def agent_message(
    message: Request,
) -> Response:
    """Get response from agent."""
    content = await agent.chat(message.content, response_format=Response)
    return Response(**json.loads(content))
