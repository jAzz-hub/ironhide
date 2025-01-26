# Ironhide

Ironhide simplify the process of building high-quality autonomous agents by representing them as classes.

## Install

```
uv add ironhide
```

## Example

```python
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

    def add(
        self,
        a: Annotated[int, "the first operation number"],
        b: Annotated[int, "the second operation number"],
    ) -> int:
        """Add two integers and returns the result integer."""
        return self.value

    def multiply(
        self,
        a: Annotated[int, "the first operation number"],
        b: Annotated[int, "the second e operation number"],
    ) -> int:
        """Multiply two integers and returns the result integer."""
        return self.value


class Request(BaseModel):
    """User Message to Agent."""

    content: str


class Response(BaseModel):
    """Agent Message to User."""

    result: int


agent = Calculator(value=999)


@app.post("/")
async def agent_message(
    message: Request,
) -> Response:
    """Get response from agent."""
    content = await agent.chat(message.content, response_format=Response)
    return Response(**json.loads(content))
```

<details>

<summary>Last generated API call Json</summary>

```json
{
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "system",
            "content": "You are a function-calling agent designed to calculate expressions through a chain of reasoning. You will receive a mathematical expression, and your task will be to identify and execute the correct functions in the proper order, passing the return values of previously executed functions to subsequent ones that depend on those results to resolve the expression. You are not an agent that performs calculations directly, only one that executes functions to calculate. You are not allowed to infer the result of any operation.",
        },
        {
            "role": "user",
            "content": "2+2",
        },
        {
            "role": "system",
            "content": "Lets think step by step and define the sequence of tools needs to be executed to solve the problem.",
        },
        {
            "role": "assistant",
            "content": "To solve the expression 2 + 2, we need to follow these steps:\n\n1. **Identify the operation**: The operation needed is addition.\n2. **Prepare the parameters**: We need to prepare the parameters for the addition function, which are `a = \"2\"` and `b = \"2\"`.\n3. **Execute the addition**: Call the `functions.add` with the prepared parameters.\n\nNow, I will proceed to execute the addition.",
        },
        {
            "role": "system",
            "content": "Evaluate the previous reasoning to ensure that everything is correct and no operation result is being inferred. If find any issue, explain how to fix it.",
        },
        {
            "role": "assistant",
            "content": "The reasoning provided is correct and has not inferred any results. The operations needed to perform the addition are clearly outlined:\n\n1. Identify the operation\u2014addition is the correct operation for the expression 2 + 2.\n2. Prepare the parameters for the addition function\u2014\"2\" and \"2\" are simply passed as strings.\n3. Execute the addition function using `functions.add`.\n\nThere are no issues with the reasoning or the steps to be executed. I will now proceed to call the `functions.add` function with the parameters `\"2\"` and `\"2\"`.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_hk7EYThfB1rQd698LjSpROu0",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": "{\"a\":\"2\",\"b\":\"2\"}"
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "999",
            "tool_call_id": "call_hk7EYThfB1rQd698LjSpROu0",
        },
        {
            "role": "assistant",
            "content": "The result of the addition operation is 999. If you need further calculations or operations based on this result, please let me know!",
        },
        {
            "role": "system",
            "content": "Evaluate the previous steps and oly approve it if the function calls in the proper order, passing the return values of previously executed functions to subsequent ones that depend on those results to resolve the expression and without infer any result value. Otherwise, reject it and explain how to fix it. You must don't evaluate the correctness of the result.",
        },
        {
            "role": "assistant",
            "content": "Upon evaluating the previous steps, it appears that the function calls were executed in the proper order. The sequence of operations was as follows:\n\n1. The addition operation was correctly identified and parameters were prepared as `a = \"2\"` and `b = \"2\"`.\n2. The `functions.add` function was executed with these parameters.\n\nNo results were inferred throughout the steps; the entire reasoning followed the correct procedure of defining the operation and executing the required function. \n\nThus, I will approve the steps as they appropriately executed the function without improperly inferring any results. If additional steps or operations are needed, please let me know!",
        },
        {
            "role": "assistant",
            "content": "{\"is_approved\":true}",
        },
        {
            "role": "assistant",
            "content": "{\"result\":999}"
        }
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "Response",
            "schema": {
                "description": "Agent Message to User.",
                "properties": {
                    "result": {
                        "title": "Result",
                        "type": "integer"
                    }
                },
                "required": [
                    "result"
                ],
                "title": "Response",
                "type": "object",
                "additionalProperties": false
            },
            "strict": true
        }
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers and returns the result integer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "string",
                            "description": "the first operation number"
                        },
                        "b": {
                            "type": "string",
                            "description": "the second operation number"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "additionalProperties": false
                },
                "strict": true
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_message",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": ""
                        }
                    },
                    "required": [
                        "message"
                    ],
                    "additionalProperties": false
                },
                "strict": true
            }
        },
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two integers and returns the result integer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "string",
                            "description": "the first operation number"
                        },
                        "b": {
                            "type": "string",
                            "description": "the second e operation number"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "additionalProperties": false
                },
                "strict": true
            }
        }
    ],
    "tool_choice": "auto"
}

```
</details>


## Features

- OOP Abstraction
    - The Ironhide core is `BaseAgent` abstract class that convert the child class into a Agent.
- Static Typing
    - All the source code is based on Mypy strict, Ruff and Pydantic.
- Dependency Injection
    -  The child class constructor can receive extra arguments
- Structured Output
    - The agent can have a global or per message output_format
- Chain of Thought
    - Is possible to define reasoning steps before call the methods
- Auto Function Calling
    - Automatically extract the methods information and convert it tools
- Feedback Loop
    - After call the tools, is possible to evaluate the previous behavior and aprove or reject it
