from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuild import ToolNode
from langgraph.types import interrupt


@tool
def search(query: str):
    return f"I looked up: {query}. Result: It's sunny in San Francisco, but you better look out if you're a Gemini 😈."


tools = [search]
tool_node = ToolNode(tools)

# Set up the model
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

from pydantic import BaseModel


class AskHuman(BaseModel):
    question: str


model = model.bind_tools(tools + [AskHuman])


# Define the nodes and conditional graphs
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there's no function calls, then we finish
    if not last_message.tool_calls:
        return END
    # If the tool call is asking Human, we return that node
    # You could also add logic here to let some system know that there's something that requires Human Input
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    else:
        return "action"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# Define a fake node to ask the human
def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    location = interrupt("Please provide your location:")
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": location}]
    return {"messages": tool_message}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the three nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("ask_human", ask_human)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")
