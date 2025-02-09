from typing import TypedDict

from langgraph.graph import END, StateGraph, START


class State(TypedDict):
    topic: str
    joke: str


def refined_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_jokes(state: State):
    return {"joke": f"the joke is about {state['topic']}"}


workflow = StateGraph(State)
workflow.add_node("refined_topic", refined_topic)
workflow.add_node("generate_jokes", generate_jokes)
workflow.add_edge(START, "refined_topic")
workflow.add_edge("refined_topic", "generate_jokes")
workflow.add_edge("generate_jokes", END)
graph = workflow.compile()

### Start Streaming / It got different types


# Stream all values in the state (stream_mode="values")
for output in graph.stream(
    {"topic": "dogs"}
):
    print(output)
# Output: {'refined_topic': {'topic': 'dogs and cats'}} {'generate_jokes': {'joke': 'the joke is about dogsand cats'}}

# Stream state updates from the nodes (stream_mode="updates")
for output in graph.stream(
    {"topic": "dogs"}, stream_mode="updates"
):
    print(output)
# Output: {'refined_topic': {'topic': 'dogs and cats'}} {'generate_jokes': {'joke': 'the joke is about dogs and cats'}}
    
