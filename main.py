import os
from dotenv import load_dotenv
from typing import Literal, Union
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
# NOTE: you must use langchain-core >= 0.3 with Pydantic v2

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Voice-Agent"

memory = MemorySaver()

# State

class State(MessagesState):
    summary: str

# Tavily Search Tool Node

search_tool = TavilySearchResults(max_results=2, search_depth="advanced")
tools = [search_tool]

tool_node = ToolNode(tools=[search_tool])

# Model

model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.2
)

# Human Node

# Helper Function
def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )

def doctor_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from Doctor.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_doctor": False,
    }

# Summarization

def summarize_conversation(state: State):
    """
    Summarizes the conversation if the number of messages exceeds 6 messages.
    
    Args:
        state (State): The current conversation state.
        model (object): The model to use for summarization.

    Returns:
        Dict[str, object]: A dictionary containing updated messages.
    """
   
    # Get any existing summary
    summary = state.get("summary", "")

    # Create summarization prompt based on whether there is an existing summary
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add the summarization prompt to the conversation history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=getattr(m, "id", None)) for m in state["messages"][:-2]]
    
    return {"summary": response.content, "messages": delete_messages}


# Conditional Function

def select_next_node(state: State) -> Union[Literal["tools", "summarize", "doctor"], str]:
    # Check if the conversation requires human intervention
    if state["ask_doctor"]:
        return "doctor"

    messages = state["messages"]
    last_message = messages[-1]

    # If there are more than six messages, route to "summarize_conversation"
    if len(messages) > 6:
        return "summarize"
    
    # If the LLM makes a tool call, route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, route to "final" or end
    return END

# Invoke Messages

def call_model(state: State, config: RunnableConfig):

    # Ensure state contains 'messages'
    if "messages" not in state:
        raise ValueError("State must contain a 'messages' key.")
    
    # Initialize messages from the state
    messages = state["messages"]
    
    # Check if a summary exists and prepend it as a system message if present
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + messages

    # Safely invoke the model
    try:
        response = model.invoke(messages, config)

    except Exception as e:
        raise RuntimeError(f"Error invoking the model: {e}")

    # Append the response to messages
    messages.append(response)

    # Return the updated state with messages
    return {"messages": messages[-1]}

# Build Graph

builder = StateGraph(State)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("doctor", doctor_node)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "agent")
builder.add_edge("tools", "agent")
builder.add_edge("doctor", "agent")
builder.add_edge("summarize", END)

builder.add_conditional_edges(
    "agent",
    select_next_node,
    {"summarize": "summarize", "tools": "tools", END: END},
)

graph = builder.compile(checkpointer=memory)
