import os
from dotenv import load_dotenv
from typing import Literal, Union
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['SERPER_API_KEY'] =  os.getenv('SERPER_API_KEY')

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Voice-Agent"

memory = MemorySaver()

# State

class State(MessagesState):
    summary: str

#Search Tool Node

search = GoogleSerperAPIWrapper()

# Define a tool function
@tool
def search_tool(query: str) -> str:
    """Search the web using SerpAPI."""
    return search.run(query)

# Custom tool for vector search + response generation

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = PineconeVectorStore.from_existing_index(index_name="demo", embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash", streaming=True)

@tool
def vector_search_tool(query: str) -> str:
    """
    Searches restaurant data using vector embeddings and generates a relevant AI response.
    Args:
        query (str): The user's query about menu items, descriptions, or ratings.
    Returns:
        str: AI-generated response based on retrieved restaurant data.
    """

    # Retrieve relevant docs from Pinecone
    retrieved_docs = retriever.invoke(query)
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    # Generate AI response with context
    prompt = f"Use the following restaurant data to answer: {context} \n\nCustomer: {query} \nAI:"
    response = llm.invoke(prompt)

    return response.content

tools = [search_tool, vector_search_tool]

tool_node = ToolNode(tools=[search_tool, vector_search_tool])

# Model

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2
)

# We can bind the llm to a tool definition
model = model.bind_tools(tools)

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

def select_next_node(state: State) -> Union[Literal["tools", "summarize"], str]:

    messages = state["messages"]
    last_message = messages[-1]

    # Route to "tools" node if the message relates to web search or restaurant data
    if any(keyword in last_message.content.lower() for keyword in ["qarmashi", "menu", "dish", "rating", "food", "weather", "update", "search"]):
        return "tools"

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
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "agent")
builder.add_edge("tools", "agent")
builder.add_edge("summarize", END)

builder.add_conditional_edges(
    "agent",
    select_next_node,
    {"summarize": "summarize", "tools": "tools", END: END},
)

graph = builder.compile(checkpointer=memory)

# Run the Graph

user_input = "Do you have Truffle Mushroom Burger in menu?"
config = {"configurable": {"thread_id": "1"}} # you can make this thread ID unique for each call by giving call sid

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
