from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage ,HumanMessage,AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START,StateGraph
from langgraph.prebuilt import tools_condition
from langchain_groq import ChatGroq
from langchain.tools import Tool
from huggingface_hub import list_models
import random
from dotenv import load_dotenv
import os
from langchain_community.utilities import  SerpAPIWrapper
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
#os.environ["SERPAPI_API_KEY"]=os.getenv("SERPAPI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
serp_api_key=os.getenv("SERPAPI_API_KEY")

from langchain_community.utilities import SerpAPIWrapper

search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
search_tool = Tool(
    name="SerpAPI Search",
    func=search.run,
    description="Search the web using SerpAPI"
)

    



### weather tool
def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)



##most downloaded 
def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)










### langchain
import datasets
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage,HumanMessage,AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START,StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint ,ChatHuggingFace
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]






bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)

# Generate the chat interface , including the tools
llm = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
tools = [guest_info_tool,search_tool,weather_info_tool,hub_stats_tool]
llm_with_tools = llm.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = alfred.invoke({"messages": messages})



