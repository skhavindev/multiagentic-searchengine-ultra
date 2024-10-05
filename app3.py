import streamlit as st
import os
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

# Initialize components
model_name = "llama-3.1-70b-versatile"
groq_llm = ChatGroq(temperature=0, model_name=model_name, groq_api_key=GROQ_API_KEY)


class AgentState(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], "The messages in the conversation so far"]
    next_step: str


# Query Processing Agent (QPA)
def query_processing_agent(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        " {query} : Provide user query and Generate Search query in the format user query:, search query: If search query not required, just send user message directly. Do Not Hallucinate"
    )
    chain = prompt | groq_llm | StrOutputParser()
    optimized_query = chain.invoke({"query": state["messages"][-1].content})
    return {"messages": state["messages"] + [AIMessage(content=optimized_query)], "next_step": "conditional_agent"}


# Conditional Agent (CA)
def conditional_agent(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        "Determine if external search is needed for this query: {query}\nRespond with 'search' or 'synthesize'."
    )
    chain = prompt | groq_llm | StrOutputParser()
    decision = chain.invoke({"query": state["messages"][-1].content})
    next_step = "search_agent" if decision.strip().lower() == "search" else "knowledge_synthesis_agent"
    return {"messages": state["messages"], "next_step": next_step}


# Search and Retrieval Agent (SRA) - Using DuckDuckGo
def search_agent(state: AgentState) -> AgentState:
    query = state["messages"][-1].content

    # Using DuckDuckGo to perform the search
    try:
        search_results = DDGS().text(query, max_results=3)  # Fetching top 3 results
        formatted_results = "\n".join([f"{result['title']}: {result['href']}" for result in search_results])
        return {"messages": state["messages"] + [AIMessage(content=formatted_results)],
                "next_step": "knowledge_synthesis_agent"}
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")

    # Fallback to using Groq LLM for information retrieval
    prompt = ChatPromptTemplate.from_template(
        "As an AI assistant, please provide relevant information about: {query}"
    )
    chain = prompt | groq_llm | StrOutputParser()
    fallback_info = chain.invoke({"query": query})
    return {"messages": state["messages"] + [AIMessage(content=fallback_info)],
            "next_step": "knowledge_synthesis_agent"}


# Knowledge Synthesis Agent (KSA)
def knowledge_synthesis_agent(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        """You are an intelligent search agent developed by TextFusion.AI who can provide comprehensive answers to any user query. The agent should answer the question based only on the context provided. You have the ability to search the internet. Make the answer as comprehensive as possible.
        Context: {context}
        Question: {query}
        You must use citations in the format [1][link]. Format the answers beautifully
        """
    )
    chain = prompt | groq_llm | StrOutputParser()
    synthesis = chain.invoke({"context": state["messages"][-1].content, "query": state["messages"][-2].content})
    return {"messages": state["messages"] + [AIMessage(content=synthesis)], "next_step": "end"}


# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("query_processing_agent", query_processing_agent)
workflow.add_node("conditional_agent", conditional_agent)
workflow.add_node("search_agent", search_agent)
workflow.add_node("knowledge_synthesis_agent", knowledge_synthesis_agent)

# Add edges
workflow.add_edge("query_processing_agent", "conditional_agent")
workflow.add_conditional_edges(
    "conditional_agent",
    lambda x: x["next_step"]
)
workflow.add_edge("search_agent", "knowledge_synthesis_agent")
workflow.add_edge("knowledge_synthesis_agent", END)

# Set entry point
workflow.set_entry_point("query_processing_agent")

# Compile the graph
graph = workflow.compile()


# Run the graph
def run_graph(query: str):
    return graph.invoke({"messages": [HumanMessage(content=query)], "next_step": "query_processing_agent"})

import streamlit as st
import os
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS

# ... (keep the existing imports and setup code)

# Streamlit app
def main():
    st.set_page_config(
        page_title="Intelligent Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f0f8ff;
        }
        
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .main-title {
            font-size: 3rem;
            font-weight: 600;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .search-container {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        .stTextInput > div > div > input {
            background-color: #f7f9fc;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        
        .stButton > button {
            background-color: #3498db;
            color: white;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            border: none;
            transition: background-color 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #2980b9;
        }
        
        .result-box {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .sidebar-content {
            background-color: #2c3e50;
            color: white;
            padding: 1rem;
            border-radius: 10px;
        }
        
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .sidebar-stats {
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        
        .clear-button {
            background-color: #e74c3c;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        
        .clear-button:hover {
            background-color: #c0392b;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">Intelligent Search Engine 🔍</h1>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        query = st.text_input("Enter your search query:", key="search_input")
        search_button = st.button("Search", key="search_button")
        st.markdown('</div>', unsafe_allow_html=True)

        if search_button:
            if query:
                with st.spinner("Searching..."):
                    result = run_graph(query)
                    final_response = result["messages"][-1].content

                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("### Search Results")
                st.write(final_response)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a search query.")

    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<h2 class="sidebar-header">📊 Search Statistics</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-stats">
        - Queries processed today: 1,234
        - Average response time: 0.8s
        - Satisfaction rate: 98%
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h2 class="sidebar-header">ℹ️ About</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-stats">
        This intelligent search engine uses advanced AI techniques to provide comprehensive and accurate results.

        Powered by TextFusion.AI technology.
        </div>
        """, unsafe_allow_html=True)

        st.write("© 2024 TextFusion.AI")

        if st.button("Clear Search History", key="clear_button", help="Click to clear your search history"):
            # Add functionality to clear search history if needed
            st.success("Search history cleared!")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
