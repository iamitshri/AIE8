"""Agent graph with tool calling and conditional routing"""

from langgraph.graph import StateGraph, END
from src.agents.agent_state import AgentState
from src.agents.agent_nodes import create_call_model_node, should_continue
from langgraph.prebuilt import ToolNode


def create_agent_graph(model, tools):
    """
    Create an agentic loop with tool calling.
    
    Args:
        model: ChatOpenAI instance with tools bound
        tools: List of tools for the agent
        
    Returns:
        Compiled agent graph
        
    Graph flow:
        START -> agent -> [conditional]
                            - "action" -> tools -> agent (loop)
                            - "end" -> END
    """
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", create_call_model_node(model))
    graph.add_node("action", ToolNode(tools))
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add conditional routing from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",  # If should_continue returns "action", go to "action" node
            "end": END           # If should_continue returns "end", finish
        }
    )
    
    # Add edge from tools back to agent (for multi-turn tool use)
    graph.add_edge("action", "agent")
    
    # Compile and return
    compiled_graph = graph.compile()
    print("Successfully compiled agent graph with conditional routing")
    return compiled_graph