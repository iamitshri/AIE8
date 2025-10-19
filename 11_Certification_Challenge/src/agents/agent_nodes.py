"""Nodes for the agentic loop with tool calling"""

from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from src.agents.agent_state import AgentState


def create_call_model_node(model):
    """
    Factory function to create a call_model node.
    
    Args:
        model: ChatOpenAI instance with tools bound
        
    Returns:
        Function that invokes the model
    """
    def call_model(state: AgentState):
        """Call the LLM with the current messages"""
        messages = state["messages"]
        print(f"Agent calling model with {len(messages)} messages")
        response = model.invoke(messages)
        return {"messages": [response]}
    
    return call_model


def should_continue(state: AgentState):
    """
    Routing function to determine next step.
    
    Args:
        state: Current agent state
        
    Returns:
        str: "action" to call tools, "end" to finish
    """
    last_message = state["messages"][-1]
    
    # If LLM makes a tool call, route to tools
    if last_message.tool_calls:
        print(f"Agent wants to use tool: {last_message.tool_calls[0]['name']}")
        return "action"
    
    # Otherwise, end the conversation
    print("Agent finished - no more tool calls")
    return "end"