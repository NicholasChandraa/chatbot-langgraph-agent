"""
Custom Handoff Tool with Task Description
Allows supervisor to pass detailed context to sub-agents
"""
from typing import Annotated

from langchain.tools import tool, BaseTool, InjectedToolCallId
from langchain.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION

from app.utils.logger import logger

def create_custom_handoff_tool(
    *,
    agent_name: str,
    name: str | None = None,
    description: str | None = None
) -> BaseTool:
    """
    Create custom handoff tool with task description support.

    Args:
        agent_name: Target agent identifier
        name: Tool name (optional, defaults to f"handoff_to_{agent_name})
        description: Tool description
    
    Returns:
        BaseTool configured for handoff with task description
    """

    if name is None:
        name = f"handoff_to_{agent_name}"

    if description is None:
        description = f"Handoff task to {agent_name} with detailed context"
        
    @tool(name, description=description)
    def handoff_to_agent(
        task_description: Annotated[
            str, 
            "Detailed description of what the next agent should do, including all relevant context from the user's question"
        ],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """
        Handoff to target agent with task description.

        Args:
            task_description: Clear task for the next agent
            state: Current state from supervisor
            tool_call_id: Tool call identifier

        Returns:
            Command to execute handoff
        """
        # Get current agent from state
        current_agent = state.get("active_agent", "unknown")

        logger.info(f"🔄 [HANDOFF] {current_agent} → {agent_name}")
        logger.info(f"📋 [TASK] {task_description}")
        logger.info(f"🆔 [TOOL_CALL_ID] {tool_call_id}")

        # Create success message
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name} with task: {task_description}",
            name=name,
            tool_call_id=tool_call_id
        )

        messages = state["messages"]

        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": messages + [tool_message],
                "active_agent": agent_name
            },
        )
    
    # Set metadata for routing
    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}

    return handoff_to_agent