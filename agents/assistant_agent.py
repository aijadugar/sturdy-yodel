from langgraph.graph import StateGraph, END
from agents.state import AssistantState
from agents.nodes import risk_router_node, web_qa_node, reasoning_node

workflow = StateGraph(AssistantState)

workflow.add_node("risk_router", risk_router_node)
workflow.add_node("web", web_qa_node)
workflow.add_node("reason", reasoning_node)

workflow.set_entry_point("risk_router")

workflow.add_edge("risk_router", "web")
workflow.add_edge("web", "reason")
workflow.add_edge("reason", END)

assistant_agent = workflow.compile()