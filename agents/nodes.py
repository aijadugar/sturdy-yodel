from langchain_google_genai import ChatGoogleGenerativeAI
from agents.state import AssistantState
from agents.tools import (
    hospital_search_tool,
    lab_search_tool,
    homecare_tool,
    web_search_tool
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

#Risk-based routing node
# -------------------------------
def risk_router_node(state: AssistantState):
    risk = state["risk_level"]

    if risk == "HIGH":
        hospitals = hospital_search_tool(state["user_location"])
        return {"hospital_results": hospitals}

    if risk == "MEDIUM":
        labs = lab_search_tool(state["user_location"])
        return {"lab_results": labs}

    if risk == "LOW":
        return {"homecare_results": homecare_tool()}

    return {}


#User question handling
# -------------------------------
def web_qa_node(state: AssistantState):
    if not state.get("user_query"):
        return {}

    web_info = web_search_tool(state["user_query"])
    return {"web_results": web_info}


#Final reasoning node
# -------------------------------
def reasoning_node(state: AssistantState):
    prompt = f"""
You are a medical AI assistant for tuberculosis screening.

Risk Level: {state['risk_level']}
User Location: {state['user_location']}

Hospitals:
{state.get('hospital_results')}

Labs:
{state.get('lab_results')}

Home Care:
{state.get('homecare_results')}

User Question:
{state.get('user_query')}

Additional Medical Info:
{state.get('web_results')}

Rules:
- HIGH risk → urgent hospital visit
- MEDIUM risk → diagnostic tests first
- LOW risk → home care + monitoring
- Do NOT prescribe exact medicines
- Be supportive and clear
- Add medical disclaimer
"""

    response = llm.invoke(prompt)
    return {"final": response.content}