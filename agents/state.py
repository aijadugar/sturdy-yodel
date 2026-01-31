from typing import TypedDict, Optional, List

class AssistantState(TypedDict):
    risk_level: str
    user_location: str
    user_query: Optional[str]

    hospital_results: Optional[List[str]]
    lab_results: Optional[List[str]]
    homecare_results: Optional[str]

    web_results: Optional[str]
    final: str