from pydantic import BaseModel

class TBRequest(BaseModel):
    Age: int
    Gender: str
    Cough: int
    Fever: str
    WeightLoss: int
    NightSweats: str
    ChestPain: str
    Hemoptysis: str
    Breathlessness: str
    ContactHistory: str
    TravelHistory: str
    HIVStatus: str
    PreviousTB: str
    ChestXRay: str
    SputumTest: str

class TBResponse(BaseModel):
    risk_score: float
    risk_level: str
