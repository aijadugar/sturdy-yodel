import io
import random
import librosa
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents.assistant_agent import assistant_agent
from utils.preprocessing import audio_to_logmel
from utils.audio_utils import mp4_to_wav_bytes
# from utils.tb_predict import predict_tb
from models.tbCoughMLClass import TBRequest, TBResponse
from models.ml_model_load import model, scaler, le_dict, target_encoder
from utils.preprocess_in import preprocess_input
import torch.nn.functional as F
from models.tbCoughCNN import TBCoughCNN
from huggingface_hub import hf_hub_download

app=FastAPI(title="Lockdown: Tuberculosis Screening via Cough Sound Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

model_path = hf_hub_download(
repo_id="aijadugar/tb-cough-coswara",
filename="tb_cough_model.pt"
)
model = TBCoughCNN()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

class AssistentRequest(BaseModel):
    risk_level: str
    user_location: str
    user_query: str | None=None


def predict_tb(log_mel):
    """
    Input: log_mel -> (128, time)
    Output: TB probability (0-1) and label
    """

    

    x = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,128,T)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].numpy()
    
    tb_prob = float(probs[1])  # index 1 = TB
    if tb_prob > 0.7:
        label = "High Risk"
    elif tb_prob > 0.4:
        label = "Medium Risk"
    else:
        label = "Low Risk"
    
    return tb_prob, label

@app.get('/')
def lockdown():
    return {
        "status": "CoughLock backend is building..."
    }

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            message = await websocket.receive()

            if "bytes" not in message:
                continue

            raw_bytes = message["bytes"]

            try:
                # 🔍 Detect MP4 (video container)
                if b"ftyp" in raw_bytes[:32]:
                    audio_bytes = mp4_to_wav_bytes(raw_bytes)
                else:
                    audio_bytes = raw_bytes  # wav / webm / mp3

                # Audio → log-mel
                log_mel = audio_to_logmel(audio_bytes)

                # Model inference
                tb_prob, label = predict_tb(log_mel)

                await websocket.send_json({
                    "confidence": round(tb_prob, 3),
                    "label": label,
                    "spectrogram_shape": list(log_mel.shape)
                })

            except Exception as e:
                print("Processing error:", e)
                await websocket.send_json({
                    "error": "Unsupported audio/video format"
                })

    except Exception as e:
        print("WebSocket error:", e)

    finally:
        await websocket.close()

@app.post("/api/predict", response_model=TBResponse)
async def predict_tb(data: TBRequest):
    data_dict=data.dict()
    x=preprocess_input(data_dict, scaler, le_dict)
    risk_score=model.predict_proba(x)[0][1]
    if risk_score > 0.7:
        risk_level="High Level"
    elif risk_score>0.4:
        risk_level="Medium Level"
    else:
        risk_level="Low Level"
    
    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level
    }

@app.post("/assistant")
async def assistant(req: AssistentRequest):
    state={
        "risk_level": req.risk_level.upper(),
        "user_location": req.user_location,
        "user_query": req.user_query
    }

    results=assistant_agent.invoke(state)
    return {
        "response": results["final"]
    }



