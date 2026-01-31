import joblib
from huggingface_hub import hf_hub_download

HF_REPO = "aijadugar/tb-risk-model"

model = joblib.load(hf_hub_download(HF_REPO, "tb_model.pkl"))
scaler = joblib.load(hf_hub_download(HF_REPO, "tb_scaler.pkl"))
le_dict = joblib.load(hf_hub_download(HF_REPO, "categorical_encoders.pkl"))
target_encoder = joblib.load(hf_hub_download(HF_REPO, "target_encoder.pkl"))