import torch.nn.functional as F
from models.tbCoughCNN import TBCoughCNN
from huggingface_hub import hf_hub_download

def predict_tb(log_mel):
    """
    Input: log_mel -> (128, time)
    Output: TB probability (0-1) and label
    """

    model_path = hf_hub_download(
    repo_id="aijadugar/tb-cough-coswara",
    filename="tb_cough_model.pt"
    )
    model = TBCoughCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

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