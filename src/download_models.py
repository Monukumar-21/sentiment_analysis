import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "monusharma21/portfolio"
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# The 5 models expected by the app
MODELS_TO_DOWNLOAD = [
    "logistic_model.joblib",
    "decision_tree_model.joblib",
    "random_forest_model.joblib",
    "rnn_model.pt",
    "lstm_model.pt"
]

def ensure_models_downloaded():
    """Ensure all models are downloaded from Hugging Face Hub."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Checking for models in {MODEL_DIR}...")
    for model_name in MODELS_TO_DOWNLOAD:
        local_path = MODEL_DIR / model_name
        if not local_path.exists():
            print(f"Downloading {model_name} from {REPO_ID}...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=model_name,
                    local_dir=str(MODEL_DIR),
                    local_dir_use_symlinks=False
                )
                print(f"Successfully downloaded {model_name}.")
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
                
if __name__ == "__main__":
    ensure_models_downloaded()
