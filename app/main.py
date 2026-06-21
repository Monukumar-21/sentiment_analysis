import streamlit as st
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent         
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def ensure_models_downloaded():
    """Ensure all models are downloaded from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        st.error("Please install huggingface_hub: pip install huggingface_hub")
        st.stop()
        
    REPO_ID = "monusharma21/portfolio"
    MODELS_TO_DOWNLOAD = [
        "logistic_model.joblib",
        "decision_tree_model.joblib",
        "random_forest_model.joblib",
        "rnn_model.pt",
        "lstm_model.pt"
    ]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    for model_name in MODELS_TO_DOWNLOAD:
        local_path = MODEL_DIR / model_name
        if not local_path.exists():
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=model_name,
                    local_dir=str(MODEL_DIR),
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                st.error(f"Error downloading {model_name}: {e}")

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True,
                          dropout=dropout if dropout > 0 else 0)
        self.fc = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)                
        _, hidden = self.rnn(embedded)             
        hidden = hidden.squeeze(0)                  
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.fc(hidden))
        return self.sigmoid(self.out(hidden)).squeeze(1)


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            dropout=dropout if dropout > 0 else 0)
        self.fc = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)        
        hidden = hidden.squeeze(0)
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.fc(hidden))
        return self.sigmoid(self.out(hidden)).squeeze(1)


@st.cache_resource
def load_sklearn_pipeline(filename):
    """Load a scikit-learn pipeline from a joblib file."""
    path = MODEL_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_resource
def load_pytorch_model(filename):
    """Load a PyTorch model checkpoint and return (model, word2idx, max_len)."""
    path = MODEL_DIR / filename
    if not path.exists():
        return None, None, None

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model_type = checkpoint['model_type']
    vocab_size = checkpoint['vocab_size']
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    output_dim = checkpoint['output_dim']

    if model_type == 'RNN':
        model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
    else:
        model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['word2idx'], checkpoint['max_len']

def predict_sklearn(pipeline, text):
    """Return (sentiment_str, prob_neg, prob_pos) using a pipeline."""
    proba = pipeline.predict_proba([text])[0]
    pred = pipeline.predict([text])[0]
    sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
    return sentiment, proba[0], proba[1]

def predict_pytorch(text, model, word2idx, max_len):
    """Return (sentiment_str, prob_pos) for a PyTorch model."""
    # tokenize and convert
    tokens = re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
    ids = [word2idx.get(t, 1) for t in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids += [0] * (max_len - len(ids))
    input_tensor = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        prob_pos = model(input_tensor).item()
    pred = 1 if prob_pos >= 0.5 else 0
    sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
    prob_neg = 1 - prob_pos
    return sentiment, prob_neg, prob_pos

MODEL_REGISTRY = {
    "Logistic Regression": {
        "file": "logistic_model.joblib",
        "loader": "sklearn",
        "interpretable": "linear"
    },
    "Decision Tree": {
        "file": "decision_tree_model.joblib",
        "loader": "sklearn",
        "interpretable": "tree"
    },
    "Random Forest": {
        "file": "random_forest_model.joblib",
        "loader": "sklearn",
        "interpretable": "tree"
    },
    "RNN (PyTorch)": {
        "file": "rnn_model.pt",
        "loader": "pytorch",
        "interpretable": "none"
    },
    "LSTM (PyTorch)": {
        "file": "lstm_model.pt",
        "loader": "pytorch",
        "interpretable": "none"
    }
}

# Download models from HF if not present locally
with st.spinner("Downloading models from Hugging Face if needed..."):
    ensure_models_downloaded()

# Load all available models
models = {}
for name, info in MODEL_REGISTRY.items():
    if info["loader"] == "sklearn":
        pipe = load_sklearn_pipeline(info["file"])
        if pipe is not None:
            models[name] = {"pipeline": pipe, "type": "sklearn", "interpretable": info["interpretable"]}
    else:  # pytorch
        model, w2i, maxlen = load_pytorch_model(info["file"])
        if model is not None:
            models[name] = {
                "model": model,
                "word2idx": w2i,
                "max_len": maxlen,
                "type": "pytorch",
                "interpretable": info["interpretable"]
            }

if not models:
    st.error("No model files found! Please place .joblib / .pt files in the 'model/' folder.")
    st.stop()

# ======================== Streamlit UI ========================
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")
st.title("🎬 Movie Sentiment Analyzer")
st.markdown("Paste a movie review and choose a model to analyse.")

# Sidebar
st.sidebar.header("⚙️ Model Selection")
model_names = list(models.keys())
model_choice = st.sidebar.selectbox("Choose a model", model_names)
show_comparison = st.sidebar.checkbox("Compare all models side-by-side")

# Input area
user_input = st.text_area(
    "Your review:",
    height=150,
    placeholder="e.g. This film was absolutely brilliant! The acting was top-notch..."
)

def show_linear_contributions(pipeline, text):
    """For linear models (LR, SVM) show top word contributions."""
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    coef = clf.coef_[0]   # only for binary linear
    input_vec = tfidf.transform([text])
    non_zero = input_vec[0].indices

    contributions = []
    for idx in non_zero:
        word = feature_names[idx]
        impact = coef[idx] * input_vec[0, idx]
        contributions.append((word, impact))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    if contributions:
        st.write("**Word** | **Impact** (⬆️ pushes Positive)")
        for word, impact in contributions[:15]:
            arrow = "⬆️" if impact > 0 else "⬇️"
            st.write(f"{word} {arrow} {impact:.4f}")
    else:
        st.write("No known vocabulary words found.")

def show_tree_importances(pipeline):
    """Global feature importance for tree models."""
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    st.write("**Top 20 global features:**")
    for i in indices:
        st.write(f"{feature_names[i]}: {importances[i]:.4f}")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        st.markdown("---")

        # ---------- Single model prediction ----------
        if not show_comparison:
            selected = models[model_choice]
            if selected["type"] == "sklearn":
                sentiment, prob_neg, prob_pos = predict_sklearn(selected["pipeline"], user_input)
            else:  # pytorch
                sentiment, prob_neg, prob_pos = predict_pytorch(
                    user_input, selected["model"], selected["word2idx"], selected["max_len"])

            st.subheader(f"Prediction ({model_choice}): {sentiment}")

            col1, col2 = st.columns(2)
            col1.metric("Negative", f"{prob_neg*100:.1f}%")
            col2.metric("Positive", f"{prob_pos*100:.1f}%")

            confidence = prob_pos if "Positive" in sentiment else prob_neg
            st.progress(float(confidence), text="Confidence")

            # Interpretability
            interpret = selected["interpretable"]
            with st.expander("Model explanation"):
                if interpret == "linear":
                    show_linear_contributions(selected["pipeline"], user_input)
                elif interpret == "tree":
                    show_tree_importances(selected["pipeline"])
                else:
                    st.write("No built-in explanation for this model.")

        else:
            st.subheader("📊 Model Comparison")
            results = {}
            for name, info in models.items():
                if info["type"] == "sklearn":
                    sentiment, prob_neg, prob_pos = predict_sklearn(info["pipeline"], user_input)
                else:
                    sentiment, prob_neg, prob_pos = predict_pytorch(
                        user_input, info["model"], info["word2idx"], info["max_len"])
                results[name] = {
                    "Sentiment": sentiment,
                    "Neg %": prob_neg * 100,
                    "Pos %": prob_pos * 100
                }

            df_results = pd.DataFrame(results).T
            st.dataframe(df_results.style.format({
                "Neg %": "{:.1f}%",
                "Pos %": "{:.1f}%"
            }).background_gradient(cmap="RdYlGn", subset=["Pos %"]))

            st.bar_chart(df_results["Pos %"])

st.markdown("---")
st.caption("Models: Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost, SVM, RNN, LSTM · Trained on movie reviews")