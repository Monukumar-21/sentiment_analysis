import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Page config ----------
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")
st.title("🎬 Movie Sentiment Analyzer")
st.markdown("Paste a movie review and choose a model to analyze.")

# ---------- Load all models (cached) ----------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_model(filename):
    file_path = MODEL_DIR / filename
    if file_path.exists():
        return joblib.load(file_path)
    return None

available_models = {
    "Logistic Regression": "logistic_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "Random Forest": "random_forest_model.joblib"
}
# Load all models that exist
models = {}
for name, file in available_models.items():
    model = load_model(file)
    if model is not None:
        models[name] = model

if not models:
    st.error("No model files found! Please place .joblib files in the app directory.")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
show_comparison = st.sidebar.checkbox("Compare all models side-by-side")

# ---------- Main input area ----------
user_input = st.text_area(
    "Your review:",
    height=150,
    placeholder="e.g. This film was absolutely brilliant! The acting was top-notch..."
)

# ---------- Prediction logic ----------
def predict_one(pipeline, text):
    """Return (sentiment, prob_neg, prob_pos) for a single pipeline."""
    if pipeline is None:
        return "Unknown", 0.0, 0.0
    try:
        proba = pipeline.predict_proba([text])[0]
        pred = pipeline.predict([text])[0]
        sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
        return sentiment, proba[0], proba[1]
    except Exception as e:
        return f"Error: {e}", 0.0, 0.0

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        st.markdown("---")

        # ---------- Single model prediction ----------
        if not show_comparison:
            sentiment, prob_neg, prob_pos = predict_one(models[model_choice], user_input)
            st.subheader(f"Prediction ({model_choice}): {sentiment}")

            col1, col2 = st.columns(2)
            col1.metric("Negative", f"{prob_neg*100:.1f}%")
            col2.metric("Positive", f"{prob_pos*100:.1f}%")
            st.progress(float(prob_pos if sentiment.startswith("Positive") else prob_neg),
                        text="Confidence")

            # ---------- Show most influential words (if logistic regression) ----------
            if model_choice == "Logistic Regression":
                with st.expander("Most influential words for this prediction"):
                    pipeline = models[model_choice]
                    tfidf = pipeline.named_steps['tfidf']
                    clf = pipeline.named_steps['clf']
                    feature_names = tfidf.get_feature_names_out()
                    coeffs = clf.coef_[0]
                    input_vec = tfidf.transform([user_input])
                    non_zero_indices = input_vec[0].indices

                    contributions = []
                    for idx in non_zero_indices:
                        word = feature_names[idx]
                        coef = coeffs[idx]
                        tfidf_val = input_vec[0, idx]
                        impact = coef * tfidf_val
                        contributions.append((word, impact))
                    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

                    if contributions:
                        for word, impact in contributions[:15]:
                            direction = "⬆️ (positive)" if impact > 0 else "⬇️ (negative)"
                            st.write(f"**{word}** {direction}: {impact:.4f}")
                    else:
                        st.write("No known vocabulary words found in your review.")

            # ---------- Feature importance for tree models ----------
            elif model_choice in ["Decision Tree", "Random Forest"]:
                with st.expander("Global feature importance (for this model)"):
                    pipeline = models[model_choice]
                    tfidf = pipeline.named_steps['tfidf']
                    clf = pipeline.named_steps['clf']
                    feature_names = tfidf.get_feature_names_out()
                    importances = clf.feature_importances_
                    indices = np.argsort(importances)[::-1][:20]
                    for i in indices:
                        st.write(f"**{feature_names[i]}**: {importances[i]:.4f}")

        # ---------- Side-by-side comparison ----------
        else:
            st.subheader("📊 Model Comparison")
            results = {}
            for name, pipeline in models.items():
                sentiment, prob_neg, prob_pos = predict_one(pipeline, user_input)
                results[name] = {"Sentiment": sentiment, "Neg %": prob_neg*100, "Pos %": prob_pos*100}

            # Display as a table
            df_results = pd.DataFrame(results).T
            st.dataframe(df_results.style.format({
                "Neg %": "{:.1f}%",
                "Pos %": "{:.1f}%"
            }).background_gradient(cmap="RdYlGn", subset=["Pos %"]))

            # Also show a bar chart of positive probabilities
            st.bar_chart(df_results["Pos %"])

# ---------- Footer ----------
st.markdown("---")
st.caption("Models: Logistic Regression, Decision Tree, Random Forest (if available) · Trained on movie reviews")