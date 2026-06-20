import streamlit as st
import joblib
import numpy as np

import os

@st.cache_resource
def load_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'sentiment_model.joblib')
    return joblib.load(model_path)

pipeline = load_pipeline()

# ---- UI Layout ----
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Paste a movie review and the model will predict if it's **Positive** or **Negative**.")

user_input = st.text_area(
    "Your review:",
    height=150,
    placeholder="e.g. This film was absolutely brilliant! The acting was top-notch..."
)

# ---- Prediction ----
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        # Predict probabilities and class
        proba = pipeline.predict_proba([user_input])[0]   # [prob_neg, prob_pos]
        pred = pipeline.predict([user_input])[0]

        sentiment = "Positive 😊" if pred == 1 else "Negative 😞"

        st.markdown("---")
        st.subheader(f"Prediction: {sentiment}")

        # Confidence display
        col1, col2 = st.columns(2)
        col1.metric("Negative", f"{proba[0]*100:.1f}%")
        col2.metric("Positive", f"{proba[1]*100:.1f}%")

        # Show a confidence bar
        confidence = proba[1] if pred == 1 else proba[0]
        st.progress(float(confidence), text="Confidence")

        # (Optional) Show top contributing words
        with st.expander("See most influential words for this prediction"):
            # Get feature names and model coefficients
            feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
            coefficients = pipeline.named_steps['clf'].coef_[0]
            # Find indices of non-zero TF-IDF values for the input
            input_vec = pipeline.named_steps['tfidf'].transform([user_input])
            non_zero_indices = input_vec[0].indices
            # Sort by absolute contribution (coefficient * tfidf value)
            contributions = []
            for idx in non_zero_indices:
                word = feature_names[idx]
                coef = coefficients[idx]
                tfidf_val = input_vec[0, idx]
                impact = coef * tfidf_val
                contributions.append((word, impact))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            if contributions:
                st.write("Word | Impact (positive → positive sentiment)")
                for word, impact in contributions[:15]:
                    direction = "⬆️" if impact > 0 else "⬇️"
                    st.write(f"{word} {direction} {impact:.4f}")
            else:
                st.write("No known words found in vocabulary.")

st.markdown("---")
st.caption("Model: Logistic Regression with TF‑IDF features · Trained on movie reviews")