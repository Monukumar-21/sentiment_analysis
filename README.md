# 🎬 Movie Sentiment Analyzer

A machine learning web application built with Streamlit that analyzes the sentiment of movie reviews. It predicts whether a review is Positive 😊 or Negative 😞 using a variety of classic machine learning and deep learning models.

## ✨ Features

- **Multiple Models**: Choose from Logistic Regression, Decision Tree, Random Forest, RNN, and LSTM to analyze reviews.
- **Model Comparison**: Compare the performance and predictions of all models side-by-side.
- **Interpretability**:
  - *Linear Models*: Highlights the exact words that influenced the prediction (positive or negative impact).
  - *Tree Models*: Displays the global top 20 most important features.
- **Interactive UI**: Simple, user-friendly Streamlit interface with real-time confidence scores.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn (Logistic Regression, Decision Trees, Random Forests)
- **Deep Learning**: PyTorch (RNN, LSTM)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Wordcloud

## 📁 Project Structure

```
movie_sentiment_analysis/
│
├── app/
│   └── main.py              # Streamlit application
├── models/                  # Pre-trained models (.joblib, .pt)
├── notebooks/               # Jupyter notebooks for EDA and Model Training
│   ├── 01_data_inspection_and_eda.ipynb
│   └── models_pipeline.ipynb
├── data/                    # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── src/                     # Helper scripts
│   └── download_models.py
├── pyproject.toml           # Project configuration
└── requirements.txt         # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python >= 3.12 installed.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/movie_sentiment_analysis.git
   cd movie_sentiment_analysis
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

To start the Streamlit web application, run the following command from the root directory:

```bash
streamlit run app/main.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

## 🧠 Models Available

This application dynamically loads machine learning models from Hugging Face Hub (Repository: `monusharma21/portfolio`). 

When you start the application, it will automatically download the required model files if they are not already present in the `models/` directory:
- `logistic_model.joblib`
- `decision_tree_model.joblib`
- `random_forest_model.joblib`
- `rnn_model.pt`
- `lstm_model.pt`

This setup keeps the repository lightweight and makes deployment straightforward!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.
