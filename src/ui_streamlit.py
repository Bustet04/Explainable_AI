import streamlit as st
import numpy as np
from pathlib import Path
from joblib import load
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Sleep Staging Explorer", layout='wide')

st.title("Sleep Staging Model Explorer")
script_dir = Path(__file__).parent.absolute()

# Helper: simple model summary print
def model_summary_lines(model):
    lines = []
    for name, module in model.named_children():
        lines.append(f"{name}: {module.__class__.__name__}")
    return lines

# UI: select model file
st.sidebar.header("Model & Data")
model_file = st.sidebar.selectbox("Select model file", options=["pytorch_sleep_model.pt","pytorch_sleep_model_baseline.pt","train_pytorch_improved.py (not a model)"], index=0)
scaler_file = st.sidebar.text_input("Scaler filename", value="feature_scaler.joblib")

# Load features and labels
if st.sidebar.button("Load dataset from features.npy"):
    try:
        features = np.load(script_dir / 'features.npy')
        labels = np.load(script_dir / 'labels.npy', allow_pickle=True)
        st.session_state['features'] = features
        st.session_state['labels'] = labels
        st.success(f"Loaded features {features.shape}")
    except Exception as e:
        st.error(f"Failed to load features: {e}")

if 'features' in st.session_state:
    features = st.session_state['features']
    labels = st.session_state['labels']
else:
    features = None
    labels = None

col1, col2 = st.columns([2,1])
with col1:
    st.header("Example prediction")
    if features is None:
        st.info("Load `features.npy` via the sidebar to try example predictions.")
    else:
        idx = st.number_input('Select epoch index (0-based)', min_value=0, max_value=features.shape[0]-1, value=0)
        X = features[idx]
        st.write('Feature vector (first 20 values):')
        st.write(X[:20].round(4))
        # Load scaler and model
        try:
            scaler = load(script_dir / scaler_file)
            Xs = scaler.transform(X.reshape(1,-1))
            device = torch.device('cpu')
            # define a minimal model architecture compatible with saved weights
            class SimpleMLP(nn.Module):
                def __init__(self, input_dim, n_classes=5):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(256, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 5)
                    )
                def forward(self, x):
                    return self.net(x)
            model = SimpleMLP(X.shape[0]).to(device)
            model_path = script_dir / model_file
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    out = model(torch.from_numpy(Xs).float().to(device))
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                    pred = int(out.argmax(dim=1).cpu().numpy()[0])
                stage_labels = ['N1','N2','N3','R','W']
                st.subheader('Prediction')
                st.write(f"Predicted stage: {stage_labels[pred]}")
                fig = px.bar(x=stage_labels, y=probs, labels={'x':'Stage','y':'Probability'}, title='Class probabilities')
                st.plotly_chart(fig)
            else:
                st.warning(f"Model file {model_file} not found in project folder")
        except Exception as e:
            st.error(f"Error loading model/scaler: {e}")

with col2:
    st.header('Model summary')
    st.write('Shows a simple textual summary of the loaded model (top-level modules).')
    try:
        if model_path.exists():
            lines = model_summary_lines(model)
            for ln in lines:
                st.text(ln)
    except Exception:
        st.text('No model loaded yet.')

st.markdown('---')

st.header('Visualizations')
viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    if (script_dir / 'visualizations' / 'confusion_matrix.png').exists():
        st.image(str(script_dir / 'visualizations' / 'confusion_matrix.png'), caption='Confusion Matrix')
    if (script_dir / 'visualizations' / 'training_history.png').exists():
        st.image(str(script_dir / 'visualizations' / 'training_history.png'), caption='Training History')
with viz_col2:
    if (script_dir / 'visualizations' / 'class_distribution.png').exists():
        st.image(str(script_dir / 'visualizations' / 'class_distribution.png'), caption='Class Distribution')
    if (script_dir / 'visualizations' / 'feature_distributions.png').exists():
        st.image(str(script_dir / 'visualizations' / 'feature_distributions.png'), caption='Feature Distributions')

st.markdown('---')
st.info('To run this UI: in a terminal run `streamlit run ui_streamlit.py` from the project `Code/Projekt` folder after installing requirements.')