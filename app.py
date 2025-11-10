"""Gradio app for renewable energy classification with verbose diagnostics."""
from pathlib import Path
import platform
import sys

import gradio as gr
import torch
from fastai import __version__ as fastai_version
from fastai.learner import load_learner

MODEL_PATH = Path(__file__).parent / "model.pkl"


def log_startup():
    """Print helpful info when running locally."""
    print("=== Startup diagnostics ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"fastai version: {fastai_version}")
    print(f"torch version: {torch.__version__}")
    resolved = MODEL_PATH.resolve()
    print(f"Expecting model at: {resolved}")
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"model.pkl found ({size_mb:.2f} MiB)")
    else:
        print("model.pkl is missing.")
    print("===========================")


def load_model(path: Path):
    """Load the fastai learner and surface extra context on failure."""
    if not path.exists():
        raise RuntimeError(
            "model.pkl is missing. Export with learn.export() and copy it beside app.py."
        )
    try:
        learner = load_learner(path)
        print("Successfully loaded model.pkl")
        return learner
    except Exception as err:
        print("Failed to load model.pkl:", repr(err))
        raise RuntimeError(
            "model.pkl could not be loaded. Re-export with learn.export() and download it again."
        ) from err


log_startup()
learn = load_model(MODEL_PATH)
LABELS = list(map(str, learn.dls.vocab))


def predict(image):
    """Return class probabilities for the uploaded image."""
    try:
        _, _, probs = learn.predict(image)
        return {label: float(probs[i]) for i, label in enumerate(LABELS)}
    except Exception as err:
        print("Prediction failed:", repr(err))
        raise


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a renewable energy photo"),
    outputs=gr.Label(label="Predicted energy source", num_top_classes=3),
    title="Renewable Energy Classifier",
    description="Drop in your trained model.pkl and launch this Space.",
)

if __name__ == "__main__":
    demo.launch(show_error=True, queue=False)
