"""Minimal Gradio app for renewable energy image classification."""
from pathlib import Path
import gradio as gr
from fastai.learner import load_learner

MODEL_PATH = Path(__file__).parent / "model.pkl"  # Keep the export beside app.py.


def load_model(path: Path):
    """Load the fastai learner and surface a friendly error if setup is incomplete."""
    if not path.exists():
        raise RuntimeError("model.pkl is missing. Export with learn.export() and copy it beside app.py.")
    try:
        return load_learner(path)
    except Exception as err:
        raise RuntimeError("model.pkl could not be loaded. Re-export with learn.export() and download it again.") from err

learn = load_model(MODEL_PATH)
LABELS = list(map(str, learn.dls.vocab))


def predict(image):
    """Return class probabilities for the uploaded image."""
    _, _, probs = learn.predict(image)
    return {label: float(probs[i]) for i, label in enumerate(LABELS)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a renewable energy photo"),
    outputs=gr.Label(label="Predicted energy source", num_top_classes=3),
    title="Renewable Energy Classifier",
    description="Drop in your trained model.pkl and launch this Space.",
)

if __name__ == "__main__":
    demo.launch()
