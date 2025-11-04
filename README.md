# Renewable Energy Classifier

A tiny end-to-end demo for experimenters who want to classify renewable energy
images with fastai, deploy the model to HuggingFace Spaces, and present a public
demo with GitHub Pages.

## Project layout

```
renewable_energy/
├── app.py            # Gradio interface source that you copy into hf-space/
├── index.html        # Static UI that calls your HuggingFace Space API
├── requirements.txt  # Dependencies shared with the Space runtime
├── hf-space/         # Cloned HuggingFace Space repo (contains model.pkl)
└── README.md         # This file
```

## How to use

1. **Train & export on Kaggle**  
   Fine-tune a fastai vision model and export it as `model.pkl`. You can follow
   fastai's `learn.export()` workflow, then download the file into
   `hf-space/`.

2. **Deploy to HuggingFace Spaces**  
   - Create a new Space (Gradio runtime).  
   - Copy `app.py`, `requirements.txt`, and your new `model.pkl` into `hf-space/`.  
   - Commit and push from that folder (`git add`, `git commit`, `git push`).  
   - The Space rebuilds automatically once the push completes.

3. **Update GitHub Pages**  
   - Host this repo on GitHub and enable Pages (e.g. `main` branch, `/`).  
   - Edit `index.html` and set `apiUrl` to the URL of your live HuggingFace
     Space (see comment in the file).  
   - Commit and push; Pages builds a static site that forwards requests to your
     Space for predictions.

## Live demos (replace later)

- **HuggingFace Space:** _coming soon_  
- **GitHub Pages:** _coming soon_

## Next steps

- Add sample images to help visitors test the demo quickly.  
- Customize styling or messaging in `index.html` to match your project.
