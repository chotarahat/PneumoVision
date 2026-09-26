from pathlib import Path
import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model import build_model
from gradcam import GradCAM


MODEL_PATH = "models/pneumovision_resnet18.pth"


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    classes = checkpoint["classes"]

    model = build_model(
        num_classes=len(classes),
        pretrained=False,
        dropout=checkpoint["config"]["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, classes, device


def preprocess(image):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return tfm(image).unsqueeze(0)


st.set_page_config(page_title="PneumoVision", page_icon="🫁")

st.title("PneumoVision")
st.caption("Explainable AI research prototype for pneumonia detection from chest X-ray images.")

st.warning(
    "Research/education prototype only. It is not a medical device and must not be used "
    "for diagnosis or treatment decisions."
)

if not Path(MODEL_PATH).exists():
    st.error("Trained model not found. Train the baseline model first.")
    st.stop()

model, classes, device = load_model()

uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("L")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    x = preprocess(image).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))

    st.subheader("Prediction")
    st.write(f"**{classes[pred_idx]}**")
    st.write(f"Model confidence: **{probs[pred_idx]:.2%}**")

    st.subheader("Class probabilities")
    for name, p in zip(classes, probs):
        st.write(f"{name}: {p:.2%}")

    st.subheader("Grad-CAM explanation")
    explainer = GradCAM(model, model.layer4[-1].conv2)
    cam, _, _ = explainer(x, pred_idx)

    original = image.resize((224, 224))
    overlay = np.array(original.convert("RGB")).copy()

    # Create a simple red/yellow heatmap using matplotlib-compatible colormap.
    import matplotlib.cm as cm
    heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    blended = (0.55 * overlay + 0.45 * heat).astype(np.uint8)

    st.image(blended, caption="Grad-CAM visualization", use_container_width=True)
