# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Image Disease Detector", layout="centered")

# ---------- USER CONFIG ----------
# Replace these labels with your model's labels, in the same order as your model's output
LABELS = ["Normal", "Disease A", "Disease B"]

# Default path for model weights (user can upload via sidebar)
DEFAULT_MODEL_PATH = "model.pth"
NUM_CLASSES = len(LABELS)
# -------------------------------

@st.cache_resource
def load_model(path: str, num_classes: int = NUM_CLASSES, device='cpu'):
    """
    Example: loads a ResNet18 and replaces fc for num_classes.
    Replace with your model architecture if different.
    """
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    try:
        state = torch.load(path, map_location=device)
        # handle state dict or whole model
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)
    except Exception as e:
        st.warning(f"Could not load model from {path}: {e}")
        st.info("Using randomly initialized model architecture for demo (predictions will be meaningless).")
    model.to(device)
    model.eval()
    return model

# Preprocessing transform consistent with common pretrained nets
def get_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def predict_image_pytorch(model, pil_img, device='cpu', topk=3):
    transform = get_transform()
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = np.argsort(probs)[::-1][:topk]
    return [(LABELS[i], float(probs[i]), int(i)) for i in top_idx], probs, logits

# Simple Grad-CAM implementation for ResNet-like models
def grad_cam(model, pil_img, target_class=None, device='cpu'):
    """
    Returns heatmap (H x W) overlayed on original image and bare heatmap.
    Assumes model has attribute .layer4 (ResNet). Adapt for other archs.
    """
    model.eval()
    # Hook to capture gradients and activations
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # Try to locate a last conv layer (ResNet: layer4[-1].conv2)
    target_layer = None
    try:
        # For ResNet-like
        target_layer = model.layer4[-1].conv2
    except Exception:
        # fallback: try to find last nn.Conv2d in model
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                target_layer = m
                break
    if target_layer is None:
        raise RuntimeError("Couldn't find a conv layer for Grad-CAM. Adapt the code to your model.")

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    transform = get_transform()
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)
    if target_class is None:
        target_class = int(torch.argmax(probs, dim=1).item())

    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    act = activations['value'][0].cpu().numpy()  # C x H x W
    grad = gradients['value'][0].cpu().numpy()   # C x H x W

    # Global average pool of gradients -> weights
    weights = np.mean(grad, axis=(1,2))  # C
    cam = np.zeros(act.shape[1:], dtype=np.float32)  # H x W
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]

    # ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    cam = cv2.resize(cam, (pil_img.width, pil_img.height))

    # Create color heatmap (0-255)
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR

    # Convert original PIL to OpenCV BGR
    orig = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)

    # Convert back to RGB PIL images
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    handle_f.remove()
    handle_b.remove()
    return Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb)

# ---------- Streamlit UI ----------
st.title("🩺 AI Image Disease Detector")
st.write("Upload an image (X-ray, skin photo, leaf, etc.) and the model will return predictions and a Grad-CAM heatmap.")

with st.sidebar:
    st.header("Model")
    st.write("Upload your PyTorch `.pth` model (optional). If not provided, the app will try to load `model.pth` in the working directory.")
    uploaded_model = st.file_uploader("Upload model.pth", type=["pth", "pt"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Device: **{device}**")
    if uploaded_model is not None:
        # Save to temp file and load
        model_bytes = uploaded_model.read()
        tmp_path = "uploaded_model.pth"
        with open(tmp_path, "wb") as f:
            f.write(model_bytes)
        model_path = tmp_path
    else:
        model_path = DEFAULT_MODEL_PATH
    if st.button("Load model"):
        st.session_state['model'] = load_model(model_path, num_classes=NUM_CLASSES, device=device)
        st.success("Model loaded (or attempted).")

# Ensure model loaded (try cache or load on demand)
if 'model' not in st.session_state:
    st.session_state['model'] = load_model(model_path, num_classes=NUM_CLASSES, device=device)

model = st.session_state['model']

st.subheader("Input")
uploaded_image = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
use_camera = st.checkbox("Use webcam (if supported by browser)", value=False)
if use_camera and uploaded_image is None:
    try:
        # streamlit camera input (works in many environments)
        cam_img = st.camera_input("Take a picture")
        if cam_img:
            uploaded_image = cam_img
    except Exception:
        pass

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Input image", use_column_width=True)

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Predict"):
            with st.spinner("Running model..."):
                topk, probs, logits = predict_image_pytorch(model, image, device=device, topk=5)
                st.success("Prediction done.")
                st.write("**Top predictions**")
                for label, p, idx in topk:
                    st.write(f"- **{label}** — {p*100:.2f}%")
                # Show probability bar chart for all classes
                probs_series = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
                st.bar_chart(probs_series)

    with col2:
        if st.button("Show Grad-CAM"):
            try:
                with st.spinner("Computing Grad-CAM..."):
                    # Choose predicted class automatically
                    predicted_idx = int(torch.argmax(torch.tensor(probs)).item())
                    overlay_img, heatmap_img = grad_cam(model, image, target_class=predicted_idx, device=device)
                    st.image(overlay_img, caption="Grad-CAM overlay", use_column_width=True)
                    st.image(heatmap_img, caption="Heatmap (jet)", use_column_width=True)
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}\nThis implementation assumes a ResNet-like architecture. Adapt target layer selection for other models.")

else:
    st.info("Upload an input image to start.")

st.markdown("---")
st.markdown("**Notes & next steps**")
st.markdown("""
- Replace `LABELS` with your model's class names in the same order the model outputs them.  
- Drop your trained PyTorch `.pth` file into the working directory as `model.pth` or upload it via the sidebar.  
- If your model architecture is not ResNet-like, change the `load_model` and the `grad_cam` target-layer selection accordingly.
""")