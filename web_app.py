import time
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# ---------------- CONFIG ----------------
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_deeplab.pth"

CLASS_NAMES = [
    "Sky", "Sand", "Dry Grass", "Trees", "Bushes",
    "Rocks", "Logs", "Ground Clutter", "Lush Bushes", "Background"
]

COLOR_PALETTE = np.array([
    [61, 90, 128], [152, 193, 217], [224, 251, 252], [238, 108, 77],
    [41, 50, 65], [100, 140, 180], [200, 80, 60], [120, 200, 120],
    [180, 120, 180], [250, 250, 250]
], dtype=np.uint8)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, 1)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True),
        strict=False
    )
    model.to(DEVICE).eval()
    return model

model = load_model()

# ---------------- UI ----------------
st.set_page_config("DesertMind Analytics", layout="wide")
st.title("🌵 DesertMind Analytics — Off-Road Segmentation")

uploaded = st.file_uploader("Upload an off-road image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # ---------- Inference ----------
    start = time.time()
    with torch.no_grad():
        out = model(input_tensor)["out"]
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    latency = (time.time() - start) * 1000

    probs = torch.softmax(out, dim=1)
    pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
    probs_np = probs.squeeze(0).cpu().numpy()

    # ---------- Visualization ----------
    mask = COLOR_PALETTE[pred]
    mask_img = Image.fromarray(mask).resize(image.size, Image.NEAREST)
    blended = Image.blend(image, mask_img, alpha=0.6)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Input Image", use_column_width=True)
    col2.image(blended, caption="Segmentation Output", use_column_width=True)

    # ---------- ANALYTICS ----------
    st.markdown("## 📊 Scene Analytics")

    # --- 1. Class Distribution (Donut) ---
    unique, counts = np.unique(pred, return_counts=True)
    fig1, ax1 = plt.subplots()
    ax1.pie(
        counts,
        labels=[CLASS_NAMES[i] for i in unique],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    ax1.set_title("Terrain Composition")
    st.pyplot(fig1)

    # --- 2. Mean Confidence per Class ---
    confidences = []
    for i in range(NUM_CLASSES):
        mask_i = pred == i
        if mask_i.any():
            confidences.append(probs_np[i][mask_i].mean())
        else:
            confidences.append(0.0)

    fig2, ax2 = plt.subplots()
    ax2.bar(CLASS_NAMES, confidences)
    ax2.set_ylim(0, 1)
    ax2.set_title("Mean Confidence per Class")
    ax2.tick_params(axis="x", rotation=45)
    st.pyplot(fig2)

    # --- 3. Pixel Confidence Heatmap (NEW) ---
    confidence_map = probs_np.max(axis=0)
    fig3, ax3 = plt.subplots()
    im = ax3.imshow(confidence_map, cmap="inferno")
    ax3.set_title("Pixel-wise Confidence Heatmap")
    plt.colorbar(im, ax=ax3)
    st.pyplot(fig3)

    # --- 4. Performance ---
    st.markdown("## ⚡ Performance")
    st.metric("Inference Latency (ms)", f"{latency:.2f}")
