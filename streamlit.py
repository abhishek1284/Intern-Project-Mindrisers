import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

# ====== 1️⃣ Load Trained Model ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Model
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
num_classes = 4  # Update with your actual number of classes
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
model.load_state_dict(torch.load(r"C:\Users\Dell\Desktop\train1\my_trained_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define class labels (Update these with your actual class names)
class_names = ["cat", "cats", "dogs", "horse"]
confidence_threshold = 30  # Minimum confidence to consider a prediction valid

# ====== 2️⃣ Prediction Function ======
def predict_image(image):
    """Predicts the class of an input image."""
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item() * 100

    if confidence < confidence_threshold:
        return "No class found", confidence
    return class_names[predicted_class], confidence

# ====== 3️⃣ Streamlit UI ======
st.set_page_config(page_title="Image Classifier", page_icon="🔍", layout="wide")

# Sidebar
st.sidebar.title("🔹 Instructions")
st.sidebar.info("1️⃣ Upload an image\n2️⃣ Model will classify it\n3️⃣ Confidence level will be shown")

# Main App
st.title("🔍 Image Classification with MobileNetV3")
st.write("📷 Upload an image to predict its class!")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])  # Layout

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔄 Predicting... Please wait!"):
            predicted_label, confidence = predict_image(image)

        # Confidence Color Coding
        if confidence >= 75:
            color = "✅ **High Confidence** (Green)"
            bar_color = "green"
        elif 50 <= confidence < 75:
            color = "⚠️ **Moderate Confidence** (Yellow)"
            bar_color = "yellow"
        else:
            color = "🚨 **Low Confidence** (Red)"
            bar_color = "red"

        if predicted_label == "No class found":
            st.error("🚨 No class found! The model is not confident enough.")
        else:
            st.success(f"✅ **Predicted Class: {predicted_label}**")
            st.write(f"🔹 Confidence: **{confidence:.2f}%** - {color}")

            # Confidence Bar
            st.progress(int(confidence))

st.write("📌 **Tip:** Try different images for better results.")
