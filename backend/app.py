from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Load models (adjust paths as needed)
MODEL1_PATH = "./models/non_cw_best_model.pth"
MODEL2_PATH = "./models/cw_best_model.pth"

# Define class names
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model1 = load_model(MODEL1_PATH)
model2 = load_model(MODEL2_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded image
    filename = str(uuid.uuid4()) + ".jpg"
    upload_path = os.path.join("static", "uploads", filename)
    file.save(upload_path)

    # Preprocess image
    img = Image.open(upload_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Get predictions from both models
    with torch.no_grad():
        output1 = model1(img_tensor)
        output2 = model2(img_tensor)
        probs1 = torch.nn.functional.softmax(output1, dim=1)[0] * 100
        probs2 = torch.nn.functional.softmax(output2, dim=1)[0] * 100

    # Format results
    results = {
        "image_url": f"/static/uploads/{filename}",
        "model1": {
            "prediction": CLASS_NAMES[torch.argmax(output1).item()],
            "confidence": round(torch.max(probs1).item(), 2),
            "all_probs": {cls: round(prob.item(), 2) for cls, prob in zip(CLASS_NAMES, probs1)}
        },
        "model2": {
            "prediction": CLASS_NAMES[torch.argmax(output2).item()],
            "confidence": round(torch.max(probs2).item(), 2),
            "all_probs": {cls: round(prob.item(), 2) for cls, prob in zip(CLASS_NAMES, probs2)}
        }
    }

    return jsonify(results)

if __name__ == "__main__":
    os.makedirs(os.path.join("static", "uploads"), exist_ok=True)
    app.run(debug=True)