import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  #To allow cross-origin requests

#Defining the class labels for TrashNet
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

#Loading the pre-trained model
def load_model():
    #Initializing ResNet18 model with the number of classes in TrashNet
    # hanging from ResNet50 to ResNet18 based on your model architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(CLASSES))
    
    #Loading the trained weights
    checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
    
    #If the checkpoint contains the full model rather than just state_dict
    if not isinstance(checkpoint, dict) or 'state_dict' in checkpoint:
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint  #The checkpoint is the full model
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

#Initializing model
try:
    model = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Defining the transformation for input images (same as used during training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/classify', methods=['POST'])
def classify_image():
    if model is None:
        return jsonify({'error': 'Model failed to load. Please check server logs.'})
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        #Opening and preprocess the image
        img = Image.open(file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  #Adding batch dimension
        
        #Performing inference
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
            #Getting probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            #Creating result dictionary
            result = {
                'predicted_class': CLASSES[predicted.item()],
                'confidence': float(probabilities[predicted].item()),
                'all_probabilities': {
                    class_name: float(probabilities[i].item()) 
                    for i, class_name in enumerate(CLASSES)
                }
            }
            
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    if model is None:
        return jsonify({'status': 'unhealthy', 'reason': 'Model failed to load'})
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
