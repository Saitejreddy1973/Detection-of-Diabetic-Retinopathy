import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)

# ✅ Set upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Define the CNN Model (MUST match how the model was trained)
class CNN_Retino(nn.Module):
    def __init__(self):
        super(CNN_Retino, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 100)
        self.fc2 = nn.Linear(100, 5)  # ✅ Change if model was trained for 5 classes

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Load Model & Weights
model = CNN_Retino()
model.load_state_dict(torch.load("models\weights.pt", map_location=torch.device("cpu")))
model.eval()

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]  # ✅ Change to 5-class labels if needed

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            image = preprocess_image(filepath)

            with torch.no_grad():
                output = model(image)
                prediction = output.argmax().item()

            result = LABELS[prediction]

            return render_template("index.html", result=result, filepath=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
