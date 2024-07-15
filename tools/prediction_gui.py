import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import tkinter as tk
from PIL import Image, ImageDraw, ImageGrab
import numpy as np

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load the pre-trained model
model = Net()
#model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

# Create the main window
root = tk.Tk()
root.title("MNIST Digit Recognizer")

# Create a canvas for drawing
canvas = tk.Canvas(root, width=280, height=280, bg="black")
canvas.pack()

# Function to draw on the canvas
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="white", width=0)

canvas.bind("<B1-Motion>", paint)

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")

# Function to predict the digit
def predict_digit():
    # Get the image from the canvas
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab().crop((x, y, x1, y1))
    
    # Preprocess the image
    image = image.resize((28, 28)).convert('L')
    image = np.array(image)
    image = 255 - image  # Invert the image colors
    image = image / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
    
    result_label.config(text=f"Predicted Digit: {pred.item()}")

# Create buttons and label
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack()

result_label = tk.Label(root, text="Predicted Digit: ")
result_label.pack()

root.mainloop()
