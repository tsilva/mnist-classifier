import torch
from torchvision import transforms
import tkinter as tk
from PIL import Image, ImageDraw
import os
from glob import glob
import argparse

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load models from a given path
def load_models(model_path):
    if os.path.isdir(model_path):
        model_files = glob(os.path.join(model_path, "*.pth"))
    else:
        model_files = [model_path]
    models = [torch.jit.load(model_file, map_location=device).to(device).eval() for model_file in model_files]
    return models

# Function to make predictions using an ensemble of models
def ensemble_predict(models, input_tensor):
    outputs = [model(input_tensor) for model in models]
    avg_output = sum(outputs) / len(outputs)
    return avg_output.argmax(dim=1, keepdim=True)

def main(args):
    # Load the models
    models = load_models(args.model_path)

    # Create the main window
    root = tk.Tk()
    root.title("MNIST Digit Recognizer")

    # Create a canvas for drawing
    canvas_width, canvas_height = 280, 280
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="black")
    canvas.pack()

    # Create an image to draw on
    image = Image.new("L", (canvas_width, canvas_height), color="black")
    draw = ImageDraw.Draw(image)

    # Function to draw on the canvas and the image
    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill="white", width=0)
        draw.ellipse([x1, y1, x2, y2], fill="white")

    # Bind the paint function to the mouse motion event
    canvas.bind("<B1-Motion>", paint)

    # Function to clear the canvas and the image
    def clear_canvas():
        canvas.delete("all")
        nonlocal image, draw
        image = Image.new("L", (canvas_width, canvas_height), color="black")
        draw = ImageDraw.Draw(image)

    # Function to predict the digit
    def predict_digit():
        # Preprocess the image
        img_resized = image.resize((28, 28))
        img_tensor = transforms.ToTensor()(img_resized)
        img_normalized = transforms.Normalize((0.1376800686120987,), (0.3125477433204651,))(img_tensor)
        
        # Make prediction
        with torch.no_grad():
            input_tensor = img_normalized.unsqueeze(0).to(device)
            pred = ensemble_predict(models, input_tensor)
        
        # Update the result label with the predicted digit
        result_label.config(text=f"Predicted Digit: {pred.item()}")

    # Create buttons and label
    clear_button = tk.Button(root, text="Clear", command=clear_canvas)
    clear_button.pack()

    predict_button = tk.Button(root, text="Predict", command=predict_digit)
    predict_button.pack()

    result_label = tk.Label(root, text="Predicted Digit: ")
    result_label.pack()

    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the model or directory containing model files")
    args = parser.parse_args()
    main(args)