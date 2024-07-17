import argparse
import os
from glob import glob

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import tkinter as tk
from datasets import load_dataset, get_dataset_names

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(model_path):
    # Check if the path is a directory or a single file
    if os.path.isdir(model_path):
        model_files = glob(os.path.join(model_path, "*.jit"))
    else:
        model_files = [model_path]
    
    # Load and return the models
    return [torch.jit.load(model_file, map_location=DEVICE).to(DEVICE).eval() for model_file in model_files]

def ensemble_predict(models, input_tensor):
    # Get predictions from all models
    outputs = [model(input_tensor) for model in models]
    avg_output = sum(outputs) / len(outputs)
    return avg_output.argmax(dim=1, keepdim=True)

class ImageRecognitionGUI:
    def __init__(self, models, dataset_class):
        self.models = models
        self.dataset_class = dataset_class

        self.setup_gui()
        self.setup_image()

    def setup_gui(self):
        # Create the main window
        self.root = tk.Tk()
        self.root.title(f"{self.dataset_class.__name__} Recognizer")

        # Create the drawing canvas
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas.pack()

        # Create the clear button
        clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        clear_button.pack()

        # Create the predict button
        predict_button = tk.Button(self.root, text="Predict", command=self.predict_class)
        predict_button.pack()

        # Create the label to display the prediction result
        self.result_label = tk.Label(self.root, text="Predicted Class: ")
        self.result_label.pack()

        # Bind the paint function to mouse motion
        self.canvas.bind("<B1-Motion>", self.paint)

    def setup_image(self):
        # Create a new image and drawing context
        self.image = Image.new("L", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        # Calculate the coordinates for the oval
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)

        # Draw on both the canvas and the image
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill="white")

    def clear_canvas(self):
        # Clear the canvas and reset the image
        self.canvas.delete("all")
        self.setup_image()

    def predict_class(self):
        # Resize the image to 28x28 (standard input size for MNIST-like datasets)
        img_resized = self.image.resize((28, 28))

        # Convert the image to a PyTorch tensor
        img_tensor = transforms.ToTensor()(img_resized)

        # Make the prediction
        with torch.no_grad():
            input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            pred = ensemble_predict(self.models, input_tensor)

        # Get the predicted class name and update the label
        predicted_class = self.dataset_class.classes[pred.item()]
        self.result_label.config(text=f"Predicted Class: {predicted_class}")

    def run(self):
        # Start the Tkinter event loop
        self.root.mainloop()

def main(args):
    # Load the models
    models = load_models(args.model_path)

    # Load the dataset
    dataset = load_dataset(args.dataset)
    dataset_class = dataset["dataset_class"]

    # Create and run the GUI
    gui = ImageRecognitionGUI(models, dataset_class)
    gui.run()

if __name__ == "__main__":
    dataset_names = get_dataset_names()
    parser = argparse.ArgumentParser(description="Image Recognition GUI")
    parser.add_argument("model_path", type=str, help="Path to the model or directory containing model files")
    parser.add_argument("--dataset", type=str, choices=dataset_names, default='mnist', help="Dataset to use")
    args = parser.parse_args()
    
    # Run the main function
    main(args)