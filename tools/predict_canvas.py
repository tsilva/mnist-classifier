import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import tkinter as tk
from models import load_model
from datasets import load_dataset, get_dataset_names, get_dataset_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageRecognitionGUI:
    def __init__(self, model, dataset_class, dataset_metrics):
        self.model = model
        self.dataset_class = dataset_class
        self.dataset_metrics = dataset_metrics

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
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        # Apply the default transformation to the datasets
        # (normalize the inputs using the overall mean and standard deviation)
        dataset_mean = self.dataset_metrics['mean']
        dataset_std = self.dataset_metrics['std']
        normalize_transform = transforms.Normalize((dataset_mean,), (dataset_std,))
        input_tensor = normalize_transform(input_tensor)

        # Make the prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        prediction = torch.argmax(outputs, dim=1).item()
    
        # Get the predicted class name and update the label
        predicted_class = self.dataset_class.classes[prediction]
        self.result_label.config(text=f"Predicted Class: {predicted_class}")

    def run(self):
        # Start the Tkinter event loop
        self.root.mainloop()

def main(args):
    # Load dataste and calculate its metrics
    dataset = load_dataset(args.dataset)


    # Load the models
    model = load_model(args.model_path, DEVICE)

    # Set model in eval mode (disables dropout and batch 
    # normalization, which would affect the prediction)
    model.eval()

    # Load the dataset
    dataset = load_dataset(args.dataset)
    dataset_class = dataset["dataset_class"]
    dataset_metrics = get_dataset_metrics(dataset)

    # Create and run the GUI
    gui = ImageRecognitionGUI(model, dataset_class, dataset_metrics)
    gui.run()

if __name__ == "__main__":
    dataset_names = get_dataset_names()
    parser = argparse.ArgumentParser(description="Image Recognition GUI")
    parser.add_argument("model_path", type=str, help="Path to the model or directory containing model files")
    parser.add_argument("--dataset", type=str, choices=dataset_names, default='mnist', help="Dataset to use")
    args = parser.parse_args()
    
    # Run the main function
    main(args)