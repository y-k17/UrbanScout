import torch
from torchvision import transforms
from PIL import Image

# Define the YOLO model class
class YOLOModel(torch.nn.Module):
    def __init__(self, model_path):
        super(YOLOModel, self).__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def forward(self, x):
        return self.model(x)

# Function to count objects in an image
def count_objects(image_path, classes_of_interest):
    # Load and preprocess the image
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)

    # Load the YOLO model
    model = YOLOModel(model_path="")

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Get predicted labels and counts
    predicted_labels = outputs[0]['class_labels']
    predicted_counts = {class_name: 0 for class_name in classes_of_interest}

    # Count objects of interest
    for label in predicted_labels:
        if label in classes_of_interest:
            predicted_counts[label] += 1

    # Print counts
    for class_name, count in predicted_counts.items():
        print(f"Number of {class_name}s detected: {count}")

if __name__ == "__main__":
    # Define the path to your input image
    input_image_path = "urban_app/roundabout_015.jpg"

    # Define the classes of interest
    classes_to_count = ["bridge", "freeway", "intersection", "roundabout"]

    # Call the function to count objects
    count_objects(input_image_path, classes_to_count)
