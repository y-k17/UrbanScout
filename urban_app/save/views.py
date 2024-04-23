from django.shortcuts import render
from .models import *
import json
from django.core import serializers
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
from django.db.models import Count
import re
import os
import glob
from datetime import datetime
from datetime import date
from django.views.decorators.cache import never_cache
from django.core.files.storage import FileSystemStorage
from threading import Thread
import threading
from django.core.mail import EmailMessage
import pickle
import cv2
import numpy as np
import rasterio
from ultralytics import YOLO
from PIL import Image
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt


model=YOLO("Models/best.pt")


@never_cache
def show_index(request):
    return render(request, "login.html", {})


@never_cache
def logout(request):
    if 'uid' in request.session:
        del request.session['uid']
    return render(request,'login.html')




def check_login(request):
    username = request.POST.get("username")
    password = request.POST.get("password")

    if username=="Admin" and password=="Admin":

        return HttpResponse("<script>alert('Login Successful');window.location.href='/show_home_admin/'</script>")
    else:
        return HttpResponse("<script>alert('Invalid');window.location.href='/show_index/'</script>")


@never_cache
###############ADMIN START
def show_home_admin(request):
    return render(request,'home_admin.html') 



@never_cache
def show_emergency(request):

    return render(request,'emergency_admin.html') 


@never_cache
def display_safety(request):

    return render(request,'safety_admin.html') 



class_descriptions = {
    "commercial_area": "An area primarily designated for commercial activities such as shopping centers, office buildings, and retail stores.",
    "desert": "A dry, barren area with little or no vegetation, often characterized by sand dunes and extreme temperatures.",
    "forest": "A large area covered chiefly with trees and undergrowth, typically home to diverse wildlife.",
    "ground_track_field": "A flat, open area used for athletic events such as track and field competitions.",
    "industrial_area": "An area characterized by the presence of factories, warehouses, and other industrial facilities.",
    "island": "A piece of land surrounded by water, smaller than a continent and larger than a rock.",
    "lake": "A large body of water surrounded by land, typically freshwater.",
    "meadow": "An open area of grassland, often used for grazing livestock or as a recreational space.",
    "medium_residential": "An area primarily consisting of medium-density residential housing, such as suburban neighborhoods.",
    "mountain": "A large natural elevation of the Earth's surface, typically with steep sides and a peak.",
    "parking_lot": "An area designated for parking vehicles, often found adjacent to buildings or recreational facilities.",
    "rectangular_farmland": "Farmland that is divided into rectangular plots for agricultural purposes.",
    "river": "A large, natural flow of water that typically empties into a sea, lake, or another river.",
    "sparse_residential": "An area with low-density residential housing, typically characterized by larger plots of land.",
    "wetland": "An area of land consisting of marshes or swamps, often characterized by saturated soil and abundant vegetation.",
    "beach": "A sandy or pebbly shore, typically along the edge of a body of water such as an ocean or lake.",
    "chaparral": "A type of vegetation consisting of dense, shrubby plants adapted to dry climates, often found in Mediterranean regions."
}


def get_metadata(image_path):
    with rasterio.open(image_path) as src:
        # Read the image bands
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)
        
        # Get image metadata
        metadata = src.meta
        print("Image Metadata:")
        print(metadata)

        return metadata


recommendations = {
    "commercial_area": "Consider zoning regulations and urban design principles to ensure efficient use of space and accessibility for businesses and customers.",
    "desert": "Preserve natural habitats and consider desert conservation strategies to protect biodiversity and prevent desertification.",
    "forest": "Implement sustainable forestry practices and consider forest management strategies to preserve and enhance urban green spaces.",
    "ground_track_field": "Provide adequate facilities for sports and recreational activities, including maintenance and accessibility considerations.",
    "industrial_area": "Implement pollution control measures and consider buffer zones to mitigate environmental impact on surrounding residential areas.",
    "island": "Implement conservation measures to protect island ecosystems and consider sustainable tourism practices to minimize ecological footprint.",
    "lake": "Implement water quality management measures and consider recreational access while preserving natural habitats and ecosystems.",
    "meadow": "Implement green space planning strategies and consider wildlife conservation measures to preserve urban biodiversity.",
    "medium_residential": "Consider mixed land-use development to promote walkability and access to amenities, while ensuring affordable housing options.",
    "mountain": "Implement land-use planning measures to protect mountain ecosystems and consider sustainable tourism practices to minimize ecological impact.",
    "parking_lot": "Implement smart parking solutions to optimize space usage and reduce traffic congestion, while considering green infrastructure for stormwater management.",
    "rectangular_farmland": "Implement agricultural zoning regulations and consider sustainable farming practices to support local food production.",
    "river": "Implement riverfront revitalization projects to enhance recreational opportunities and improve water quality through riparian restoration efforts.",
    "sparse_residential": "Consider infill development strategies to optimize land use and promote neighborhood connectivity while preserving open space.",
    "wetland": "Implement wetland conservation measures and consider urban green infrastructure to mitigate flood risk and support biodiversity.",
    "beach": "Implement coastal management strategies to protect against erosion and sea-level rise while providing public access and preserving natural ecosystems.",
    "chaparral": "Implement wildfire prevention measures and consider native plant landscaping to reduce fire risk while preserving chaparral habitats."
}


@never_cache
def display_test_page(request):
    
    return render(request,'testpage.html') 


# To predict on a single image input
def predict_single_image(image_path, label_encoder, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = np.array(image, dtype="float16") / 255.0
    image = np.expand_dims(image, axis=0)  # Adding batch dimension
    prediction = model.predict(image)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]



def perform_classification(image_array):
    # Load the LabelEncoder
    with open('Models/label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)

    # Load the model architecture from JSON
    with open('Models/model1.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    # Load the model weights
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('Models/model1.h5')

    # Ensure image has 3 color channels (remove alpha channel if present)
    if image_array.shape[-1] == 4:  # Check if image has 4 channels (RGBA)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB

    # Resize the input image to match the model's expected input size
    resized_image = cv2.resize(image_array, (150, 150))  # Resize to (150, 150)
    resized_image = resized_image.astype("float32") / 255.0  # Normalize pixel values

    # Expand dimensions to create a batch of one image
    input_image = np.expand_dims(resized_image, axis=0)

    # Perform prediction using the loaded model
    prediction = loaded_model.predict(input_image)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def perform_object_detection(image_array, output_dir, file_number):
    # Placeholder for object detection logic (replace with actual implementation)
    # Assuming `model` performs object detection and saves the result
    
    # Convert RGB to BGR for OpenCV compatibility (cv2.imwrite expects BGR)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


    # Call iyour object detection model with the input image array
    # Replace `model` with your actual object detection model
    processed_image = model(image_bgr, save=True, project=output_dir, exist_ok=True)

    # Placeholder data for description, recommendation, and metadata (based on object detection)
    description = "Description based on object detection"
    recommendation = "Recommendation based on object detection"
    meta_data = "Metadata based on object detection"

    # Generate a unique filename for the processed image
    output_filename = os.path.join(output_dir, f"output_{np.random.randint(10000)}.jpg")
    print(output_filename)
    cv2.imwrite(image_bgr, output_filename)

    # Save the processed image (assuming `processed_image` is the result of object detection)
    cv2.imwrite(output_filename, processed_image)
    
    return output_filename, description, recommendation, meta_data


def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        # Retrieve the uploaded file
        f2 = request.FILES['file']
        
        # Load the uploaded image using PIL
        uploaded_image = Image.open(f2)
        width, height = uploaded_image.size
        
        # Divide the image into a 2x2 grid
        sub_images = []
        sub_width = width // 2
        sub_height = height // 2
        
        for i in range(2):
            for j in range(2):
                # Define the region to crop (left, upper, right, lower)
                box = (j * sub_width, i * sub_height, (j + 1) * sub_width, (i + 1) * sub_height)
                # Crop the sub-image
                sub_image = uploaded_image.crop(box)
                # Convert PIL Image to numpy array
                sub_image_array = np.array(sub_image)

                # Perform classification on the sub-image
                label = perform_classification(sub_image_array)
                print(label)

                # Perform object detection and retrieve processed data
                output_filename, description, recommendation, meta_data = perform_object_detection(sub_image_array, "urban_app/static/Output", (i+i+j+1))

                # Append results for the current sub-image
                sub_images.append((sub_image_array, label, output_filename, description, recommendation, meta_data))
        
        # Prepare context data to pass to the template
        context = {
            "results": sub_images,  # Ensure this contains valid data for each sub-image
        }

        print(context)  # Debug: Print context to console for verification

        # Render the result page template with the processed data
        return render(request, 'resultpage.html', context)
    
    else:
        # Handle GET request or invalid form submission
        return render(request, 'upload.html')

