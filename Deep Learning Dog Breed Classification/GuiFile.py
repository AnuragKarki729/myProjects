import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from ExtractNames import get_breed_name
from keras.utils import load_img, img_to_array
import cv2
import os
import shutil
from collections import Counter
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16

# Load the trained models
model_inception = tf.keras.models.load_model('DogModelInception.keras')
model_val_resnet = tf.keras.models.load_model('DogModelResNet50.keras')
model_val_VGG16 = tf.keras.models.load_model('DogModelVGG16.keras')
model_inception2 = tf.keras.models.load_model('DogModelInception2.keras')
model_resnet2 = tf.keras.models.load_model('DogModelResNet502.keras')

# Define class names
class_names = get_breed_name()

def preprocess_for_model(image_path, model_name):
    # Define target sizes based on model
    if model_name == 'VGG16':
        target_size = (224, 224)  # VGG16 typically uses 224x224
        from keras.applications.vgg16 import preprocess_input
    elif model_name in ['Inception', 'Inception2']:
        target_size = (299, 299)  # Inception models typically use 299x299
        from keras.applications.inception_v3 import preprocess_input
    elif model_name in ['ResNet', 'ResNet2']:
        target_size = (299, 299)  # ResNet models typically use 299x299
        from keras.applications.resnet50 import preprocess_input
    else:
        target_size = (299, 299)  # Default size for other models
        preprocess_input = None  # No specific preprocessing for unspecified models

    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image
    if preprocess_input:
        img_array = preprocess_input(img_array)
    else:
        img_array = img_array / 255.0  # Default normalization

    return img_array

def classify_image(image_path):
    # Preprocess the image for different models
    img_inception_array = preprocess_for_model(image_path, 'Inception')
    img_vgg16_array = preprocess_for_model(image_path, 'VGG16')
    img_resnet_array = preprocess_for_model(image_path, 'ResNet')
    
    # Get predictions from models
    predictions_vgg16 = model_val_VGG16.predict(img_vgg16_array)
    predicted_class_vgg16 = np.argmax(predictions_vgg16, axis=1)[0]
    confidence_vgg16 = np.max(predictions_vgg16)
    
    predictions_inception = model_inception.predict(img_inception_array)
    predicted_class_inception = np.argmax(predictions_inception, axis=1)[0]
    confidence_inception = np.max(predictions_inception)
    
    predictions_inception2 = model_inception2.predict(img_inception_array)
    predicted_class_inception2 = np.argmax(predictions_inception2, axis=1)[0]
    confidence_inception2 = np.max(predictions_inception2)
    
    predictions_val_resnet = model_val_resnet.predict(img_resnet_array)
    predicted_class_val_resnet = np.argmax(predictions_val_resnet, axis=1)[0]
    confidence_val_resnet = np.max(predictions_val_resnet)
    
    predictions_resnet = model_resnet2.predict(img_resnet_array)
    predicted_class_resnet = np.argmax(predictions_resnet, axis=1)[0]
    confidence_resnet = np.max(predictions_resnet)
    
    def correct_class_name(class_index):
        class_name = class_names[class_index]
        if class_name == "Shih":
            return "Shih-Tzu"
        
        return class_name

    corrected_class_vgg16 = correct_class_name(predicted_class_vgg16)
    corrected_class_inception = correct_class_name(predicted_class_inception)
    corrected_class_inception2 = correct_class_name(predicted_class_inception2)
    corrected_class_val_resnet = correct_class_name(predicted_class_val_resnet)
    corrected_class_resnet = correct_class_name(predicted_class_resnet)
    
    # Update GUI with results
    result_label_vgg16.config(
        text=f"VGG16 Model Prediction: {corrected_class_vgg16} \nConfidence: {confidence_vgg16:.2f}",
        fg="blue"
    )
    
    result_label_inception1.config(
        text=f"Inception Model 1 Prediction: {corrected_class_inception} \nConfidence: {confidence_inception:.2f}",
        fg="green"
    )
    
    result_label_inception2.config(
        text=f"Inception Model 2 Prediction: {corrected_class_inception2} \nConfidence: {confidence_inception2:.2f}",
        fg="green"
    )
    
    result_label_resnet1.config(
        text=f"ResNet Model (val_accuracy) Prediction: {corrected_class_val_resnet} \nConfidence: {confidence_val_resnet:.2f}",
        fg="red"
    )
    
    result_label_resnet2.config(
        text=f"ResNet Model (accuracy) Prediction: {corrected_class_resnet} \nConfidence: {confidence_resnet:.2f}",
        fg="red"
    )
    
    return corrected_class_vgg16, corrected_class_inception, corrected_class_inception2, corrected_class_val_resnet, corrected_class_resnet


def classify_video(video_path):
    if os.path.exists("video_frames"):
        shutil.rmtree("video_frames")
    os.makedirs("video_frames")

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    success = True
    
    # Initialize lists to hold corrected predictions
    predictions_list_vgg16 = []
    predictions_list_inception = []
    predictions_list_inception2 = []
    predictions_list_resnet = []
    predictions_list_resnet2 = []

    while success:
        success, frame = cap.read()
        if success and count % int(frame_rate) == 0:  
            resized_frame = cv2.resize(frame, (299, 299))
            frame_path = f"video_frames/frame_{count}.jpg"
            cv2.imwrite(frame_path, resized_frame)
            
            # Use the classify_image function for each frame
            (corrected_class_vgg16, corrected_class_inception, 
             corrected_class_inception2, corrected_class_val_resnet, 
             corrected_class_resnet) = classify_image(frame_path)
            
            # Append corrected predictions to lists
            predictions_list_vgg16.append(corrected_class_vgg16)
            predictions_list_inception.append(corrected_class_inception)
            predictions_list_inception2.append(corrected_class_inception2)
            predictions_list_resnet.append(corrected_class_val_resnet)
            predictions_list_resnet2.append(corrected_class_resnet)
            
            display_frame(resized_frame)
            window.update()
        count += 1
    
    # Determine the majority class for each model
    majority_class_vgg16 = Counter(predictions_list_vgg16).most_common(1)[0][0]
    majority_class_inception = Counter(predictions_list_inception).most_common(1)[0][0]
    majority_class_inception2 = Counter(predictions_list_inception2).most_common(1)[0][0]
    majority_class_resnet = Counter(predictions_list_resnet).most_common(1)[0][0]
    majority_class_resnet2 = Counter(predictions_list_resnet2).most_common(1)[0][0]
    
    def correct_class_name(class_name):
        if class_name == "Shih":
            return "Shih-Tzu"
        # Add other manual corrections here
        return class_name
    # Ensure that class names are indexed properly
    
    result_label_vgg16.config(text=f"Majority Class (VGG16 | val_accuracy): {correct_class_name(majority_class_vgg16)}", fg="blue")
    result_label_inception1.config(text=f"Majority Class (Inception | accuracy): {correct_class_name(majority_class_inception)}", fg="green")
    result_label_inception2.config(text=f"Majority Class (Inception2 | accuracy): {correct_class_name(majority_class_inception2)}", fg="green")
    result_label_resnet1.config(text=f"Majority Class (ResNet | val_accuracy): {correct_class_name(majority_class_resnet)}", fg="red")
    result_label_resnet2.config(text=f"Majority Class (ResNet2 | accuracy): {correct_class_name(majority_class_resnet2)}", fg="red")

    cap.release()
    shutil.rmtree("video_frames")

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        
        classify_image(file_path)
        
def open_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        classify_video(file_path)
        
def display_frame(frame):
    # Convert the OpenCV frame (BGR) to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL Image
    img = Image.fromarray(frame_rgb)

    # Resize for display in GUI (if needed)
    img = img.resize((200, 200))

    # Convert to ImageTk format for displaying in tkinter
    img_tk = ImageTk.PhotoImage(img)

    # Update the GUI label to show the frame
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Set up the GUI window
window = tk.Tk()
window.title("Dog Breed Classifier")

# Set up the frame for buttons and labels
frame = tk.Frame(window, padx=10, pady=10)
frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# Title label
title_label = tk.Label(frame, text="Dog Breed Classifier", font=("Helvetica", 20, "bold"))
title_label.pack(pady=(0, 10))

# Upload buttons
button_frame = tk.Frame(frame)
button_frame.pack(pady=10)

upload_image_button = tk.Button(button_frame, text="Upload Image", command=open_image, width=20, height=2, bg="#4CAF50", fg="white", font=("Helvetica", 14))
upload_image_button.pack(side=tk.LEFT, padx=5)

upload_video_button = tk.Button(button_frame, text="Upload Video", command=open_video, width=20, height=2, bg="#2196F3", fg="white", font=("Helvetica", 14))
upload_video_button.pack(side=tk.LEFT, padx=5)

# Image display label
img_label = tk.Label(frame, bg="#f0f0f0", borderwidth=2, relief="solid")
img_label.pack(pady=10)

# Classification result labels
result_label_vgg16 = tk.Label(frame, text="VGG16 Model results will appear here", font=("Helvetica", 16), wraplength=400)
result_label_vgg16.pack(pady=10)

result_label_inception1 = tk.Label(frame, text="Inception Model 1 results will appear here", font=("Helvetica", 16), wraplength=400)
result_label_inception1.pack(pady=10)

result_label_inception2 = tk.Label(frame, text="Inception Model 2 results will appear here", font=("Helvetica", 16), wraplength=400)
result_label_inception2.pack(pady=10)

result_label_resnet1 = tk.Label(frame, text="ResNet Model (val_accuracy) results will appear here", font=("Helvetica", 16), wraplength=400)
result_label_resnet1.pack(pady=10)

result_label_resnet2 = tk.Label(frame, text="ResNet Model (accuracy) results will appear here", font=("Helvetica", 16), wraplength=400)
result_label_resnet2.pack(pady=10)

# Start the GUI loop
window.mainloop()
