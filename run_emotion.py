import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from keras.models import Sequential
import cv2
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.neural_network import MLPClassifier
import joblib

# Load the trained physiological model
def load_physiological_model(model_path="models/physio_model.pkl"):
    return joblib.load(model_path)


# Load the pre-trained image model
def load_image_model(weight_file):
    # Define the architecture of the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    # Load weights into the model
    model=load_weights(weight_file)
    return model


# Predict emotion using the image-based model
def predict_emotion_image(image_path, emotion_model):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (48, 48))
    gray_image = gray_image.astype("float32") / 255.0
    gray_image = np.expand_dims(gray_image, axis=-1)
    gray_image = np.expand_dims(gray_image, axis=0)
    return emotion_model.predict(gray_image)[0]


# Multimodal fusion
def predict_multimodal_emotion(image_probs, physio_probs):
    fusion_probs = 0.5 * image_probs + 0.5 * physio_probs
    final_prediction = np.argmax(fusion_probs)
    emotion_labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    return emotion_labels[final_prediction]


# GUI Implementation
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        image_path.set(file_path)


def predict_emotion():
    try:
        # Get physiological inputs
        hr = float(heart_rate.get())
        sc = float(skin_conductance.get())
        br = float(breathing_rate.get())
        hrv = float(hr_variability.get())
        bt = float(body_temp.get())

        physiological_input = np.array([[hr, sc, br, hrv, bt]])
        physio_probs = physio_model.predict_proba(physiological_input).flatten()

        # Get image input
        img_path = image_path.get()
        if not img_path:
            messagebox.showerror("Error", "Please select an image file.")
            return
        image_probs = predict_emotion_image(img_path, image_model)

        # Perform multimodal fusion
        final_emotion = predict_multimodal_emotion(image_probs, physio_probs)
        result_label.config(text=f"Predicted Emotion: {final_emotion}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Load models
physio_model_path = "models/physio_model.pkl"  # Replace with your physiological model path
image_model_path = "models/emotion_model.h5"  # Replace with your image model weight file
physio_model = load_physiological_model(physio_model_path)
image_model = load_image_model(image_model_path)

# Tkinter GUI
root = tk.Tk()
root.title("Multimodal Emotion Recognition")

# Input fields
tk.Label(root, text="Physiological Signals").grid(row=0, column=0, columnspan=2, pady=10)

tk.Label(root, text="Heart Rate (HR):").grid(row=1, column=0, sticky="e")
heart_rate = tk.Entry(root)
heart_rate.grid(row=1, column=1)

tk.Label(root, text="Skin Conductance (SC):").grid(row=2, column=0, sticky="e")
skin_conductance = tk.Entry(root)
skin_conductance.grid(row=2, column=1)

tk.Label(root, text="Breathing Rate (BR):").grid(row=3, column=0, sticky="e")
breathing_rate = tk.Entry(root)
breathing_rate.grid(row=3, column=1)

tk.Label(root, text="Heart Rate Variability (HRV):").grid(row=4, column=0, sticky="e")
hr_variability = tk.Entry(root)
hr_variability.grid(row=4, column=1)

tk.Label(root, text="Body Temperature (BT):").grid(row=5, column=0, sticky="e")
body_temp = tk.Entry(root)
body_temp.grid(row=5, column=1)

tk.Label(root, text="Image File:").grid(row=6, column=0, sticky="e")
image_path = tk.StringVar()
image_entry = tk.Entry(root, textvariable=image_path, width=40)
image_entry.grid(row=6, column=1)

browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.grid(row=6, column=2)

# Predict button
predict_button = tk.Button(root, text="Predict Emotion", command=predict_emotion, bg="green", fg="white")
predict_button.grid(row=7, column=0, columnspan=3, pady=10)

# Result display
result_label = tk.Label(root, text="Predicted Emotion: ", font=("Arial", 14))
result_label.grid(row=8, column=0, columnspan=3, pady=20)

root.mainloop()


# In[ ]:




