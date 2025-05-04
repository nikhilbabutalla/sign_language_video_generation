import os
import whisper
import shutil

# Define the model name and paths
MODEL_NAME = "large"
MODEL_DIR = "whisper_model"
MODEL_FILENAME = "whisper_large_model.pt"

def load_whisper_model():
    # Check if the model directory exists, create if not
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Load the model (this will download it if not already cached)
    model = whisper.load_model(MODEL_NAME)

    # Move the downloaded model to your specified filename
    downloaded_model_path = f"{MODEL_NAME}.pt"  # Default download location
    desired_model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

    if not os.path.exists(desired_model_path):
        print("Saving the Whisper model to:", desired_model_path)
        shutil.move(downloaded_model_path, desired_model_path)  # Move the model to your specified directory

    return model

# Load the pre-trained Whisper model
model = load_whisper_model()


