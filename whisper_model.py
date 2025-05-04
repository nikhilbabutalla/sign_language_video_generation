import os
import whisper
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define the model name and paths
MODEL_NAME = "tiny"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
MODEL_FILENAME = "tiny.pt"  # Update this if the filename is different

def load_whisper_model():
    model_path = os.path.join(CACHE_DIR, MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        print("Model file not found. Downloading...")
        model = whisper.load_model(MODEL_NAME)  # This will download the model
        print("Model downloaded successfully.")
    else:
        print("Loading existing model from cache...")
        model = whisper.load_model(MODEL_NAME)  # Load from the cache
        print("Model loaded successfully from cache.")
    
    return model

# Load the Whisper model
model = load_whisper_model()

# Optional: Add a line to indicate the script has finished running
print("Whisper model is ready to use.")
