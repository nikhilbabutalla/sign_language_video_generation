from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from pydub import AudioSegment
import whisper
from moviepy import VideoFileClip, concatenate_videoclips
import re
import uuid
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

model = whisper.load_model("tiny")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STATIC_FOLDER = 'static'
os.makedirs(STATIC_FOLDER, exist_ok=True)

word_to_video = {}
letter_to_video = {}

dataset_path = 'C:\\Users\\nikhi\\OneDrive\\Documents\\SignLanguage\\SignLang\\SignLang\\INDIAN SIGN LANGUAGE ANIMATED VIDEOS'

def load_video_mappings():
    global word_to_video, letter_to_video
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0].strip().lower()
                if len(file_name) == 1 and file_name.isalpha():
                    letter_to_video[file_name] = file_path
                else:
                    word_to_video[file_name] = file_path

load_video_mappings()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process_text(text):
    words = word_tokenize(text.lower())
    processed_words = []
    for word in words:
        if word in word_to_video:
            processed_words.append(word)
        elif word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            if lemma in word_to_video:
                processed_words.append(lemma)
    return " ".join(processed_words)

def convert_to_wav(input_file):
    file_extension = os.path.splitext(input_file)[1].lower()
    if file_extension in ['.mp3', '.mp4']:
        audio = AudioSegment.from_file(input_file)
        wav_file = os.path.join(UPLOAD_FOLDER, "converted_audio.wav")
        audio.export(wav_file, format="wav")
        return wav_file
    elif file_extension == '.wav':
        return input_file
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def audio_to_text(audio_file_path):
    wav_file = convert_to_wav(audio_file_path)
    result = model.transcribe(wav_file, beam_size=1)
    return result["text"]

def generate_isl_video(input_text):
    video_clips = []
    words = input_text.split()

    for word in words:
        if word in word_to_video:
            video_path = word_to_video[word]
            clip = VideoFileClip(video_path)
            video_clips.append(clip)
        else:
            for letter in word:
                if letter in letter_to_video:
                    video_path = letter_to_video[letter]
                    clip = VideoFileClip(video_path)
                    video_clips.append(clip)

    if video_clips:
        unique_id = str(uuid.uuid4())[:8]
        final_video_file_path = os.path.join(STATIC_FOLDER, f"output_{unique_id}.mp4")
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video.write_videofile(final_video_file_path, codec="libx264")
        return final_video_file_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        audio_file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(audio_file_path)

        try:
            transcribed_text = audio_to_text(audio_file_path)
            processed_text = process_text(transcribed_text)
            video_path = generate_isl_video(processed_text)

            static_video_path = f"/static/{os.path.basename(video_path)}"
            return jsonify({
                "transcribed_text": transcribed_text,
                "processed_text": processed_text,
                "video_path": static_video_path
            })
        except Exception as e:
            return jsonify({"message": str(e)}), 500

    return jsonify({"message": "No file uploaded!"}), 400

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(STATIC_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8085)