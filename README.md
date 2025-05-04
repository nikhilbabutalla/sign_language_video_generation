# Indian Sign Language (ISL) Translator using Seq2Seq and Whisper

This project is a complete pipeline for converting speech/audio into Indian Sign Language (ISL) video output. It integrates Whisper ASR for speech-to-text, a Seq2Seq model for sentence optimization, and a video generator that maps processed text to pre-recorded ISL sign clips.

## 🔧 Features

- 🎤 Converts audio input to text using OpenAI Whisper.
- 🧠 Optimizes text to match ISL vocabulary using a Seq2Seq model.
- 🗂 Handles synonym replacement to maximize word coverage.
- 🎬 Generates a final sign language video by stitching relevant clips.
- 🧪 Includes utilities for dataset generation and vocabulary handling.

## 📁 Project Structure

- `app.py` – Main driver script for the full pipeline (audio to video).
- `whisper_model.py` – Handles speech-to-text conversion.
- `seq2seq_model.py` – Sequence-to-sequence model for sentence simplification.
- `sentence_generator.py` – Converts input into ISL-compatible sentence.
- `dataset_generator.py` – Prepares data for training and testing.
- `synonym_pairs.json` – Stores synonym mappings for word optimization.
- `converted_audio.wav` – Example audio input file.
- `l_output_video.mp4` – Example generated ISL output video.
- `Homepage.html` – Basic HTML interface (for future web integration).

## 🛠 Technologies Used

- Python
- OpenAI Whisper
- TensorFlow/Keras (for Seq2Seq)
- MoviePy
- Flask (if UI used)
- JSON for data management

## 🚀 How to Run

1. Clone the repository.
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
3.Run the full pipeline:

   ```bash
   python app.py```
4.Provide audio input and get ISL video output.

🤝 Contributing
Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
