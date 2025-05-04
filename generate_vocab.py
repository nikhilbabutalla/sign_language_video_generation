import json
import os

def generate_vocabulary(dataset_path="data/isl_data.json", vocab_path="data/vocab.json"):
    """Generates the vocabulary from the dataset."""

    try:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {dataset_path}")
        return

    all_words = set()
    for pair in dataset:
        all_words.update(pair["input"].split())
        all_words.update(pair["target"].split())

    # Add special tokens FIRST
    word_to_index = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<START>": 2,
        "<END>": 3,
    }

    # Assign unique indices to words, starting AFTER special tokens.
    next_index = len(word_to_index)  # Gets the next index after the special tokens.
    for word in all_words:
        word_to_index[word] = next_index
        next_index += 1

    # Reverse mapping for predictions
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    # Save vocab
    try:
        with open(vocab_path, "w") as f:  # Changed to data/vocab.json to match other file.
            json.dump({"word_to_index": word_to_index, "index_to_word": index_to_word}, f, indent=4)
        print("✅ Vocabulary generated and saved in data/vocab.json")
    except IOError:
        print(f"Error: Could not write vocabulary to {vocab_path}")

# Generate vocabulary if it doesn't exist.
if not os.path.exists("data/vocab.json"):
    generate_vocabulary()
else:
    generate_vocabulary() #generate anyways, to make sure the indexes are correct.