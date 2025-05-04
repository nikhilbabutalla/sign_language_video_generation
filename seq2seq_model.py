import torch
import torch.nn as nn
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

class Language:
    def __init__(self, vocab):
        self.word2index = vocab["word_to_index"]
        self.index2word = {index: word for word, index in self.word2index.items()}

    def sentence_to_tensor(self, sentence):
        words = sentence.lower().split()
        indexes = [self.word2index.get(word, 1) for word in words]
        return torch.tensor(indexes, dtype=torch.long)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        embedded = self.dropout(self.embedding(input_seq))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        if hidden.dim() == 4:
            hidden, cell = hidden.squeeze(0), cell.squeeze(0)
        elif hidden.dim() == 2:
            hidden, cell = hidden.unsqueeze(0), cell.unsqueeze(0)

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, embedding_dim=256, n_layers=2, dropout=0.5, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_size, n_layers, dropout).to(device)
        self.decoder = Decoder(output_dim, embedding_dim, hidden_size, n_layers, dropout).to(device)
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.8):
        batch_size, trg_len = target_seq.shape
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(input_seq)
        input_token = target_seq[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = target_seq[:, t] if teacher_force else top1

        return outputs.permute(1, 0, 2)

    def predict(self, sentence, input_lang, output_lang, device):
     with torch.no_grad():
        input_tensor = input_lang.sentence_to_tensor(sentence).unsqueeze(0).to(device)
        hidden, cell = self.encoder(input_tensor)

        start_token = output_lang.word2index["<START>"]
        decoder_input = torch.tensor([[start_token]], device=device)

        decoder_hidden, decoder_cell = hidden, cell
        predicted_words = []
        input_words = sentence.split()

        max_length = len(input_words)  

        for i in range(max_length):
            prediction, decoder_hidden, decoder_cell = self.decoder(decoder_input.squeeze(0), decoder_hidden, decoder_cell)
            topi = prediction.argmax(1)
            predicted_word_index = topi.item()

            if predicted_word_index in output_lang.index2word:
                predicted_word = output_lang.index2word[predicted_word_index]
            else:
                predicted_word = input_words[i]  

            if predicted_word == "<END>":
                break

            predicted_words.append(predicted_word)
            decoder_input = torch.tensor([[predicted_word_index]], device=device)

        return " ".join(predicted_words)

def load_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab["word_to_index"])
    return vocab, vocab_size

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = word_tokenize(text)
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(processed_words)

def tokenize_sentence(sentence, vocab):
    words = sentence.lower().split()
    word_to_index = vocab["word_to_index"]
    return [word_to_index.get(word, 1) for word in words]
__all__ = ["Seq2Seq", "load_vocab", "preprocess_text", "tokenize_sentence"]