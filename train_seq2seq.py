import torch
import torch.nn as nn
import torch.optim as optim
import json
from seq2seq_model import Seq2Seq, load_vocab, Language
from torch.utils.data import Dataset, DataLoader

class ISLDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_seq = torch.tensor([self.vocab["word_to_index"].get(word, 1) for word in item["input"].lower().split()], dtype=torch.long)
        target_seq = torch.tensor([self.vocab["word_to_index"].get(word, 1) for word in item["target"].lower().split()], dtype=torch.long)
        return input_seq, target_seq

def collate_batch(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seq_padded = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True)
    target_seq_padded = torch.nn.utils.rnn.pad_sequence(target_seqs, batch_first=True)
    return input_seq_padded, target_seq_padded

with open("data/isl_data.json", "r") as f:
    dataset = json.load(f)

vocab, vocab_size = load_vocab("data/vocab.json")

HIDDEN_SIZE = 256
BATCH_SIZE = 16

model = Seq2Seq(vocab_size, HIDDEN_SIZE, vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=0) 
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = ISLDataset(dataset, vocab)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

EPOCHS = 100
for epoch in range(EPOCHS):
    for input_seq, target_seq in train_dataloader:
        optimizer.zero_grad()
        output = model(input_seq, target_seq)
        loss = criterion(output.permute(0, 2, 1), target_seq)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), "models/seq2seq_model.pth")
print(" Model trained and saved.")

model.load_state_dict(torch.load("models/seq2seq_model.pth"))

input_lang = Language(load_vocab("data/vocab.json")[0])
output_lang = Language(load_vocab("data/vocab.json")[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

sample_sentence = "hello"
predicted_translation = model.predict(sample_sentence, input_lang, output_lang, device)
print(f"Input: {sample_sentence}")
print(f"Prediction: {predicted_translation}")