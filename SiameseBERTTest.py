import torch
from transformers import BertModel, BertTokenizer
from SiameseBERT import SiameseBERT
from TripletLoss import TripletLoss
from train_loop import train_loop

# Load the BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# Define NN, loss and optimizer
model = SiameseBERT(bert_model, tokenizer)
loss_fn = TripletLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Example usage, need to put real dataset here
anchors = ["I am happy", "He is running", "It works", "she writes a letter"]
positives = ["I feel great", "She is jogging", "It operates", "she writes a message with her pan"]
negatives = ["I am sad", "He is sleeping", "Nothing seems to work", "she hates writing"]
batch_size = 1
num_epochs = 4
model.to_device()

for epoch in range(num_epochs):
    loss = train_loop(model, optimizer, loss_fn, anchors, positives, negatives, batch_size)
    print(f"Epoch {epoch + 1} Loss: {loss:.4f}")
