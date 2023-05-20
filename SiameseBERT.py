import torch
import torch.nn as nn


# Siamese network model
class SiameseBERT(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, device=None):
        super(SiameseBERT, self).__init__()
        self.bert = bert_model
        self.tokenizer = bert_tokenizer
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, 128)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        embeddings = self.fc(pooled_output)
        return embeddings

    def to_device(self):
        self.to(self.device)

    # Generate embeddings for a sentences using the network
    def get_embeddings(self, sentences):
        inputs = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            embeddings = self(input_ids, attention_mask)

        return embeddings
