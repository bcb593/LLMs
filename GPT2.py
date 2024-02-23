import torch
from transformers import GPT2Model, GPT2Tokenizer, AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from transformers import AdamW

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")

data_files = {"train":"train-00000-of-00001.parquet", "test":"test-00000-of-00001.parquet", "unsupervised":"unsupervised-00000-of-00001.parquet"}
imdb = load_dataset("parquet", data_dir="/scratch0/bashyalb/LLMs/imdb", data_files=data_files)

class SentimentAnalysisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_token_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class GPT2Classifier(nn.Module):
    def __init__(self, n_classes=2):
        super(GPT2Classifier, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]  # Use the representation of the [CLS] token
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

# Assume SentimentAnalysisDataset is defined similarly as before, adapted for GPT-2 tokenization

# Initialize tokenizer and datasets
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Make sure to update paths and dataset loading according to your setup
train_dataset = SentimentAnalysisDataset(imdb['train']['text'], imdb['train']['label'], tokenizer)
test_dataset = SentimentAnalysisDataset(imdb['test']['text'], imdb['test']['label'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model, optimizer, and loss function
model = GPT2Classifier().to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss().to(device)

# Training loop
model.train()
for epoch in range(3):  # Example: 3 epochs
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {total_loss / len(train_loader)}")

# Evaluation logic remains similar, ensure model is set to eval mode and calculate accuracy
