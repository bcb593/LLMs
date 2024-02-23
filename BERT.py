import torch
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


tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')

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

class fineTuneBert(nn.Module):
    def __init__(self, num_labels=2):
        super(fineTuneBert, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased',return_dict=False)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        _,outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs)
        return logits

train_dataset = SentimentAnalysisDataset(
    texts=imdb['train']['text'],
    labels=imdb['train']['label'],
    tokenizer=tokenizer
)
validation_dataset=SentimentAnalysisDataset(
     texts=imdb['train']['text'],
    labels=imdb['train']['label'],
    tokenizer=tokenizer
)
test_dataset = SentimentAnalysisDataset(
    texts=imdb['test']['text'],
    labels=imdb['test']['label'],
    tokenizer=tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model=fineTuneBert()
epochs=20

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)

for epoch in range(epochs):
    print('Epoch:',epoch)
    for batch in tqdm(train_loader): 
        input_ids=batch['input_ids'].to(device)
        labels=batch['labels'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        output=model(input_ids,attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss, total_accuracy = 0, 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    print(f"Accuracy: {correct_predictions.double() / len(data_loader.dataset):.4f}")

evaluate_model(model, test_loader, device)






