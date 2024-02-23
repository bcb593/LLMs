


import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn


if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Listing devices:")
    print(torch.cuda.list_gpu_processes())

data_files = {"train":"train-00000-of-00001.parquet", "test":"test-00000-of-00001.parquet", "unsupervised":"unsupervised-00000-of-00001.parquet"}
imdb = load_dataset("parquet", data_dir="../LLMS/imdb", data_files=data_files)

device_index = 2  # Change this to 0 or 2 to use the first or third GPU, respectively.
torch.cuda.set_device(device_index)
device = torch.device(f"cuda:{device_index}")
print(f"Using GPU: {torch.cuda.get_device_name(device)}")

tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[10]:


model=fineTuneBert()
epochs=20
model=model.to(device)

# In[12]:


for epoch in tqdm(range(epochs)):
   # print('Epoch:',epoch)
    for batch in train_loader: 
        input_ids=batch['input_ids'].to(device)
        labels=batch['labels'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        output=model(input_ids,attention_mask)





