#!/usr/bin/env python
# coding: utf-8

# In[10]:


# D:\Ninad\SeventhProject\Book1.xlsx
# get_ipython().system('pip install transformers')


# In[11]:


# get_ipython().system('pip install torch nn')


# In[66]:


import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score


# In[67]:


# Load the Excel file into a DataFrame
excel_file = 'D:\\Ninad\\SeventhProject\\Book1.xlsx'
df = pd.read_excel(excel_file)


# In[68]:


df


# In[69]:


# Preprocess the data
# Convert text to lowercase and remove whitespace
df.drop(df.columns[[2, 3, 4]], axis=1, inplace=True)
df['Query'] = df['Query'].str.lower().str.strip()
df


# In[70]:


# Encode intents as numerical labels
intent_map = {'Time': 0, 'Weather': 1, 'Search': 2,
              'Joke': 3, 'News': 4, 'Wikipedia': 5, 'Play': 6}
df['Intent'] = df['Intent'].map(intent_map)
df


# In[71]:


null_mask = df.isnull().any(axis=1)
null_rows = df[null_mask]
null_rows


# In[72]:


# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# In[73]:


train_df


# In[74]:


test_df


# In[75]:


# Define a custom Dataset class to handle data loading


class IntentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data.iloc[idx]['Query']
        intent = self.data.iloc[idx]['Intent']

        encoding = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'intent': torch.tensor(intent, dtype=torch.long)
        }


# In[77]:


# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=7)


# In[78]:


# Define a simple neural network for intent recognition
class IntentModel(nn.Module):
    def __init__(self, bert_model):
        super(IntentModel, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits


# In[79]:


# Initialize the intent recognition model
model = IntentModel(bert_model)


# In[80]:


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)


# In[81]:


# Define training parameters
batch_size = 16
max_length = 128
epochs = 10


# In[82]:


# Create DataLoader for training and testing sets
train_dataset = IntentDataset(train_df, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = IntentDataset(test_df, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[83]:


# Train the model
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['intent']

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")


# In[84]:


# Evaluate the model
model.eval()
all_predictions = []
all_targets = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['intent']

        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1).tolist()

        all_predictions.extend(predictions)
        all_targets.extend(labels.tolist())

accuracy = accuracy_score(all_targets, all_predictions)
print(f"Test Accuracy: {accuracy}")


# In[85]:


torch.save(model.state_dict(), "D:/Ninad/SeventhProject/intent_model.pth")


# In[ ]:
