import re
from keras.utils import pad_sequences
import numpy as np
import pandas as pd
import torch
import torchtext
from torch import nn
import time
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.layers import Embedding
import tensorflow as tf
from sklearn.model_selection import train_test_split
import contractions

print("PyTorch Version : {}".format(torch.__version__))
print("Torch Text Version : {}".format(torchtext.__version__))

dataFrame = pd.read_csv('NLP Datasets/DrugReview/drugsComTest_raw.csv')

columns = ['review', 'condition', 'drugName', 'usefulCount']
df = dataFrame.loc[:, columns]
features = ['review', 'drugName', 'usefulCount']
target = ['condition']

X = dataFrame.loc[:, features]
y = dataFrame.loc[:, target]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33)


def removeNumbers(input):
    outputRew = []
    outputDrug = []
    outputUseFul = []
    for i in range(len(input)):
        outputRew.append(re.sub(r'[0-9]', '#', input.iloc[i].review))
        outputDrug.append(input.iloc[i].drugName)
        outputUseFul.append(input.iloc[i].usefulCount)
    data = {'review': outputRew, 'drugName': outputDrug, 'usefulCount': outputUseFul}
    df = pd.DataFrame(data=data)
    df.to_csv('test.csv')
    return df


# Remember that the text input will change when other data is passed in.
def cleanSpecialCharText(textInput):
    # print(textInput)
    outputRew = []
    outputDrug = []
    outputUseFul = []
    pattern = r'[^a-zA-z0-9\s]'
    # For loop to cycle through the array and remove special characters
    for i in range(len(textInput)):
        outputRew.append(re.sub(pattern, '', textInput.iloc[i].review))
        outputDrug.append(textInput.iloc[i].drugName)
        outputUseFul.append(textInput.iloc[i].usefulCount)
    # Convert the array back into a dataframe
    data = {'review': outputRew, 'drugName': outputDrug, 'usefulCount': outputUseFul}
    df = pd.DataFrame(data=data)
    return df


# This one has issues
def removeContractions(inputText):
    # print(inputText)
    outputRew = []
    outputDrug = []
    outputUseFul = []
    for i in range(len(inputText)):
        # Check for contractions and remove them if found
        outputRew.append(contractions.fix(inputText.iloc[i].review))
        outputDrug.append(inputText.iloc[i].drugName)
        outputUseFul.append(inputText.iloc[i].usefulCount)
    data = {'review': outputRew, 'drugName': outputDrug, 'usefulCount': outputUseFul}
    df = pd.DataFrame(data=data)
    return df


def removeNan(input):
    # print(type(input))
    result = []
    for i in range(len(input)):
        if (type(input.iloc[i].condition) == str):
            result.append(input.iloc[i].condition)
    data = {'condition': result}
    df = pd.DataFrame(data=data)
    return df


'''
train_X = cleanSpecialCharText(train_X)
train_X = removeNumbers(train_X)
train_X = removeContractions(train_X)


test_X = cleanSpecialCharText(test_X)
test_X = removeNumbers(train_X)
test_X = removeContractions(test_X)

print('done')
'''

# train_y = removeNan(train_y)
# train_y = cleanSpecialCharText(train_y)
# train_y = removeNumbers(train_y)
# train_y = removeContractions(train_y)

# test_y = removeNan(test_y)
# test_y = removeNumbers(test_y)
# test_y = cleanSpecialCharText(test_y)
# test_y = removeContractions(test_y)

# Tokinzier
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(list(train_X) + list(test_X))
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)

# Adding a pad sequence
train_X = pad_sequences(train_X, maxlen=50)
test_X = pad_sequences(test_X, maxlen=50)

# Use the label encoder to convert the train_y and test_y to numbers
# print(train_y['condition'].unique())
# print(test_y['condition'].unique())
le = LabelEncoder()
train_y = le.fit_transform(train_y['condition'])

# Still has some sort of error that is pain
# test_y = le.transform(test_y['condition'])


# Different attempt at using Glove
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english")  # We'll use tokenizer available from PyTorch
tokenizer("Hello, How are you?")
from torchtext.vocab import GloVe

global_vectors = GloVe(name='840B', dim=300)
embeddings = global_vectors.get_vecs_by_tokens(tokenizer("Hello, How are you?"), lower_case_backup=True)
print(embeddings.shape)
global_vectors.get_vecs_by_tokens([""], lower_case_backup=True)

from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

max_words = 30
embed_len = 300


def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [tokenizer(x) for x in X]
    X = [tokens + [""] * (max_words - len(tokens)) if len(tokens) < max_words else tokens[:max_words] for tokens in X]
    X_tensor = torch.zeros(len(batch), max_words, embed_len)
    for i, tokens in enumerate(X):
        X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
    return X_tensor.reshape(len(batch), -1), torch.tensor(Y) - 1  # Subtracted 1 from labels to bring in range [0,1,2,3] from [1,2,3,4]


target_classes = ["anger", "fear", "joy", "sadness"]

# Create New Column with numbers that correspond with an associated emotion: [[Anger,1],[Fear,2],[Joy,3],[Sadness,4]]
def addNumEmotion(dataFrame):
    result = []
    textArr = []
    for i in range(len(dataFrame)):
        if (dataFrame.iloc[i].sentiment == 'anger'):
            result.append(1)
        elif (dataFrame.iloc[i].sentiment == 'fear'):
            result.append(2)
        elif (dataFrame.iloc[i].sentiment == 'joy'):
            result.append(3)
        elif (dataFrame.iloc[i].sentiment == 'sadness'):
            result.append(4)
        textArr.append(dataFrame.iloc[i].content)
    data = {'Class Index': result, 'content': textArr}
    df = pd.DataFrame(data=data)
    df.head()
    print(df)
    return df

# Import my dataset (ISER-simple)
ISEARDataFrame = pd.read_csv('NLP Datasets/ISEAR - simple/eng_dataset.csv')

addNumEmotion = addNumEmotion(ISEARDataFrame)

# addNumEmotion.to_csv('emotText.csv', index=False)

addNumEmotion = pd.read_csv('emotText.csv')

# train_dataset, test_dataset = torchtext.datasets.AG_NEWS() # DataPipe that yields tuple of [label (1 to 4) and text]
# train_dataset, test_dataset = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)

dataToUse = addNumEmotion.apply(tuple, axis=1)


print(dataToUse)

train_dataset, test_dataset = dataToUse, dataToUse
train_dataset, test_dataset = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=vectorize_batch)
test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=vectorize_batch)


# Define a neural network to process the data

from torch import nn
from torch.nn import functional as F


class EmbeddingClassifier(nn.Module):
    def __init__(self):
        super(EmbeddingClassifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(max_words * embed_len, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, len(target_classes)),
        )

    def forward(self, X_batch):
        return self.seq(X_batch)


# Train the Neural network

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import gc


def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [], [], []
        for X, Y in val_loader:
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())
            Y_shuffled.append(Y)
            Y_preds.append(preds.argmax(dim=-1))

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))


def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    for i in range(1, epochs + 1):
        losses = []
        for X, Y in tqdm(train_loader):
            Y_preds = model(X)

            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 5 == 0:
            print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
            CalcValLossAndAccuracy(model, loss_fn, val_loader)


from torch.optim import Adam

epochs = 25
learning_rate = 1e-3

loss_fn = nn.CrossEntropyLoss()
embed_classifier = EmbeddingClassifier()
optimizer = Adam(embed_classifier.parameters(), lr=learning_rate)

TrainModel(embed_classifier, loss_fn, optimizer, train_loader, test_loader, epochs)


# Reference https://coderzcolumn.com/tutorials/artificial-intelligence/how-to-use-glove-embeddings-with-pytorch

# Evaluate the model
def MakePredictions(model, loader):
    Y_shuffled, Y_preds = [], []
    for X, Y in loader:
        preds = model(X)
        Y_preds.append(preds)
        Y_shuffled.append(Y)
    gc.collect()
    Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

    return Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()


Y_actual, Y_preds = MakePredictions(embed_classifier, test_loader)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
print("\nClassification Report : ")
print(classification_report(Y_actual, Y_preds, target_names=target_classes))
print("\nConfusion Matrix : ")
print(confusion_matrix(Y_actual, Y_preds))
