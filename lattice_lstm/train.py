import time
import torch
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from model import LatticeLSTM
from load_data import char2idx, word2idx, label2idx, data_generator


char_size = len(char2idx)
word_size = len(word2idx)
embed_dim = 300
hidden_dim = 128


EPOCHS = 20
TRAIN_DATA_PATH = ''

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LatticeLSTM(char_size, word_size, label2idx, embed_dim, hidden_dim).to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.001)

start = time.time()

loss_vals = []
for epoch in range(EPOCHS):
    epoch_loss = []
    for sent, input_ids, input_words, labels_idx in data_generator(TRAIN_DATA_PATH, char2idx, word2idx, label2idx, shuffle=True):
        model.zero_grad()
        loss = model.neg_log_likelihood(input_ids, input_words, labels_idx)
        loss.backward()

        epoch_loss.append(loss.item())
        optimizer.step()

    loss_vals.append(np.mean(epoch_loss))
    print(f'Epoch{epoch} - Loss:{np.mean(epoch_loss)}')

torch.save(model.state_dict(), "model.pth")
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)
end = time.time()
print(f'Training costs: {end - start} seconds')