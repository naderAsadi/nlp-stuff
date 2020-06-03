import numpy as np
import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F

corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]


##########################################
# Data Manipulation
##########################################

words = [x.split() for x in corpus]
vocabulary = []
for w in words:
    for t in w:
        vocabulary.append(t)
vocabulary = set(vocabulary)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)


window_size = 2
idx_pairs = []
for sentence in words:
    indices = [word2idx[w] for w in sentence]
    for center_pos in range(len(indices)):
        for w in range(-window_size, window_size + 1):
            context_pos = center_pos + w
            if context_pos < 0 or context_pos >= len(indices) or center_pos == context_pos:
                continue
            idx_pairs.append((indices[center_pos], indices[context_pos]))

idx_pairs = np.array(idx_pairs)


###########################################
# Word2Vec Training
###########################################

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary).float()
    x[word_idx] = 1.0
    return x

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 100
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data[0]
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')









