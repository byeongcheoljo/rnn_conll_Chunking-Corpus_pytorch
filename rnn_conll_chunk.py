import torch
import numpy
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


file_name = 'Train data.txt'

sentence = []
word = []
vocab = set([])
setTags = set([])
tag = []
tags = []

for ln in open(file_name):
    if ln == "\n":
        sentence.append(word)
        tags.append(tag)
        word = []
        tag = []
        continue
        
    line = ln.strip().split()
    word.append(line[0])
    vocab.add(line[0])
    tag.append(line[2])
    setTags.add(line[2])
    
vocab_set = list(vocab)
set_tags = (list(setTags))

word2index = {word : index for index, word in enumerate(vocab_set,2)}
word2index["<pad>"] = 0
word2index["<unk>"] = 1
index2word = {index: word for word, index in word2index.items()}

tag2index = {tag:index for index, tag in enumerate(set_tags)}
index2tag = {index:tag for tag, index in tag2index.items()}
tags2index = [[tag2index[t] for t in tag]for tag in tags]

sentence2index = [[word2index[word] for word in sent]for sent in sentence]
sequence_length = torch.LongTensor(list(map(len, sentence2index)))
tags_length = torch.LongTensor(list(map(len, tags2index)))

seq_zero_vector = torch.zeros(len(sentence2index), sequence_length.max()).long()
tensor = torch.ones((2,), dtype=torch.int64)
tag_zero_vector = tensor.new_full((len(tags2index), tags_length.max()), -100).long()
for idx, (seqindex, seqlen) in enumerate(zip(sentence2index, sequence_length)):
    seq_zero_vector[idx, : seqlen] = torch.LongTensor(seqindex)

for idx, (tagindex, taglen) in enumerate(zip(tags2index, tags_length)):
    tag_zero_vector[idx, : taglen] = torch.LongTensor(tagindex)

sequence_length, perm_idx = sequence_length.sort(0, descending=True)
tags_length, tag_perm_idx = tags_length.sort(0, descending=True)

seq_zero_vector = seq_zero_vector[perm_idx]
tag_zero_vector = tag_zero_vector[tag_perm_idx]


batch_size = 8936
max_sequence_length = 78
input_size = 10
hidden_size = 20
num_layer = 2
vocab_size = len(index2word)
tagSize = len(setTags)

class Model(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings = vocab_size, embedding_dim = input_size)
        self.rnn = nn.RNN(input_size, hidden_size,nonlinearity ='relu', num_layers=num_layer, dropout = .3, bidirectional = False, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        print(self.embedding_layer,"embedding_layer")
        print(self.rnn, "rnn")
        print(self.fc, "fc")
#
    def forward(self, x):
        out = self.embedding_layer(x)
        out = out.view(batch_size, max_sequence_length, input_size)
        print(out.shape)
        h_0 = torch.zeros(num_layer, batch_size, hidden_size)
        packed_input = pack_padded_sequence(out, sequence_length.cpu().numpy(),  batch_first=True)
        packed_output, hidden = self.rnn(packed_input, h_0)

        print(hidden.shape)
        output, _ = pad_packed_sequence(packed_output, padding_value = 0,batch_first = True)
        out = self.fc(output)

        return out
#
#
model = Model(vocab_size, input_size, hidden_size, tagSize)

cost = nn.CrossEntropyLoss(ignore_index = -100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
out = model(seq_zero_vector)

#
for epoch in range(101):
    optimizer.zero_grad()
    y_pred = model(seq_zero_vector)
    # loss = cost(torch.argmax(y_pred, dim=2), y)
    loss = cost(y_pred.view(-1,22), tag_zero_vector.view(-1))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        pred = y_pred.softmax(-1).argmax(dim=2).tolist()
        print(pred)
        print(tag2index)
        print(loss)
        print(sequence_length, perm_idx,"sequenceLength, perm_idx")
