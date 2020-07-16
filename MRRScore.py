###############################################################################
# Language Modeling
#
# This file complete broken sentences using the language model
#
###############################################################################

import argparse
import subprocess
from copy import deepcopy
import sys

import torch
from torch.autograd import Variable
import model

import data
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MRR score Language Model')

# Model parameters
parser.add_argument('--data', type=str, default='./data/achemenet_data_20102019',
                    help='location of the data corpus')
parser.add_argument('--log', type=str, default='./logs/MRRLog20102019.txt',
                    help='location of the log file')
parser.add_argument('--MRRLines', type=str, default='./logs/MRRLines_20102019.txt',
                    help='location of the log file')
parser.add_argument('--checkpoint', type=str, default='model_best.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--start', action='store_true',
                    help='only use first part of sentence')
parser.add_argument('--remove-index', type=int,default=4, help='what index to remove')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=350,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
args = parser.parse_args()


# Set the random seed manually for reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.manual_seed(args.seed)
else:
    device = 'cpu'


if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

# Opening the model to generate from


# Loading the data
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
softmax = torch.nn.Softmax(2)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, 0.5, args.tied)
model.load_state_dict(torch.load(args.checkpoint))
model = model.to(device)
model.eval()
scores = 0.
ranks = []
removeIdx = args.remove_index


with open(args.MRRLines, 'r') as file:
    good_lines = file.readlines()
with torch.no_grad():
    for line in good_lines:
        hidden = model.init_hidden(1)
        line = line.split()
        #print(line)
        # Enter all the words but the last to the model
        sentence = []

        for word in line:
            try:
                sentence.append(corpus.dictionary.word2idx[word])
            except:
                sentence.append((corpus.dictionary.word2idx["<UNK>"]))
                print(f"Error at adding the word '{word}' to the model since its not in the dictionary"
                      f" (Added '<UNK>' instead)")
        sentence = np.array(sentence)
        word_input = torch.from_numpy(sentence[:removeIdx]).view(-1,1)
        word_input.data = word_input.data.to(device)
        # Current top 100 sentences to complete
        top_100_sentences = []

        # Getting the top 100 matches to the first word
        # Getting the current output and hidden layers
        output, _ = model(word_input, hidden)
        output = torch.log(softmax(output))
        logits = output[-1, 0, :].data.cpu().numpy()
        indexs = np.argsort(-logits)
        rank = np.where(indexs==sentence[removeIdx])[0][0] +1
        if not args.start and rank>=100.: #recompute base on top 100
            rank = -1 #if not top 100 we know we will fail
        elif not args.start:
            new_index = rank-1
            beam = sentence.reshape(sentence.shape[0], 1).repeat(100, 1)
            beam[removeIdx,:] = indexs[:100]
            hidden = model.init_hidden(100)
            beam = torch.from_numpy(beam)
            beam = beam.to(device)
            output2, _ = model(beam, hidden)
            output2 = torch.log(softmax(output2))

            probs = np.zeros(100)
            for i in range(sentence.shape[0]-1):
                for j in range(100):
                    probs[j] += output2[i,j,beam[i+1,j]].item()
            rank =np.where(np.argsort(-probs)==new_index)[0][0] +1
        if rank > 0:
            scores += 1. / rank
        ranks.append(rank)
        #print()

with open(args.log, 'w') as log:
    for rank in ranks:
        log.write(str(rank))
        log.write("\n")

print(f"Mean reciprocal rank: {scores}/{len(good_lines)} = {scores / len(good_lines)}")
ranks = np.array(ranks)
ranks[ranks==-1] = 2000000
print('Hit@1 = ',(ranks<=1).mean())
print('Hit@5 = ',(ranks<=5).mean())
print('Hit@10 = ',(ranks<=10).mean())