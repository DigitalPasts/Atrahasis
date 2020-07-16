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
import numpy as np
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
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--start', action='store_true',
                    help='only use first part of sentence')
parser.add_argument('--num-missing', type=int,default=1, help='what index to remove')
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
num_missing = args.num_missing


def sentence_to_tokens(sentence, corpus):
    tokens = []
    for word in sentence:
        if word == '<BRK>':
            tokens.append(-1)
        else:
            try:
                tokens.append(corpus.dictionary.word2idx[word])
            except:
                tokens.append((corpus.dictionary.word2idx["<UNK>"]))
                print(f"Error at adding the word '{word}' to the model since its not in the dictionary"
                      f" (Added '<UNK>' instead)")
    return tokens

def beam_search(sentence,model,beam_size=100):
    first = np.nonzero(sentence == -1)[0][0]
    sentence = torch.from_numpy(sentence).view(-1,1).to(device)
    with torch.no_grad():
        # init until first missing word
        if first>0:
            hidden = model.init_hidden(1)
            output, hidden = model(sentence[:first], hidden)
            log_like = torch.log(softmax(output))[-1,0,:]
            rankings = torch.argsort(-log_like)
            beam = [sentence.repeat(1, beam_size)]
            beam[0][first, :] = rankings[:beam_size]
            beam.append(log_like[rankings[:beam_size]])
            output, hidden = model(rankings[:beam_size].view(1,-1), (hidden[0].repeat(1,beam_size,1),hidden[1].repeat(1,beam_size,1)))
            beam.append(hidden)
        else:
            beam = [sentence.repeat(1, beam_size)]
            beam.append(torch.zeros(beam_size).to(device))
            beam.append(model.init_hidden(100))
        # continue sentence
        for i in range(first+1,len(sentence)):
            word = sentence[i]
            if word.item() != -1:
                log_like = torch.log(softmax(output))[0, :, :]
                beam[1] = beam[1] + log_like[:,word.item()]
                beam[2] = hidden
                output, hidden = model(beam[0][i:i + 1, :], beam[-1])
            else:
                log_like = torch.log(softmax(output))[0,:,:]
                tot_log_like = beam[1].view(-1, 1) + log_like
                rankings = torch.argsort(-tot_log_like.flatten())[:beam_size]
                rows = rankings / tot_log_like.shape[1]
                cols = rankings % tot_log_like.shape[1]
                new_beam = torch.zeros_like(beam[0])
                for j in range(beam_size):
                    new_beam[:,j] = beam[0][:,rows[j]]
                    new_beam[i,j] = cols[j]
                beam[0] = new_beam
        # rerank using latest log_like
        log_like = beam[1]
        rankings = torch.argsort(-log_like)
        results = torch.zeros_like(beam[0])
        for j in range(beam_size):
            results[:,j] = beam[0][:,rankings[j]]
        return results


with open(args.MRRLines, 'r') as file:
    good_lines = file.readlines()
with torch.no_grad():
    for line in good_lines:

        line = line.split()
        original = np.array(sentence_to_tokens(line,corpus))
        missing_index = np.random.choice(range(1,len(line)),num_missing)
        for idx in missing_index:
            #print(original[idx])
            line[idx] = '<BRK>'
        sentence = sentence_to_tokens(line,corpus)
        #print(line)
        # Enter all the words but the last to the model

        sentence = np.array(sentence)
        results = beam_search(sentence, model).data.cpu().numpy()
        diff = ((results - original.reshape(-1, 1)) ** 2).sum(0)
        if len(np.where(diff==0)[0]) == 0:
            rank=-1
        else:
            rank = np.where(diff == 0)[0][0]+1
            scores += 1./rank
        ranks.append(rank)
        #print()


print(f"Mean reciprocal rank: {scores}/{len(good_lines)} = {scores / len(good_lines)}")
ranks = np.array(ranks)
ranks[ranks==-1] = 2000000
print('Hit@1 = ',(ranks<=1).mean())
print('Hit@5 = ',(ranks<=5).mean())
print('Hit@10 = ',(ranks<=10).mean())