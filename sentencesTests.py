
import argparse
from copy import copy
import torch
import numpy as np
import data
import model



parser = argparse.ArgumentParser(description='PyTorch Language Model')

# Model parameters
parser.add_argument('--data', type=str, default='./data/achemenet_data_20102019',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model_best.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
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
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

def filter_non_printable(str):
  return ''.join([c for c in str if c != '\xa0'])

def clear_line(line):
    return ' '.join(line.split(' ')[1:])


def get_data(location):
    texts = []
    with open(location) as f:
        lines = f.read().split("\n")
        while '' in lines:
            lines.remove('')
        for i, line in enumerate(lines):
            line = filter_non_printable(line)
            if "__" in line:
                texts.append([line, clear_line(lines[i+1]), clear_line(lines[i+2]),
                             clear_line(lines[i + 3]), clear_line(lines[i+4])])
    with open(location) as f:
        lines = f.read().split("\n")
        while '' in lines:
            lines.remove('')
        counter = 0
        for line in lines:
            line = filter_non_printable(line)
            if line[0].isdigit() and len(line.split()) > 4:
                texts[counter].append(clear_line(line))
                counter += 1
    return texts


count = 0
question_count = 0
texts = get_data("./testSentences/questionsDataFull")
log_file = "./testSentences/answers.txt"
wrong_log_file = "./testSentences/wrong_answers.txt"
net = model.RNNModel(args.model, 1549, args.emsize, args.nhid, args.nlayers, 0., args.tied)
net.load_state_dict(torch.load(args.checkpoint))
net.eval()

if args.cuda:
    net.cuda()
else:
    net.cpu()
softmax = torch.nn.Softmax(2)
# Loading the data
corpus = data.Corpus(args.data)

def get_token_index(sentence, token='__'):
    counter = 0
    idx = -1
    for i,s in enumerate(sentence):
        if s==token:
            idx = i
            counter += 1
    assert(counter==1)
    return idx


def tokenize(sentences, corpus):
    tokenized_sentences = []
    for sentence in sentences:
        temp = []
        for word in sentence:
            temp.append(corpus.dictionary.word2idx[word])
        tokenized_sentences.append(temp)
    tokenized_sentences = np.array(tokenized_sentences).transpose()
    return tokenized_sentences


def sentence_prob(sentence,output):
    prob = 0.
    for i,token in enumerate(sentence[1:]):
        prob += output[i,token]
    return prob.item()


with open(wrong_log_file,'w') as wf:
    with open(log_file,'w') as f:
        with torch.no_grad():
            for idx, item in enumerate(texts[:]):
                question_count += 1
                # import ipdb; ipdb.set_trace()
                print("Question", idx + 1)
                s = item[0].replace('___','__')
                sentence = s.split()
                answer = item[-1].split()
                print(sentence)
                remove_index = get_token_index(sentence)
                answer = answer[remove_index]
                options = item[1:-1]
                answer = get_token_index(options, answer)

                sentences = [copy(sentence) for x in range(len(options))]
                for i,option in enumerate(options):
                    sentences[i][remove_index] = option
                tokenized_sentences = tokenize(sentences,corpus)
                tokenized_sentences = torch.from_numpy(tokenized_sentences)
                if args.cuda:
                    tokenized_sentences = tokenized_sentences.cuda()
                hidden = net.init_hidden(len(options))
                outputs, _ = net(tokenized_sentences, hidden)
                outputs = torch.log(softmax(outputs))
                probs = []
                for i in range(len(options)):
                    probs.append(sentence_prob(tokenized_sentences[:,i],outputs[:,i,:]))
                probs = np.array(probs)
                order = np.argsort(-probs)
                pred = probs.argmax()
                if pred == answer:
                    count += 1
                f.write('Question {}\n'.format(idx+1))
                f.write(item[0]+'\n')
                for i,option in enumerate(options):
                    f.write('{}) {}\n'.format(i,option))
                f.write('Answer: {}. Correct answer: {}.\n'.format(pred,answer))
                f.write('Order = '+str(order)+'\n\n')
                ## print to log
                if pred != answer:
                    wf.write('Question {}\n'.format(idx + 1))
                    wf.write(item[0] + '\n')
                    for i, option in enumerate(options):
                        wf.write('{}) {}\n'.format(i, option))
                    wf.write('Answer: {}. Correct answer: {}.\n\n'.format(pred, answer))
            f.write("Correct in total {}, out of {}. {:2f}%".format(count,question_count,count*100./question_count))
            print("Correct in total {}, out of {}. {:2f}%".format(count,question_count,count*100./question_count))

