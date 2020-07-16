import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import data
from time import gmtime, strftime
import model
import sys
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Achemenet RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/achemenet_data_20102019',
                    help='location of the data corpus')
parser.add_argument('--name', type=str, default='model',
                    help='name of the saved files')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=350,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--weight_dacay', type=float, default=0.0001,
                    help='weight dacay')
parser.add_argument('--lr', type=float, default=10,
                    help='initial learning rate')
parser.add_argument('--beta', type=float, default=0.1,
                    help='momentum param')
parser.add_argument('--lr_decay_factor', type=int, default=2,
                    help='learning rate decay after no improvement')
parser.add_argument('--min_lr', type=float, default=0.0001,
                    help='the learning rate will not get lower than that')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--epochs_without_improvement', type=int, default=4,
                    help='number of epochs waited with no improvement before decaying the lr')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
logFile = open("./trainLogs/" + args.name + strftime("_%H:%M:%S", gmtime()) + ".txt", 'w')

logFile.write('python '+' '.join(sys.argv)+'\n')
logFile.write("Model - " + str(args.model) + "\n")
logFile.write("Units per hidden layer - " + str(args.nhid) + "\n")
logFile.write("Number of layers - " + str(args.nlayers) + "\n")
logFile.write("Initial learning rate - " + str(args.lr) + "\n")
logFile.write("Batch size - " + str(args.batch_size) + "\n")
logFile.write("Dropout - " + str(args.dropout) + "\n")

print("Model - ", args.model)
print("Units per hidden layer - ", args.nhid)
print("Number of layers - ", args.nlayers)
print("Initial learning rate - ", args.lr)
print("Batch size - ", args.bptt)
print("Dropout - ", args.dropout)

prog_started_time = time.time()
logFile.write("Program started at:" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print("Program started at:", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# Lists to plot the loss over epochs graph
epochs_list = []
valid_loss_list = []
train_loss_list = []

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# Sets the log_interval to be shorter than the number of batches in epoch so there will always be a log
args.log_interval = min(args.log_interval, len(train_data) // args.bptt - 1)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.beta, weight_decay=args.weight_dacay)


###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, Variable):
        return torch.zeros_like(h) #Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    """
    Takes a batch of data from the source starting from i.
    :param source: The data source (while train it's train_data)
    :param i: The location in the data the batch is taken from
    :param evaluation: Sets volatile to be false (in case backpropagation is not needed)
    :return: A tuple, the data and the target of the data while learning.
    """
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation)
    target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the new learning rate that was given"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)
        optimizer.step()
        total_loss += loss.data

        if batch == 0 and epoch == 1:
            cur_loss = total_loss.item() / args.log_interval
            logFile.write('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.7f} | '
                          'loss {:5.2f} | ppl {:8.2f} |\n'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                cur_loss, math.exp(cur_loss)))
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.7f} | '
                  'loss {:5.2f} | ppl {:8.2f} |'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                cur_loss, math.exp(cur_loss)))

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            train_loss_list.append(cur_loss)
            elapsed = time.time() - start_time
            logFile.write('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.7f} | ms/batch {:5.2f} | '
                          'loss {:5.2f} | ppl {:8.2f} |\n'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.7f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} |'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None
no_improvement_count = 0
save_file = args.name + '_best.pt'
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        epochs_list.append(epoch)
        val_loss = evaluate(val_data)
        train()
        val_loss = evaluate(val_data)
        valid_loss_list.append(val_loss)
        logFile.write('-' * 89 + "\n")
        logFile.write('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}\n'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss, math.exp(val_loss)))
        logFile.write('-' * 89 + "\n")
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            no_improvement_count = 0
            torch.save(model.state_dict(), save_file)
            best_val_loss = val_loss
        else:
            no_improvement_count += 1
            if no_improvement_count >= args.epochs_without_improvement:
                lr /= args.lr_decay_factor
                lr = max([lr, args.min_lr])
                no_improvement_count = 0
                adjust_learning_rate(optimizer=optimizer, new_lr=lr)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model.load_state_dict(torch.load(save_file))
model.eval()
# Run on test data.
test_loss = evaluate(test_data)

logFile.write('=' * 89 + "\n")
logFile.write('| End of training | test loss {:5.2f} | test ppl {:8.2f} |\n'.format(
    test_loss, math.exp(test_loss)))
logFile.write('=' * 89 + "\n")
logFile.write('=' * 89 + "\n")

print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} |'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

logFile.write("Program ended at: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
logFile.write("Run took: {:5.2f}s\n".format(time.time() - prog_started_time))
print("Program ended at: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print("Run took: ", time.time() - prog_started_time, "s")

logFile.close()

# Initiating the loss over epochs graph
# with open("./graphs/" + args.name + strftime("_%H:%M:%S", gmtime()) + ".png", 'w') as graphFile:
#     graphFile.write('Loss / Epoch\n')
#     graphFile.write('Loss\n')
#     graphFile.write('Epochs\n')
#
#     graphFile.write([epochs_list, valid_loss_list])
#     graphFile.write("\n")
#     graphFile.write([epochs_list, train_loss_list])
