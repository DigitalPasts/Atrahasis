import argparse
import nltk.lm
import numpy as np
from itertools import tee


parser = argparse.ArgumentParser(description='nltk 2-Gram calculate MRR score')

parser.add_argument('--TestLines', type=str, default='./logs/MRRLines_20102019.txt',
                    help='Test lines')
parser.add_argument('--data', type=str, default='./data/achemenet_data_20102019',
                    help='location of the data to evaluate - create ngram from')

args = parser.parse_args()


# Collect data

with open(args.data + '/train.txt', 'r') as file:
    train_file = file.read().split()
with open(args.TestLines) as f:
    test_lines = f.readlines()
with open(args.data + '/test.txt', 'r') as f:
    test_file = f.read().split()

# Init model and parameters

depth = 2
counter = nltk.lm.NgramCounter([nltk.ngrams(tuple(train_file), i) for i in range(1, depth + 1)])

languageModel = nltk.lm.models.KneserNeyInterpolated(order=2, vocabulary=nltk.lm.Vocabulary(train_file), counter=counter)
# languageModel = nltk.lm.MLE(order=2, vocabulary=nltk.lm.Vocabulary(train_file), counter=counter)

scores = 0
removeIdx = 4
ranks = []

# MRR score

for line in test_lines:
    line = line.split()

    context = line[removeIdx - depth: removeIdx - 1]
    predictions = sorted(languageModel.context_counts(tuple(context)).items(),
                         reverse=True, key=lambda x: x[1])

    print(f"Query: {line[:removeIdx]}")
    print("Proposed Results: ", end="")
    for location, prediction in enumerate(predictions[:20]):
        print(f"{str(location)}) {prediction}", end=", ")
    print("...")

    rank = -1
    for location, prediction in enumerate(predictions):
        if prediction[0] == line[removeIdx - 1]:
            rank = location + 1
    print(f"Correct response: {line[removeIdx - 1]} "
          f"Rank: {rank} "
          f"Reciprocal rank: 1/{rank} = {1 / rank if rank > 0 else 0}\n")
    if rank > 0:
        scores += 1 / rank
    ranks.append(rank)

print(f"Mean reciprocal rank: {scores}/{len(test_lines)} = {scores / len(test_lines)}")

# Top hit

ranks = np.array(ranks)
ranks[ranks == -1] = 2000000
print('Hit@1 = ', (ranks <= 1).mean())
print('Hit@5 = ', (ranks <= 5).mean())
print('Hit@10 = ', (ranks <= 10).mean())


# Cross entropy and perplexity

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


a = list(pairwise(train_file))
b = a

# for item in a:
#     if languageModel.entropy([item]) != float("inf"):
#         b.append(item)

print(f"Train Perplexity: {languageModel.perplexity(b)}")
print(f"Train Entropy: {languageModel.entropy(b)}")
print(len(a),len(b),len(b)*1./len(a))
a = list(pairwise(test_file))
b = a
#
# for item in a:
#     if languageModel.entropy([item]) != float("inf"):
#         b.append(item)

print(f"Test Perplexity: {languageModel.perplexity(b)}")
print(f"Test Entropy: {languageModel.entropy(b)}")
print(len(a),len(b),len(b)*1./len(a))
