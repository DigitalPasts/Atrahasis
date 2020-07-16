import argparse
import nltk.lm
import numpy as np
from itertools import tee

parser = argparse.ArgumentParser(description='nltk 2-Gram Bidirectional calculate MRR score')

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

# Init model and parameters

depth = 2
tree_size = 100
counter = nltk.lm.NgramCounter([nltk.ngrams(tuple(train_file), i) for i in range(1, depth + 1)])
languageModel = nltk.lm.MLE(order=2, vocabulary=nltk.lm.Vocabulary(train_file), counter=counter)

scores = 0
removeIdx = 4
ranks = []

# MRR score

# test_lines = test_lines[:1]
for line in test_lines:
    line = line.split()

    end = line[removeIdx]
    start = tuple(line[removeIdx - depth: removeIdx - 1])

    predictions_before = sorted(languageModel.context_counts(start).items(),
                                reverse=True, key=lambda x: x[1])[:tree_size]
    predictions_before = [(pred[0], languageModel.score(pred[0], context=start)) for pred in predictions_before]
    # Each prediction now is a tuple (prediction_word, prob)

    predictions_after = []
    for pred in predictions_before:
        predictions_after.append(
            (pred[0], pred[1] * languageModel.score(end, context=tuple([pred[0]])))
        )
    predictions = sorted(predictions_after, reverse=True, key=lambda x: x[1])

    # print(f"Query: {line[:removeIdx]}")
    # print("Proposed Results: ", end="")
    # for location, prediction in enumerate(complex_pred[:20]):
    #     print(f"{str(location)}) {prediction}", end=", ")
    # print("...")

    rank = -1
    for location, prediction in enumerate(predictions):
        if prediction[0] == line[removeIdx - 1]:
            rank = location + 1
    # print(f"Correct response: {line[removeIdx - 1]} "
    #       f"Rank: {rank} "
    #       f"Reciprocal rank: 1/{rank} = {1 / rank if rank > 0 else 0}\n")
    if rank > 0:
        scores += 1 / rank
    ranks.append(rank)

print(f"Mean reciprocal rank: {scores}/{len(test_lines)} = {scores / len(test_lines)}")

ranks = np.array(ranks)
ranks[ranks == -1] = 2000000
print('Hit@1 = ', (ranks <= 1).mean())
print('Hit@5 = ', (ranks <= 5).mean())
print('Hit@10 = ', (ranks <= 10).mean())

