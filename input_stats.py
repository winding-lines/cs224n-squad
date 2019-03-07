"""Compute stats on the input datasets.

Usage:
    > python input_stats.py

Implementation:
    Load the index
"""
import numpy as np
import json

class WordStats:
    def __init__(self, file: str):
        """Compute word stats in the preprocessed npz file.
        
        Load the preprocessed file and counts the number of idx == 0 or 1.
        Based on setup.py 0 is the idx for null and 1 is the idx for OOV
        """
        data = np.load(file)
        OOV_IDX = 1
        NULL_IDX = 1
        # ['context_idxs', 'context_char_idxs', 'ques_idxs', 'ques_char_idxs', 'y1s', 'y2s', 'ids']
        context_idxs = data['context_idxs']
        oov = 0
        total = 0
        for one in context_idxs:
            oov += np.count_nonzero(one == OOV_IDX)
            total += np.count_nonzero(one != NULL_IDX)

        self.count = len(context_idxs)
        self.oov_words = oov
        self.words = total

class ExampleStats:

    def __init__(self, file:str):
        """Compute stats on the number of examples.
        """
        with open(file, "r") as fh:
            source = json.load(fh)
            data = source['data']
            count = 0
            for article in data:
                for para in article['paragraphs']:
                    count += len(para['qas'])

            self.count = count

if __name__ == '__main__':

    for prefix in ["train", "dev"]:
        examples_stats = ExampleStats(f"data/{prefix}-v2.0.json")
        word_stats = WordStats(f"data/{prefix}.npz")
        missing_ratio = (examples_stats.count - word_stats.count) * 100.0 / examples_stats.count
        word_ratio = word_stats.oov_words * 100.0 / word_stats.words

        print(f"set: {prefix}, examples: {word_stats.count}/{examples_stats.count} {missing_ratio:.2}%  oov: {word_stats.oov_words}, total: {word_stats.words}, ratio: {word_ratio:.2}%\n")
