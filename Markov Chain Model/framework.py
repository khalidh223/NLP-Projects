"""
framework.py
Author: Khalid Hussain

Framework for training a Markov chain statistical language model from user-supplied text
corpora. The models are parameterizable (word vs. character level,
order of n-gram), and they can generate text and perform authorship attribution.
"""

import nltk
import random
import string
import statistics
from tabulate import tabulate
nltk.download('punkt')

class Framework:
  """
    corpus -- string of filename
    level -- either 'word' or 'character'; defines whether to train a character
    level or word-level language model
    order -- number of previous tokens to consider, eg. 2 for a bigram model
    to_estimate -- boolean for whether model is used to estimate z-score for some
    unknown text
  """
  def __init__(self, corpus, level, order, to_estimate):
    self.transitions = {}
    self.level = level
    self.order = order
    text = open(corpus, encoding="utf-8").read()

    if to_estimate == False:
      self.tokens = self.tokens_from_string(text)
      self.train()
    else:
      self.testing_set = text[int(len(text)*0.8):]
      training_set = text[:int(len(text)*0.8)]
      self.tokens = self.tokens_from_string(training_set)
      self.train()
      self.calculate_mean_stdev()


  def tokens_from_string(self, text):
    """
    Returns level-appropriate list of tokens.

    text -- single string of text to tokenize
    """

    if self.level == "character":
      return list(text)
    elif self.level == "word":
      return nltk.word_tokenize(text)
    else:
      print("error: invalid level")


  def train(self):
    """
    Trains a model on an associated corpus.
    """
    self.transitions = {}
    if self.order > len(self.tokens) - 1:
        print("Unable to train: Hit upper bound on order, given corpus.")
    for i in range(0, len(self.tokens) - self.order):
      ngram = tuple(self.tokens[i:i+self.order])
      if ngram in self.transitions:
        self.transitions[ngram].append(self.tokens[i+self.order])
      elif ngram not in self.transitions:
        self.transitions[ngram] = [self.tokens[i+self.order]]


  def generate(self, length, prompt=""):
    """
    Generate random key in transitions dict, after that access
    key to generate random next token, keep going until length is up

    length -- int for length of generation
    prompt -- (optional) tuple of separated tokens to start generating words from
    """
    exclude_punct = "@#$%&^*\"’‘`'"
    stop_punct = ".,:;!?"
    toprint = ""
    # contractions = ["'s", "'nt", "'d", "'ll", "'re", "'ve", "n't"]

    # set up prompt by truncating or randomizing as necessary
    if prompt == "" or len(prompt) < self.order:
      prompt = random.choice(list(self.transitions.keys()))
      toprint = " ".join(prompt)
    else:
      toprint = " ".join(prompt)
      prompt = prompt[-self.order:]

    while length-self.order > 0:
      # select next word and truncate prompt for next loop
      generated = random.choice(self.transitions[prompt])
      prompt = prompt[1:] + (generated,)

      # add whitespace according to level
      if self.level == "word":
        for char in generated:
            if char in exclude_punct:
                break
            elif (generated[len(generated) - 1] == char) and (generated[len(generated) - 1] not in exclude_punct):
                if (char not in stop_punct):
                    generated = " " + generated
                    toprint += generated
                    length -= 1
                else:
                    toprint += generated
                    length -= 1
                break
      elif self.level == "character":
        toprint += generated
        length -= 1

    print(toprint)

  def prob_calculate(self, tokens):
    """
    Calculates probability of a certain tokenized sentence in current model, normalized for length.

    tokens: set of tokens to check against model from sentence
    return: the probability of these tokens
    """

    prob = 0
    for x in range(0, len(tokens) - self.order - 1):
      prompt = tuple(tokens[x:x + self.order])
      if prompt in self.transitions:
        next_token = tokens[x + self.order]
        values = self.transitions[prompt]
        prob += (values.count(next_token))/len(values)

    return prob

  def calculate_mean_stdev(self):
    """
    Calculates mean and standard deviation of model based on self.testing_set.
    """
    sentences = [self.tokens_from_string(x) + ['.']
                for x in self.testing_set.split(".")]
    probabilities = []
    for sentence in sentences:
      # skip short sentences
      if len(sentence) <= self.order:
          continue

      prob = self.prob_calculate(sentence)
      probabilities.append(prob / (len(sentence) - self.order))

    self.mean = statistics.mean(probabilities)
    self.stdev = statistics.stdev(probabilities)


  def estimate(self, text):
    """
    Calculates z-score for a sample text based on training_set models, for
    authorship attribution.

    text: Chunk of text expressed as a single string to determine z score.
    returns: Normalized z-score estimate of likelihood that text could have been
    produced by given model.
    """

    text_tokens = self.tokens_from_string(text)
    probZ = self.prob_calculate(text_tokens) / (len(text_tokens) - self.order)
    zscore = (probZ - self.mean) / self.stdev

    return zscore


def main():
    """Testing authorship attribution"""
    framework = Framework("corpora/arthur_conan_doyle_collected_works.txt", "word", 2, True) # Training model on donald_trump_collected_tweets

    """Grab testing sets from a couple different corpora for authorship attribution"""
    conan_doyle = open("corpora/arthur_conan_doyle_collected_works.txt", encoding="utf-8").read()
    conan_doyle = conan_doyle[int(len(conan_doyle)*0.80):]
    conan_zscore = 0

    dumas = open("corpora/alexander_dumas_collected_works.txt", encoding="utf-8").read()
    dumas = dumas[int(len(dumas)*0.80):]
    dumas_zscore = 0

    william_shakespeare = open("corpora/william_shakespeare_collected_works.txt", encoding="utf-8").read()
    william_shakespeare = william_shakespeare[int(len(william_shakespeare)*0.80):]
    shakes_zscore = 0

    """From each testing set, break it down into reasonably sized chunks and estimate, accumulate z score"""
    range_v = 4
    chunk_v= 3

    for x in range(1,range_v):
      conan_zscore += framework.estimate(conan_doyle[int((x-1) * len(conan_doyle)
                        / (1/chunk_v)): int(x * len(conan_doyle) / (1/chunk_v))])

      dumas_zscore += framework.estimate(dumas[int((x-1) * len(dumas)
                        / (1/chunk_v)): int(x * len(dumas) / (1/chunk_v))])

      shakes_zscore += framework.estimate(william_shakespeare[int((x-1) * len(william_shakespeare)
                        / (1/chunk_v)): int(x * len(william_shakespeare) / (1/chunk_v))])

    """Grab the average z-score across all test chunks and for each testing set, output z-score for each"""
    conan_zscore = conan_zscore / chunk_v

    dumas_zscore = dumas_zscore / chunk_v

    shakes_zscore = shakes_zscore / chunk_v

    print("Model: Arthur Conan Doyle\n")
    print(tabulate([['Arthur Conan Doyle', conan_zscore],
                    ['Alexander Dumas', dumas_zscore],
                    ['Shakespeare', shakes_zscore]], headers=['Chunks from...', 'Average z-score']))





    """Example of generating text on the word level on some model, starting from a random token"""
    ConanFrame = Framework("corpora/arthur_conan_doyle_collected_works.txt", "word", 1, True)

    ConanFrame.generate(200, ("A",))

if __name__ == '__main__':
    main()