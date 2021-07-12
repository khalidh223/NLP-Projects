"""
framework.py
Author: Khalid Hussain
New framework for training a Markov chain statistical language model from
user-supplied text corpora. The models are parameterizable
(word vs. character level, order of n-gram, and POS mode), and they can generate
text and perform authorship attribution.
This time, models can generate text & perform authorship attribution
based on tokenizing by the part-of-speech tag that correspond to each word
in a corpus. The tags are based on the Penn Treebank tagset.
"""
import nltk
import random
import statistics
import spacy
import string
from tabulate import tabulate
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')

class Framework:
  """
    corpus -- string of filename
    level -- either 'word' or 'character'; defines whether to train a character
    level or word-level language model
    order -- number of previous tokens to consider, eg. 2 for a bigram model
    to_estimate -- boolean for whether model is used to estimate z-score for some
    unknown text
    pos - boolean for whether to enable tokenization by part-of-speech.

    NOTE: Enabling pos requires tokenization by part-of-speech w/ spacy, and is
    really slow as a result.
    NOTE 2: The maximum length of the text allowed by spacy's tokenization is 1000000.
  """
  def __init__(self, corpus, level, order, to_estimate, pos):
    self.transitions = {}
    self.level = level
    self.order = order
    self.pos = pos
    self.tokens = []
    text = open(corpus, encoding="utf8").read()[:150000]
    if type(pos) != bool or type(to_estimate) != bool:
        print("Error: pos/to_estimate parameter is not a boolean.")
    if pos == True and level == "word":
      if to_estimate == True:
          self.testing_set = text[int(len(text)*0.8):]
          training_set = text[:int(len(text)*0.8)]
          with nlp.select_pipes(enable=["tok2vec", "tagger", "ner"]):
            self.tokens = nlp(training_set)
          self.train()
          self.calculate_mean_stdev()
      else:
          with nlp.select_pipes(enable=["tok2vec", "tagger", "ner"]):
            self.tokens = nlp(text)
          self.train()
    elif pos and level == "char":
      print("Error: Cannot tokenize by POS if training a character-level model")
    else:
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

    if self.pos == True:
        if self.level == "word":
          nlp = spacy.load("en_core_web_sm")
          with nlp.select_pipes(enable=["tok2vec"]):
            tagged_tokens = nlp(text)
          tokens_as_list = [token.text for token in tagged_tokens]
          return tokens_as_list
        else:
          print("Error: invalid level")
    elif self.pos == False:
        if self.level == "word":
          return nltk.word_tokenize(text)
        elif self.level == "character":
          return list(text)
        else:
          print("Error: invalid level")

  def train(self):
    """
    Trains a model on an associated corpus, filling transitions dictionary.
    """

    self.transitions = {}
    if self.pos:
        self.POSdict = {}

    if self.order > len(self.tokens) - 1:
        print("Unable to train: Hit upper bound on order, given corpus.")

    # create POSdict for use in generation
    if (self.pos == True):
        for x in self.tokens:
            tokentext = x.text
            if (not x.is_punct or x.is_left_punct):
                tokentext = " " + x.text
                if(x.text == "\n"):
                    x.tag_ = "NLN"
                elif(x.text == "\t"):
                    x.tag_ = "TAB"
                if (x.tag_,) in self.POSdict:
                    self.POSdict[(x.tag_,)].append(tokentext)
                else:
                    self.POSdict[(x.tag_,)] = [tokentext]
        self.tokens = [token.tag_ for token in self.tokens]

    # fill transitions dict from POS/non-POS tokens for training
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

    exclude_punct = "@#$%&^*\"’‘`'()[]{}"
    stop_punct = ".,:;!?" + "\n"
    toprint = ""

    if prompt == "" or len(prompt) < self.order:
      prompt = random.choice(list(self.transitions.keys()))
      if self.pos:
        toprint = " ".join(random.choice(self.POSdict[(x,)]) for x in prompt
                            if (x,) in self.POSdict and x not in stop_punct)
      else:
        toprint = " ".join(prompt)

    else:
      if self.pos:
        toprint = " ".join(prompt)
        with nlp.select_pipes(enable=["tok2vec", "tagger"]):
          tokens = nlp(toprint)
        prompt = []
        for i in range(self.order):
          prompt += (tokens[i].tag_,)
      else:
        toprint = " ".join(prompt)
        prompt = prompt[-self.order:]

    while length-self.order >= 0: # begin generating text
      if self.level == "word":
        if self.pos == True: # replace POS in transitions with text if in POS mode
          prompt = tuple(prompt)
          generated = random.choice(self.transitions[prompt])
          prompt = prompt[1:] + (generated,)
          if ((generated.strip(),) not in self.POSdict):
              if (generated.strip() in stop_punct):
                  toprint += generated.strip()
              continue
          pos_generated = random.choice(self.POSdict[(generated.strip(),)]).lstrip()
          for char in pos_generated:
              if char in exclude_punct:
                  break
              elif (pos_generated[len(pos_generated) - 1] == char) and (pos_generated[len(pos_generated) - 1] not in exclude_punct):
                  if (char not in stop_punct):
                      pos_generated = " " + pos_generated
                      toprint += pos_generated
                      length -= 1
                  else:
                      toprint += pos_generated
                      length -= 1
                  break
        elif self.pos == False:
            generated = random.choice(self.transitions[prompt])
            prompt = prompt[1:] + (generated,)
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
        generated = random.choice(self.transitions[prompt])
        toprint += generated
        length -= 1

    print(toprint)

  def prob_calculate(self, tokens):
    """ Returns probability normalized by length of a certain sentence in current model.

        tokens -- set of tokens to check against model
    """
    prob = 0
    for x in range(0, len(tokens) - self.order - 1):
      if self.pos == True:
          prompt = tuple(x[1] for x in nltk.tag.pos_tag(tokens[x:x + self.order]))
          if prompt in self.transitions:
            next_token = tokens[x + self.order]
            next_token = nltk.tag.pos_tag([next_token])[0][1]
            values = self.transitions[prompt]
            prob += (values.count(next_token))/len(values)
      else:
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
      if len(sentence) <= self.order:
          continue
      prob = self.prob_calculate(sentence)
      probabilities.append(prob / (len(sentence) - self.order))

    self.mean = statistics.mean(probabilities)
    self.stdev = statistics.stdev(probabilities)

  def estimate(self, text):
    """ Returns z-score for a sample text based on training_set models.

        text -- string to determine z score.
    """
    text_tokens = self.tokens_from_string(text)
    probZ = self.prob_calculate(text_tokens) / (len(text_tokens) - self.order)
    zscore = (probZ - self.mean) / self.stdev

    return zscore

  def get_named_entities(self):
    """
      Prints all named entities and their respective labels
    """
    for ent in self.tokens.ents:
      print(ent.text, ent.label_)
    print("\n\n")

def main():
    """Testing authorship attribution on text tokenized by POS"""
    framework = Framework("corpora/arthur_conan_doyle_collected_works.txt", "word", 2, True, True)

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





    """Example of non-POS mode generation of text, starting from user inputted token"""
    ConanFrame = Framework("corpora/arthur_conan_doyle_collected_works.txt", "word", 1, False, False)

    ConanFrame.generate(200, ("A",))

    """Example of POS mode generation of text, starting from user inputted token"""
    ConanFrame = Framework("corpora/arthur_conan_doyle_collected_works.txt", "word", 1, False, False)

    ConanFrame.generate(30, ("A",))

if __name__ == '__main__':
    main()
