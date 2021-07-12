
# About

Framework for training Markov chain statistical language models from user-supplied text
corpora. The models are parameterizable (word vs. character level,
order of n-gram), can generate text, and perform authorship attribution.

What is new, however, about this framework is its ability to train such a model based on
the frequency of part-of-speech (POS) tags that correspond to each word in the corpus. Doing so
may help with generating text, while avoiding the data sparsity problem that is apparent
in Markov chains and other NLP statistical models that are based on word frequency. 
This problem occurs within Markov chains when the size of the n-gram is sufficiently large, meaning 
there are very few possible successors at each decision point during the generation
procedure. Thus what often occurs is a reproduction of whole pieces of the text from the corpus that the model
had been trained on for sufficently large n-grams, as seen in [my other Markov Chain model,](https://github.com/khalidh223/NLP-Projects/tree/main/Markov%20Chain%20Model) 
instead of something new. 

Long n-grams tend to occur sparingly in text corpora since natural languages have
very large vocabularies, however in POS tags we can find relatively tiny vocabularies that make up
a tagset. Therefore, longer n-grams comprised of POS tags from a given tagset are likely to recur
in a tagged corpus. The tagset utilized by this framework is the [Penn Treebank tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html),
comprised of a whopping 36 parts of speech as opposed to the extremely large vocabulary of English words, for example.

One downside to this, of course, is time efficiency when trying to generate large snippits of texts from such a small vocabulary. Therefore, it is recommended to try
training the model on a smaller piece of a given corpus, or to reduce the order of the n-gram to a manageable size.
## Installation

Will need to use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary packages in bash:

```bash
pip install nltk
pip install spacy
pip install statistics
pip install tabulate
```
Can optionally install the pip wheels manually as well:

NLTK: https://pypi.org/project/nltk/ 

spaCy: https://pypi.org/project/spacy/

Statistics: https://pypi.org/project/statistics/

Tabulate: https://pypi.org/project/tabulate/

## Usage
Simply run ```python framework.py``` in bash. Modification of ```main()``` may be necessary to utilize custom corpora & create custom models. Any custom corpora can be placed in [/corpora](https://github.com/khalidh223/NLP-Projects/tree/main/Markov%20Chain%20Model%20with%20POS%20Tagging/corpora).
As aforementioned, tokenization by part-of-speech is slow for large orders of n, and the provided code partially remedies this by reading in a piece of a given corpus. In addition, the maximum length of text that the tokenization procedure done by spaCy allows is 1,000,000.
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
