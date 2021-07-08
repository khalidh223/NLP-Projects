
# About

Framework for training Markov chain statistical language models from user-supplied text
corpora. The models are parameterizable (word vs. character level,
order of n-gram), can generate text, and perform authorship attribution.

## Installation

Will need to use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary packages in bash:

```bash
pip install nltk
pip install statistics
pip install tabulate
```
Can optionally install the pip wheels manually as well:

NLTK: https://pypi.org/project/nltk/ 

Statistics: https://pypi.org/project/statistics/

Tabulate: https://pypi.org/project/tabulate/

## Usage
Simply run ```python framework.py``` in bash. Modification of ```main()``` may be necessary to utilize custom corpora & create custom models. Any custom corpora can be placed in [/corpora](https://github.com/khalidh223/NLP-Projects/tree/main/Markov%20Chain%20Model/corpora).
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
