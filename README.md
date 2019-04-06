# Sentence-Completion-using-Hidden-Markov-Models
The goal of this script is to implement three langauge models to perform sentence completion, i.e. given a sentence with a missing word to choose the correct one from a list of candidate words. The way to use a language model for this problem is to consider a possible candidate word for the sentence at a time and then ask the language model which version of the sentence is the most probable one.

The sentences to be completed together with the candidate words are in this file: questions.txt. The word to be completed is denoted with ‘____’ while the pair of candidate words is at the end of the line (e.g. weather/whether). The character ‘:’ between the sentence and the candidates is not part of the sentece. To apply a language model on a sentence for a given candidate word, the script replaces the ‘____’ with the candidate word.

The texts to train your language models are in this file: news-corpus-500k.txt (70MB), which is a small subset of the 1 Billion Word Benchmark.

# Running the script
To run the script use: python3 lm.py news-corpus-500k.txt questions.txt

If you want to train your model on some other corpus insted of news-corpus-500k, just replace the 2nd argument with path to your own corpus, also, of you want to test your model on some different set of sentences, just replace the 3rd arugment with the path to your sentences. Keep in mind to use the same pattern for the custom sentences which you want to test your model on.

