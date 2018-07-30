import nltk

class Splitter(object):
	def __init__(self):
		self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
		self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

	def split(self, text):
		sentences = self.splitter.tokenize(text)
		tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
		return tokens