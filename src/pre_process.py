from nltk.stem import WordNetLemmatizer
from enchant.checker import SpellChecker
from autocorrect import spell
from nltk.corpus import wordnet
from rule import Rule
import re
import nltk

class Preprocesser(object):
	def __init__(self, in_fname, out_fname):
		self.out_fname = out_fname
		self.load(in_fname)
		self._spell_checker = SpellChecker('en_US')
		self._wordnet = WordNetLemmatizer()
		self.rule_corrector = Rule()

	def load(self, fname):
		with open(fname, 'r') as myFile:
			self.raw_data = myFile.read().splitlines()
			self.old_data = self.raw_data.copy()

	def process(self):
		for line, data in enumerate(self.raw_data[:100]):
			token_res = self.rule_corrector.process(data)

			# 生成词性
			pos_res = nltk.pos_tag(token_res)

			# 词形还原
			for idx, (word, pos) in enumerate(pos_res):
				if word.isalpha():
					# 句首词语大写改小写
					if idx == 0:
						word = word[0].lower() + word[1:]
						lemmatized_word = self.lemmatize(word, self.get_wordnet_pos(pos))
						token_res[idx] = lemmatized_word[0].upper() + lemmatized_word[1:]
					# 动词大写改小写
					elif pos == wordnet.VERB and word[0].isupper():
						word = word.lower()
						lemmatized_word = self.lemmatize(word, self.get_wordnet_pos(pos))
						token_res[idx] = lemmatized_word[0].upper() + lemmatized_word[1:]
					else:
						lemmatized_word = self.lemmatize(word, self.get_wordnet_pos(pos))
						token_res[idx] = lemmatized_word

			self.raw_data[line] = ' '.join(token_res)

		with open(self.out_fname, 'w') as myFile:
			for i in self.raw_data:
				myFile.write(i + '\n')

	def lemmatize(self, word, pos):
		tokenized_word = self._wordnet.lemmatize(word, pos)
		if pos == wordnet.VERB:
			if tokenized_word != word and (word.endswith('ed') or word.endswith('ing') or word.endswith('t') or tokenized_word == 'be'):
				return word
			else:
				return tokenized_word
		else:
			return tokenized_word


	def get_wordnet_pos(self, pos):
		if pos.startswith('J'):
			return wordnet.ADJ
		if pos.startswith('N'):
			return wordnet.NOUN
		if pos.startswith('V'):
			return wordnet.VERB
		if pos.startswith('R'):
			return wordnet.ADV
		return wordnet.NOUN

	def edit_distance(self, word1, word2):
		memo = {}
		return self.edit_distance_helper(word1, word2, len(word1)-1, len(word2)-1, memo)

	def edit_distance_helper(self, word1, word2, s1, s2, memo):
		if s1 < 0:
			return s2+1
		if s2 < 0:
			return s1+1
		if word1[s1] == word2[s2]:
			cost = 0
		else:
			cost = 2
		key1 = str(s1-1)+'---'+str(s2)
		key2 = str(s1)+'---'+str(s2-1)
		key3 = str(s1-1)+'---'+str(s2-1)
		if key1 not in memo:
			memo[key1] = self.edit_distance_helper(word1, word2, s1-1, s2, memo)
		if key2 not in memo:
			memo[key2] = self.edit_distance_helper(word1, word2, s1, s2 - 1, memo)
		if key3 not in memo:
			memo[key3] = self.edit_distance_helper(word1, word2, s1 - 1, s2 - 1, memo)
		return min(memo[key1]+1, memo[key2]+1, memo[key3]+cost)

if __name__ == "__main__":
	p = Preprocesser('../data/reviews.txt', 'reviews_output.txt')
	p.process()
