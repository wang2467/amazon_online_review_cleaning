from sklearn.model_selection import KFold
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
from nltk.tokenize import TreebankWordTokenizer
import os
import json
import re
import nltk
import sklearn_crfsuite

class Normalizer(object):
	def __init__(self):
		self.sentences = []
		self.tes_sentences = []
		self.load()
		self.load_test()
		self.load_dict()

	def cross_validate(self):
		kfold = KFold(n_splits=3)
		for train_ids, test_ids in kfold.split(self.sentences):
			X_train = [self.sent2features(self.sentences[i][0]) for i in train_ids]
			y_train = [self.sent2labels(self.sentences[i][0], self.sentences[i][1]) for i in train_ids]

			crf = sklearn_crfsuite.CRF(
				algorithm='lbfgs',
				c1=0.1,
				c2=0.2,
				max_iterations=100,
				all_possible_transitions=True
			)
			crf.fit(X_train, y_train)

			labels = list(crf.classes_)

			X_test = [self.sent2features(self.sentences[i][0]) for i in test_ids]
			y_test = [self.sent2labels(self.sentences[i][0], self.sentences[i][1]) for i in test_ids]
			y_pred = crf.predict(X_test)

			for idx, id in enumerate(test_ids):
				print(self.sentences[id][0])
				print(self.sentences[id][1])
				print(y_pred[idx])
				print(y_test[idx])

			# print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
			print(metrics.flat_accuracy_score(y_test, y_pred))


	def process(self):
		if not os.path.isfile('model.crfsuite'):
			X_train = [self.sent2features(self.sentences[i][0]) for i in range(len(self.sentences))]
			y_train = [self.sent2labels(self.sentences[i][0], self.sentences[i][1]) for i in range(len(self.sentences))]

			crf = sklearn_crfsuite.CRF(
				algorithm='lbfgs',
				c1=0.1,
				c2=0.2,
				max_iterations=100,
				all_possible_transitions=True
			)
			crf.fit(X_train, y_train)
			joblib.dump(crf, 'model.crfsuite', compress=1)
		else:
			crf = joblib.load('model.crfsuite')

		# labels = crf.classes_
		# X_test = [self.sent2features(x) for x, y in self.tes_sentences]
		# y_test = [self.sent2labels(x, y) for x, y in self.tes_sentences]
		#
		# y_pred = crf.predict(X_test)
		# print(len(y_pred) == len(y_test))
		#
		# for i in range(len(self.tes_sentences)):
		# 	print(self.tes_sentences[i][0])
		# 	print(self.tes_sentences[i][1])
		# 	print(y_pred[i])
		# 	print(y_test[i])
		# 	for j in range(len(y_pred[i])):
		# 		if y_pred[i][j] == 'N':
		# 			print(self.tes_sentences[i][0][j])
		#
		# print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))

		string = 'lol I lov this songgg however this is my favorite'
		a = self.sent2features(string.split(' '))
		print(a)
		print(len(a))
		print(crf.predict([a]))

	def sent2labels(self, s1, s2):
		return ['N' if i.lower() != s2[idx].lower() else 'G' for idx, i in enumerate(s1)]

	def sent2features(self, tokens):
		t_p = nltk.pos_tag(tokens)
		res = []
		for idx, (token, pos) in enumerate(t_p):
			token_feature = {}
			self.prefix_suffix(tokens, idx, token_feature)
			self.pos_tag(t_p, idx, token_feature)
			self.oov(token, token_feature)
			self.word_length(token, token_feature)
			self.isdigit(token, token_feature)
			# self.isalnum(token, token_feature)
			self.isconsecutive(token, token_feature)
			self.iscompact(token, token_feature)
			self.issingle(token, token_feature)
			self.startswithHash(token,token_feature)
			res.append(token_feature)
		return res

	def word_length(self, token, feature):
		feature.update({
			'word>4': 1 if len(token) >= 4 else 0
		})

	def oov(self, token, feature):
		feature.update({
			'OOV': not token in self.dct
		})

	def isdigit(self, token, feature):
		feature.update({
			'word.isdigit': token.isdigit()
		})

	def isalnum(self, token, feature):
		feature.update({
			'word.isalnum': token.isalnum()
		})

	def isconsecutive(self, token, feature):
		ct = 0
		res = False
		prev = ''
		for idx, i in enumerate(token):
			if idx == 0:
				prev = i
			else:
				curr = i
				if curr == prev:
					ct += 1
				else:
					ct = 0
				if ct > 2:
					res = True
					break
				prev = curr

		feature.update({
			'word.consecutive': res
		})

	def iscompact(self, token, feature):
		feature.update({
			'word.iscompact': "'" in token
		})

	def issingle(self, token, feature):
		feature.update({
			'word.issingle': len(token) == 1 and token not in set('Ia')
		})

	def startswithHash(self, token, feature):
		feature.update({
			'word.startswithHash': token[0] in set('#@&')
		})

	def pos_tag(self, t_p, idx, feature):
		self._tag(t_p[idx][1], feature, '')

		if idx < len(t_p) - 1:
			self._tag(t_p[idx+1][1], feature, '+1:')

		if idx > 0:
			self._tag(t_p[idx-1][1], feature, '-1:')

	def prefix_suffix(self, tokens, idx, feature):
		self._ps(tokens[idx], feature, '')

		if idx < len(tokens)-1:
			self._ps(tokens[idx+1], feature, '+1:')

		if idx > 0:
			self._ps(tokens[idx-1], feature, '-1:')

	def _ps(self, token, feature, offset):
		feature.update({
			offset + 'word[-3:]': token[-3:],
			offset + 'word[-2:]': token[-2:],
			offset + 'word[:2]': token[:2],
			offset + 'word[:3]': token[:3]
		})

	def _tag(self, pos, feature, offset):
		feature.update({
			offset + 'postag': pos
		})

	def load(self):
		pattern = r'({.+?})[,|\]]'
		with open('../data/train_data.json', 'r') as myFile:
			lines = myFile.read().splitlines()
		for m in re.finditer(pattern, lines[0]):
			j = json.loads(m.group(1))
			self.sentences.append([j['input'], j['output']])

	def load_test(self):
		pattern = r'({.+?})[,|\]]'
		with open('../data/test_truth.json', 'r') as myFile:
			lines = myFile.read().splitlines()
		for m in re.finditer(pattern, lines[0]):
			j = json.loads(m.group(1))
			self.tes_sentences.append([j['input'], j['output']])


	def load_dict(self):
		with open('../data/dict.txt', 'r') as myFile:
			self.dct = myFile.read().splitlines()

		with open('../data/cellphone_term.txt', 'r') as myFile:
			self.dct.extend(myFile.read().splitlines())

		self.dct = set(self.dct)

if __name__ == "__main__":
	n = Normalizer()
	n.process()