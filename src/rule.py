from wnut import Normalizer
from splitter import Splitter
from filter import Filter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.externals import joblib

class Rule(object):
	def __init__(self):
		self.sentences = []
		self.abbreviation = {}
		self.load_data()
		self.load_abbrv()
		self.normalizer = Normalizer()
		self.splitter = Splitter()
		self.corrector = Filter()
		self.lemmatizer = WordNetLemmatizer()
		self.missing_apostrophe_vocab = ['isnt',' arent', 'wasnt', 'werent', 'wont','dont', 'didnt','doesnt','couldnt',
		                                 'shouldnt', 'hasnt', 'havent', 'hadnt']
		self.tokenizer_mistake_vocab = ['isn', 'aren', 'wasn', 'weren', 'won', 'don', 'didn', 'doesn', 'couldn', 'shouldn',
		                                'hasn', 'haven', 'hadn']
		self._norm = joblib.load('model.crfsuite')

	def load_data(self):
		with open('../data/reviews.txt', 'r') as myFile:
			self.sentences = myFile.read().splitlines()

	def load_abbrv(self):
		with open('../data/abbreviation.txt', 'r') as myFile:
			self.abbreviation = {i.split('---')[0]: i.split('---')[1] for i in myFile.read().splitlines()}

	def test2(self, sentence):
		tokens = self.splitter.split(sentence.lower())
		tokens = [j for i in tokens for j in i]
		print(tokens)
		X_test = self.normalizer.sent2features(tokens)
		for i in X_test:
			print(i)

		y_pred = self._norm.predict([X_test])
		print(y_pred)


	def process(self, sentence):
		tokens = self.splitter.split(sentence.lower())
		tokens = [j for i in tokens for j in i]
		X_test = self.normalizer.sent2features(tokens)

		y_pred = self._norm.predict([X_test])

		for i in range(len(y_pred[0])):
			if y_pred[0][i] == 'N':
				o, f = self.correct(tokens[i])
				if f:
					if tokens[i][0].isupper():
						tokens[i] = o[0].upper() + o[1:]
					else:
						tokens[i] = o
		return tokens

	def test(self):
		tokens = [self.splitter.split(i.lower()) for i in self.sentences[1001:5000]]
		tokens = [j for i in tokens for j in i]
		X_test = [self.normalizer.sent2features(i) for i in tokens]

		y_pred = self._norm.predict(X_test)

		stats = {}
		output = []
		for i in range(len(tokens)):
			flag = 0
			for j in range(len(y_pred[i])):
				if y_pred[i][j] == 'N':
					print(tokens[i][j])
					o, f = self.correct(tokens[i][j])
					if f:
						output.append((tokens[i][j], o))
					if tokens[i][j] not in stats:
						stats[tokens[i][j]] = 0
					stats[tokens[i][j]] += 1
					flag = 1
			if flag == 1:
				print(' '.join(tokens[i]))
				print(y_pred[i])

		with open('inter_correct_2.txt', 'w') as myFile:
			print('start writing')
			for old, new in output:
				print(old, new)
				myFile.write(old+'\t'+new+'\n')


	def correct(self, term):
		ret = []
		flag = False  #是否被修改了
		for i in term.split('.'):
			# 排除加长词语
			i = i.lower()
			i, res = self.correct_elongated(i)
			if res:
				flag = True
				ret.append(i)
				continue

			# tokenize时有时会把didn't分成didn和't
			if i in self.tokenizer_mistake_vocab:
				ret.append(i)
				continue

			# tokenize时会将"'"囊括其中比如't，不用管
			if "'" in i:
				ret.append(i)
				continue

			# 有些词是用'-'组合而成的，不检查
			if '-' in i:
				ret.append(i)
				continue

			# 解决助动词否定词缺少"'"
			if i in self.missing_apostrophe_vocab:
				i = self.correct_missing_apostrophe(i)
				flag = True
				ret.append(i)
				continue

			# 动词过去时已经被词典囊括，只检查名词单复数和动词的第三人称
			if i in self.normalizer.dct or self.lemmatizer.lemmatize(i, wordnet.NOUN) in self.normalizer.dct \
					or self.lemmatizer.lemmatize(i, wordnet.VERB) in self.normalizer.dct:
				ret.append(i)
				continue

			# 简称简写
			if i in self.abbreviation:
				i = self.abbreviation[i]
				flag = True
				ret.append(i)
				continue

			if i.isalpha():
				res_s = self.corrector.process(i)
				tmp = []
				for res in res_s.split(' '):
					if res != i and (self.lemmatizer.lemmatize(res, wordnet.NOUN) in self.normalizer.dct or self.lemmatizer.lemmatize(res, wordnet.VERB) in self.normalizer.dct
					or self.lemmatizer.lemmatize(res, wordnet.ADJ) in self.normalizer.dct or self.lemmatizer.lemmatize(res, wordnet.ADV) in self.normalizer.dct):
						flag = True
						tmp.append(res)
				if len(tmp) != 0:
					ret.append(' '.join(tmp))
				else:
					ret.append(i)
				continue

			ret.append(i)

		if len(ret) == 2:
			return ret[0] + '. ' + ret[1], flag
		else:
			return ret[0], flag

	def correct_merged_words(self, term):
		if len(term) < 4:
			return term, False
		for i in range(1, len(term)):
			if term[:i] in self.normalizer.dct and term[i:] in self.normalizer.dct:
				return term[:i] + ' ' + term[i:], True
		return term, False

	def correct_missing_apostrophe(self, term):
		res = term[:-1]+"'"+term[-1]
		return res

	def correct_elongated(self, term):
		count = 0
		while True:
			start, end, flag = self.find_elongated_pos(term)
			if count == 0 and not flag:
				return term, False
			elif not flag:
				return term, True
			else:
				cand1 = term.replace(term[start:end + 1], term[start])
				cand2 = term.replace(term[start:end + 1], term[start] * 2)
				# 如果candidate1 在词典里，结果就是candidate1
				if cand1 in self.normalizer.dct:
					term = cand1
					continue
				# 如果candidate2 在词典里，结果就是candidate2
				if cand2 in self.normalizer.dct:
					term = cand2
					continue
				# 如果都不在，选取candidate1
				term = cand1
			count += 1

	def find_elongated_pos(self, term):
		prev = ''
		start = 0
		ct = 1
		for idx, i in enumerate(term):
			if idx == 0:
				prev = i
				start = idx
			else:
				if i == prev:
					ct += 1
					if ct > 2:
						end = idx
						while end <= len(term)-1 and term[end] == i:
							end += 1
						end = end-1
						return start, end, True
				else:
					ct = 1
					start = idx
				prev = i
		return -1, -1, False

if __name__ == "__main__":
	r = Rule()
	print(r.correct('woks'))

