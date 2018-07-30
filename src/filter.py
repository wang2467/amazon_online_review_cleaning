from jellyfish import soundex, levenshtein_distance, metaphone
from autocorrect import spell
from enchant import Dict
import time

class Filter(object):
	def __init__(self):
		self.generator = Dict('en_US')

	def process(self, term):
		if term == '':
			return term
		candidates = self.generate_candidates(term)
		if candidates:
				scores = [(0.6 * levenshtein_distance(metaphone(i), metaphone(term)) + 0.4 * levenshtein_distance(i, term), idx) for idx, i in enumerate(candidates)]
				min_value = 1000
				min_idx = -1
				if len(scores) > 0:
					for score, idx in scores:
						if candidates[idx].startswith(term[0]) and "'" not in candidates[idx]:
							if score < min_value:
								min_value = score
								min_idx = idx
					term = candidates[min_idx]
		return term

	def generate_candidates(self, term):
		if not self.generator.check(term):
			return self.generator.suggest(term)
		return None


if __name__ == '__main__':
	f = Filter()
	words = ['talkabout', 'alot','jus','batterylife','problema', 'woks']
	for i in words:
		print(f.process(i))