import json

if __name__ == "__main__":
	with open('../data/reviews_Cell_Phones_and_Accessories_5.json', 'r') as myFile:
		contents = myFile.read().splitlines()
	with open('../data/reviews.txt', 'w') as myFile:
		for i in contents:
			dct = json.loads(i)
			myFile.write(dct['reviewText']+'\n')

