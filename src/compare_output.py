if __name__ == "__main__":
	with open('inter_correct_2.txt','r') as myFile:
		r1 = myFile.read().splitlines()

	with open('inter_correct.txt', 'r') as myFile:
		r2 = myFile.read().splitlines()

	for i in set(r2) - set(r1):
		print(i)