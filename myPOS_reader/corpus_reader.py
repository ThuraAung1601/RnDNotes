# Convert myPOS word/tag format to [(word,tag)] format
# Author: Thura Aung @ LU Laboratory, Myanmar
# Date: 11 Aug 2023
# How to run: corpus_read.py <word/tag-file> >> sequence list

import sys

f = open(sys.argv[1], 'r')
def corpus_reader(f):
	lines = f.readlines()
	for line in lines:
		parsed_sentences = []
		for line in line.split('\n'):
			if line:
				word_tag_pairs = line.split()
				sentence = [(pair.split('/')[0], pair.split('/')[1]) for pair in word_tag_pairs]
				parsed_sentences.append(sentence)
	return parsed_sentences

print(corpus_reader(f))
