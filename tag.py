import nltk
from nltk.tag.stanford import StanfordNERTagger

jar = './stanford-ner/stanford-ner.jar'
stanford = './stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'

stanford_ner = StanfordNERTagger(stanford, jar, encoding='utf8')

sentence = "Hello world"
words = nltk.word_tokenize(sentence)
tags = stanford_ner.tag(words)

print(tags)