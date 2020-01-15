import nltk
from nltk.tag.stanford import StanfordNERTagger

#java -cp "stanford-ner.jar:lib/*" -mx12g edu.stanford.nlp.ie.crf.CRFClassifier -prop ../my-little-ner/prop.txt

#STANFORD TAGGING
jar = './stanford-ner/stanford-ner.jar'
stanford = './stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'

stanford_ner = StanfordNERTagger(stanford, jar, encoding='utf8')

sentence = "Hello world"
words = nltk.word_tokenize(sentence)
tags = stanford_ner.tag(words)

print(tags)

#CUSTOM TAGGING
#ner = './my-little-ner/classifiers/ner-model.ser.gz'
#my_ner = StanfordNERTagger(ner, jar, encoding='utf8')
#my_tags = my_ner.tag(words)
#print(my_tags)