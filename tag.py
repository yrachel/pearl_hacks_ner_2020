import nltk                                                         #allows you to use methods from the nltk library (ex. word_tokenize() below)
from nltk.tag.stanford import StanfordNERTagger                     #allows you to use StanfordNERTagger

#java -cp "stanford-ner.jar:lib/*" -mx12g edu.stanford.nlp.ie.crf.CRFClassifier -prop ../my-little-ner/prop.txt

#STANFORD TAGGING
jar = './stanford-ner/stanford-ner.jar'                             #a jar file is a collection of java classes packaged for easy use
stanford = './stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'   #the encrypted Stanford NER model

stanford_ner = StanfordNERTagger(stanford, jar, encoding='utf8')    #loads the Stanford NER model into a usable format

sentence = "Hello world"                                            #feel free to replace with a sentence of your choice!
words = nltk.word_tokenize(sentence)                                #splits sentence into a list of words with tags
tags = stanford_ner.tag(words)                                      #asks the stanford_ner model to give the tokenized sentence NER tags

print(tags)                                                         #see what you get!

#CUSTOM TAGGING
#ner = './my-little-ner/classifiers/ner-model.ser.gz'               #your custom NER model, encrypted
#my_ner = StanfordNERTagger(ner, jar, encoding='utf8')              #loads your custom model
#my_tags = my_ner.tag(words)                                        #asks your model to give the above tokenized sentence NER tags
#print(my_tags)                                                     #:0