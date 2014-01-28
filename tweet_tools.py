# -*- coding: cp1252 -*-
import re
import nltk
import unicodedata

class Tweet:
    def __init__(self, text, grade = None):
        self.tf = {}
        self.tf_idf = {"FACTOR":1}  
        self.text = text
        self.most_freq_word_frequence = 0
        if type(grade) == type(u"unicode"):
            grade = grade.upper()
            if grade == "POSITIVO":
                self.grade = 1
            elif grade == "NEGATIVO":
                self.grade = 0
            elif grade == "NEUTRO":
                self.grade = 0.5
        else:
            self.grade = grade

    def get_mfw(self):
        if(self.most_freq_word_frequence == 0):
            self.most_freq_word_frequence = max(self.tf.values())
        return float(self.most_freq_word_frequence)

    def set_tf(self):
        for w in get_words(self.text):
            if w in self.tf:
                self.tf[w] += 1
            else:
                self.tf[w] = 1

    def set_wordslist(self):
        self.wordslist = get_words(self.text)

def get_words(text):
    return re.split(r"\W+", text.lower())

def remove_accents(input_str):
    nkfd_form = unicodedata.normalize('NFKD', unicode(input_str))
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def read_train_data(train_file):
    f = open(train_file, "r")
    tweets = list()
    tweets_parsed = list()
    for l in f.readlines():
        l = l.decode('utf-8', errors="replace")
        tweet_match = re.match(r'(positivo|negativo|neutro)\s+(.*)', l)
        if tweet_match:
            grade = tweet_match.group(1).upper()
            text = remove_accents(tweet_match.group(2).lower())
            tweets_parsed.append((grade, text))
    #remove repeted tweets
    tweets_parsed = list(set(tweets_parsed))
    for t in tweets_parsed:
        tweets.append(Tweet(t[1], t[0]))
    return tweets

STOPWORDS = [remove_accents(unicode(w)) for w in open("stopwords.txt", "r").read().decode('cp1252').split("\n")]
