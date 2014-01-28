# -*- coding: cp1252 -*-
from tweet_tools import *
from math import log

class Naive_bayes:
    def __init__(self, tweets_file = None, trained_data = None):
        if tweets_file != None:
            self.tweets = read_train_data(tweets_file)
        self.words = set()
        for t in self.tweets:
            t.set_wordslist()
            for w in t.wordslist:
                self.words.add(w)
        self.tweets_quant = float(len(self.tweets))
        self.probabilities = {}
        if trained_data == None:
            positive_quant = len([t for t in self.tweets if t.grade == 1]) + 1
            negative_quant = len([t for t in self.tweets if t.grade == 0]) + 1
            neutral_quant = len([t for t in self.tweets if t.grade == 0.5]) + 1
            self.probabilities["POSITIVE"] = positive_quant/self.tweets_quant
            self.probabilities["NEGATIVE"] = negative_quant/self.tweets_quant
            self.probabilities["NEUTRAL"] = neutral_quant/self.tweets_quant
            print "Generating probabilities data"
            words_quant = float(len(self.words))
            for i,w in enumerate(self.words):
                print "%f" % (i / words_quant * 100)
                tweets_with_w_positives = len([t for t in self.tweets if w in t.wordslist and t.grade == 1]) + 1
                tweets_with_w_negatives = len([t for t in self.tweets if w in t.wordslist and t.grade == 0]) + 1
                tweets_with_w_neutral = len([t for t in self.tweets if w in t.wordslist and t.grade == 0.5]) + 1
                total = float(tweets_with_w_positives + tweets_with_w_negatives + tweets_with_w_neutral)
                if total == 0:
                    continue
                self.probabilities[(w, "POSITIVE")] = tweets_with_w_positives / total
                self.probabilities[(w, "NEGATIVE")] = tweets_with_w_negatives / total
                self.probabilities[(w, "NEUTRAL")] = tweets_with_w_neutral / total
        else:
            self.load_train(trained_data)

    def get(self, text):
        text = remove_accents(text.lower().encode('cp1252'))
        words = get_words(text)
        sum_positive_args = 0
        sum_negative_args = 0
        sum_neutral_args = 0
        for w in words:
            if not w in self.words:
                continue
            sum_positive_args += log(self.probabilities[(w, "POSITIVE")])
            sum_negative_args += log(self.probabilities[(w, "NEGATIVE")])
            sum_neutral_args += log(self.probabilities[(w, "NEUTRAL")])
        log_probability = {}
        log_probability["POSITIVE"] = sum_positive_args + log(self.probabilities["POSITIVE"])
        log_probability["NEGATIVE"] = sum_negative_args + log(self.probabilities["NEGATIVE"])
        log_probability["NEUTRAL"] = sum_negative_args + log(self.probabilities["NEUTRAL"])
        return log_probability

    def save_training(self, filename):
        f = open(filename, "w")
        for key in self.probabilities:
            if type(key) == tuple:
                f.write(key[0])
                f.write(",")
                f.write(key[1])
            elif type(key) == str:
                f.write(key)
            f.write(";")
            f.write("%.30f\n" % (self.probabilities[key]))
        f.close()

    def load_train(self, filename):
        f = open(filename, "r")
        for l in f.readlines():
            key, prob = l.split(";")
            splited_key = key.split(",")
            if len(key.split(",")) == 2:
                key = tuple(splited_key)
            prob = float(prob)
            self.probabilities[key] = prob
if __name__ == "__main__":
    nb = Naive_bayes("train_pol_corrigido.txt")
    nb.save_training("train.results")
    capture = ""
    while capture != "quit":
        capture = remove_accents(raw_input("Digite o texto:").decode('utf-8', errors='replace'))
        nb.get(capture)
