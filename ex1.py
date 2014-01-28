# -*- coding: cp1252 -*-
import nltk
import re
import random
import unicodedata
from tweet_tools import *
from math import log, sqrt, exp
from unicodedata import normalize
from Tweets import Tweets
from pylab import plot, show
from subprocess import call
import MLPnumpy

values = []


def f(x):
    try:
        return 1 / (1 + exp(-x))
    except:
        return 0

class Ex1:
    def __init__(self, tweets_file=None):
        if tweets_file:
            text = open(tweets_file, "r").read().decode('utf-8', errors='replace')
            text = remove_accents(text)
        else:
            text = ""
        tweets = list(set([(line.split(" ")[0].upper(), " ".join(line.split(" ")[1:])) for line in text.split("\n") if line != ""]))
        self.tf = dict()
        self.docs_with_word = dict()
        self.words_list = list()
        self.tweets = list()
        print "Setting up Tweets"
        for t in tweets:
            self.add_tweet(t[1], t[0], init_mode=True)
        self.update_idf_factors()
        self.weights = {i: random.random() * 2 - 1 for i in self.words_list}
        self.weights["FACTOR"] = random.random() * 2 - 1
        self.set_all_tf_idf()
        self.sorted_wl = sorted(self.words_list)

    def get_terms(self):
        lines = []
        for i, j in enumerate(self.words_list):
            lines.append(str(i) + " " + j)
        return "\n".join(lines)

    def add_tweet(self, text, grade=None, init_mode=False):
        if not grade is None:
            grade = grade.upper()
        tweet = Tweet(text.lower(), grade)
        for w in get_words(text):
            if w in STOPWORDS:
                continue
            if w in self.words_list:
                self.docs_with_word[w] += 1
                self.tf[w] += 1
            else:
                self.words_list.append(w)
                self.docs_with_word[w] = 1
                self.tf[w] = 1
        tweet.set_tf()
        self.tweets.append(tweet)
        if not init_mode:
            self.get_tf_idf(tweet)
        return tweet

    def update_idf_factors(self):
        n_tweets = len(self.tweets)
        self.idf_factors = {i: log(n_tweets/float(self.docs_with_word[i])) for i in self.docs_with_word}

    def get_BoW(self, tweet):
        line = []
        for w in self.words_list:
            if w in tweet.tf.keys():
                line.append("1")
            else:
                line.append("0")
        return " ".join(line)

    def get_tf_idf(self, tweet):
        values = []
        for w in self.words_list:
            if w in tweet.tf.keys() and w in self.idf_factors:
                tweet.tf_idf[w] = tweet.tf[w]/tweet.get_mfw() * self.idf_factors[w]
                values.append(str(tweet.tf_idf[w]))
            else:
                values.append("0")
        return " ".join(values)

    def solve(self):
        f1 = open("file1.txt", "w")
        f1.write(self.get_terms())
        f1.close()
        BoW_lines = []
        tf_idf_lines = []
        for t in self.tweets:
            BoW_lines.append(self.get_BoW(t))
            tf_idf_lines.append(self.get_tf_idf(t))
        f2 = open("file2.txt", "w")
        f2.write("\n".join(BoW_lines))
        f2.close()
        f3 = open("file3.txt", "w")
        f3.write("\n".join(tf_idf_lines))
        f3.close()

    def tweets_cos(self, t1, t2):
        internal_product = 0
        mT1, mT2 = 0, 0
        if(len(t1.tf_idf) == 0):
            self.get_tf_idf(t1)
        if(len(t2.tf_idf) == 0):
            self.get_tf_idf(t2)
        for w in self.words_list:
            in_both = True
            if w in t1.tf_idf.keys():
                mT1 += t1.tf_idf[w] ** 2
            else:
                in_both = in_both and False
            if w in t2.tf_idf.keys():
                mT2 += t2.tf_idf[w] ** 2
            else:
                in_both = in_both and False
            if in_both:
                internal_product += t1.tf_idf[w] * t2.tf_idf[w]
        mT1, mT2 = sqrt(mT1), sqrt(mT2)
        if (mT1 * mT2) == 0:
            return -10
        return internal_product / (mT1 * mT2)

    def get_examples(self, n):
        size = len(self.tweets)
        for i in range(n):
            index = random.randrange(size)
            print self.tweets[index].text
            self.tweets[index].grade = float(raw_input("Defina uma nota:"))

    def set_all_tf_idf(self):
        print "Setting up TF-IDF"
        for t in self.tweets:
            if(len(t.tf_idf) == 1):
                self.get_tf_idf(t)

    def logistic_regression(self, alpha, de):
        delta_error = de + 1
        last_error = None
        tweetsQuant = len(self.tweets)
        self.set_all_tf_idf()
        iteration = 0
        size = float(len(self.words_list))
        while last_error is None or delta_error > de:
            dw = {i: 0 for i in self.weights}
            dfactor = 0
            mean_error = 0
            for t in self.tweets:
                if t.grade is None:
                    continue
                error = 0.0
                for word in t.tf_idf:
                    error += t.grade - f(self.weights[word] * t.tf_idf[word])
                error /= size
                mean_error += error
                for word in t.tf_idf:
                    dw[word] += error * t.tf_idf[word]
            for word in dw:
                self.weights[word] += alpha * dw[word]
            mean_error = abs(mean_error / tweetsQuant)
            values.append(mean_error)

            if not last_error is None:

                delta_error = abs(mean_error - last_error)
            print "ERROR:", mean_error, " ", delta_error, "ITERATION:", iteration
            last_error = mean_error
            iteration += 1

    def write_train_set(self, input_set, output_set):
        filename = "train.txt"
        f = open(filename, "w")
        ni, nh, no = len(input_set[0]), 100, len(output_set[0])
        trainSteps = 10000
        examples = len(input_set)
        desired_error = 0.001
        print ni, nh, no
        f.write("%i %i %i %i %i %f\n" % (ni, nh, no, examples, trainSteps, desired_error))
        for inp, out in zip(input_set, output_set):
            for i in inp:
                f.write(str(i) + " ")
            f.write("\n")
            for o in out:
                f.write(str(o) + " ")
            f.write("\n")

    def run_train(self):
        call(["./nn.bin"])

    def get_weights(self):
        f = open("r.txt", "r")
        text = f.read()
        f.close()
        wi_string, wh_string = text.split('\n')[:2]
        self.wi = eval(wi_string)
        self.wh = eval(wh_string)

    def train_neural_network(self, nh, alpha):
        self.set_all_tf_idf()
        ni = len(self.words_list)
        input_set = [[0] * len(self.words_list) for t in self.tweets]
        output_set = [[t.grade] for t in self.tweets]
        for i, t in enumerate(self.tweets):
            for w in t.tf_idf:
                if w in self.sorted_wl:
                    input_set[i][self.sorted_wl.index(w)] = t.tf_idf[w]
        self.write_train_set(input_set, output_set)
        self.run_train()
        self.get_weights()

    def get_logistic(self, text):
        val = 0
        text = remove_accents(text.decode('cp1252'))
        tweet = self.add_tweet(text)
        for word in tweet.tf_idf:
            if word in self.weights:
                val += tweet.tf_idf[word] * self.weights[word]
            #else:
            #    val += tweet.tf_idf[word] * 0.5
        return f(val)

    def get_nn(self, text):
        text = remove_accents(text.decode('cp1252'))
        tweet = self.add_tweet(text)
        tweet_input = [0] * len(self.sorted_wl)
        for w in tweet.tf_idf:
            if w in self.sorted_wl:
                tweet_input[self.sorted_wl.index(w)] = tweet.tf_idf[w]
        return MLPnumpy.runmlp(tweet_input, self.wi, self.wh)

    def save_training(self, filename):
        train_file = open(filename, "w")
        for w in self.weights:
            train_file.write(w.encode('cp1252'))
            train_file.write(":")
            train_file.write("%.40f" % self.weights[w])
            train_file.write(":")
            if w in self.idf_factors:
                train_file.write("%.40f\n" % self.idf_factors[w])
            else:
                train_file.write("0\n")
        train_file.close()

    def load_training(self, filename, pre_loaded=False):
        train_file = open(filename, "r")
        for l in train_file.readlines():
            parsed_l = l.split(":")
            if len(parsed_l) == 3 or not pre_loaded:
                word, weight, idf_factor = parsed_l
                self.idf_factors[word] = float(idf_factor)
            elif len(parsed_l) == 2:
                word, weight = parsed_l[:2]
            self.weights[word] = float(weight)
        self.words_list = self.weights.keys()


def train_regression():
    ex1 = Ex1("./train_pol_corrigido.txt")
    ex1.logistic_regression(1, 1e-12)
    ex1.save_training("train2.lrt")
    capture = ""
    while capture != "quit":
        capture = remove_accents(raw_input("Digite o texto:").decode('cp1252'))
        t = ex1.add_tweet(capture)
        print "Sua nota é %f\n " % ex1.get_logistic(t)


def solve_regression_file():
    ex1 = Ex1("./train_pol.txt")
    ex1.load_training("train.lrt", pre_loaded=True)
    capture = ""
    while capture != "quit":
        capture = remove_accents(raw_input("Digite o texto:").decode('cp1252'))
        t = ex1.add_tweet(capture)
        print "Sua nota é %f\n " % ex1.get_logistic(t)


def train_nn():
    ex1 = Ex1("./train_pol_corrigido.txt")
    ex1.train_neural_network(10, 0.5)
    capture = ""
    while capture != "quit":
        capture = remove_accents(raw_input("Digite o texto:").decode('cp1252'))
        t = ex1.add_tweet(capture)
        print ex1.get_nn(t)


def run_nn():
    ex1 = Ex1("./train_pol_corrigido.txt")
    ex1.get_weights()
    capture = ""
    while capture != "quit":
        print ex1.get_nn(raw_input("Digite o texto:"))

if __name__ == "__main__":
    #train_regression()
    train_nn()
    #run_nn()
    #plot([log(i) for i in values])
    #show()
