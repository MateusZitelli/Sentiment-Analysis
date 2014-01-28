from nb import Naive_bayes
from ex1 import Ex1
from tweet_tools import remove_accents

print "Training Naive Bayes"
nb = Naive_bayes("train_pol_corrigido.txt")#,"train.results")
print "Training NN"
ex1 = Ex1("./train_pol_corrigido.txt")
ex1.train_nn()
print "Doing the regression"
ex1.train_regression()

#print "Loading weights"
#ex1.get_weights()
#print "Loading logistic"
#ex1.load_training("train2.lrt", pre_loaded=True)

capture = ""
while capture != "quit":
    capture = remove_accents(raw_input("Digite o texto:").decode('utf-8'))
    r = nb.get(capture)
    print "Naive Bayes:", r
    result = [0,0,0]
    result[0] =  r["NEGATIVE"] - r["POSITIVE"]
    if result[0] < 0:
        result[0] = 1
    else:
        result[0] = 0
    result[1] = ex1.get_nn(capture)[1][0]
    result[2] = ex1.get_logistic(capture)
    print "Neural network:", result[1]
    print "Regression:", result[2]
    positives = 0
    negatives = 0
    for i in result:
        if i > 0.5:
            positives += 1
        else:
            negatives += 1
    if positives > negatives:
        print "A frase e positiva"
    else:
        print "A frase e negativa"
