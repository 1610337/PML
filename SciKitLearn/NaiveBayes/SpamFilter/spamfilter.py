import os
import re
from collections import Counter
import numpy as np

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

blacklist = ["123@hotmail.de"]
whitelist = ["professoren-bounces2@ml.hs-mannheim.de"]

words_to_remove = ["e", "und"]
top_words_to_account = 50

def main():

    path_to_spams = os.path.dirname(os.path.abspath(__file__)) + "/dir.spam/"
    path_to_nospams = os.path.dirname(os.path.abspath(__file__)) + "/dir.nospam/"
    path_to_inputs = os.path.dirname(os.path.abspath(__file__)) + "/dir.mail.input/"

    spams = read_emails(path_to_spams)
    nospams = read_emails(path_to_nospams)
    input = read_emails(path_to_inputs)

    for mail in input:
        print("Is this mail a spam: ", mail["Betreff"], mail["Von"])

        if blacklist_filter(mail["Von"]):
            print("Yes --- blacklist")
            continue
        if whitelist_filter(mail["Von"]):
            print("No --- whitelist")
            continue
        print("Bayes Result: ", bayes_spam_filter(spams, nospams, input))
        break


def bayes_spam_filter(spam, nospam, inputFile):

    dictionary = get_word_dict(spam)

    retures_matrix = extract_features(spam, dictionary)

    # Prepare feature vectors per training mail and its labels

    train_labels = np.zeros(18)
    train_labels[7:17] = 1
    train_matrix = extract_features(spam, dictionary)

    # Training Naive bayes classifier

    model1 = MultinomialNB()
    model1.fit(train_matrix, train_labels)

    # Test the unseen mails for Spam
    test_dir = 'test-mails'
    test_matrix = extract_features(inputFile, dictionary)
    test_labels = np.zeros(1)
    test_labels[1:1] = 1
    result1 = model1.predict(test_matrix)
    print(confusion_matrix(test_labels, result1))

    return confusion_matrix(test_labels, result1)


def extract_features(files, dictionary_spams):

    # files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
        for i,line in enumerate(fil["Text"]):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary_spams):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix


def get_word_dict(emails):
    all_words = []
    for mail in emails:
            for i, line in enumerate(mail["Text"]):
                # if i == 2:  # Body of email is only 3rd line of text file
                    words = line.split()
                    # print(line.split())
                    all_words += words

    dictionary = Counter(all_words)
    # Paste code for non-word removal here(code snippet is given below)

    list_to_remove = list(dictionary.keys())
    list_to_remove = list_to_remove

    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    for item in words_to_remove:
        if item in list(dictionary.keys()):
            del dictionary[item]

    dictionary = dictionary.most_common(top_words_to_account)

    return dictionary


def blacklist_filter(addresse):
    if addresse in blacklist:
        return True
    return False

def whitelist_filter(addresse):
    if addresse in whitelist:
        return True
    return False


def read_emails(path):

    # TODO check if the file is actually a txt file

    mails = []
    for filename in os.listdir(path):
        finalDic = {}
        with open(path+filename, 'r') as text_file:
            header = []
            normalText = []
            noHeader = False
            for idx, line in enumerate(text_file):
                # The first empty line marks the end of the header
                if line == "\n" and not(noHeader):
                    finalDic.update(get_header_dic(header))
                    noHeader = True

                if noHeader and line.strip() != "":
                    normalText.append(line.strip())
                else:
                    header.append(line)

            finalDic["Text"] = normalText
        mails.append(finalDic)


    return mails


def get_header_dic(lines):
    returnDic = {}
    for idx, line in enumerate(lines):
        if str(line).startswith("Von:"):
            returnDic["Von"] = re.search('<(.*)>', str(line))[0][1:-1]
            # print(returnDic["Von"])
        if str(line).startswith("An:"):
            returnDic["An"] = line[3:].strip()
            # print(returnDic["An"])
        if str(line).startswith("Gesendet:"):
            returnDic["Datum"] = line.replace("Gesendet:", "").strip()
            # print(returnDic["Datum"])
        if str(line).startswith("Betreff"):
            apenndStr = ""
            try:
                if lines[idx+1] != "\n":
                    apenndStr = apenndStr + " " +lines[idx+1]
            except:
                pass
            returnDic["Betreff"] = line.replace("Betreff:", "").strip()+apenndStr
            # print(returnDic["Betreff"])

    return returnDic

main()