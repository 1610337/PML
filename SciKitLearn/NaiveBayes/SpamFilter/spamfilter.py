import os
import re
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Please set input parameters in input_file.txt
blacklist = []
whitelist = []
critical_value = 0.0
words_to_remove = []
top_words_to_account = 0
dic = {}


def main():

    # Read inputfile and fill variables
    with open("input_file.txt", 'r') as text_file:
        [dic.update(dict) for dict in [{line.split(":")[0]:line.split(":")[1].strip().split(",")} for line in text_file.readlines()]]
    global blacklist, whitelist, critical_value, words_to_remove, top_words_to_account
    blacklist = dic["blacklist"]
    whitelist = dic["whitelist"]
    critical_value = float(dic["critical_value"][0])
    words_to_remove = dic["words_to_remove"]
    top_words_to_account = dic["top_words_to_account"][0]

    # set up paths to folders
    path_to_spams = os.path.dirname(os.path.abspath(__file__)) + "/dir.spam/"
    path_to_nospams = os.path.dirname(os.path.abspath(__file__)) + "/dir.nospam/"
    path_to_inputs = os.path.dirname(os.path.abspath(__file__)) + "/dir.mail.input/"

    # read emails and append
    spams = read_emails(path_to_spams, True)
    nospams = read_emails(path_to_nospams, False)
    input = read_emails(path_to_inputs, None)
    training_data = spams + nospams

    for mail in input:
        print("Is this mail a spam: ", mail["Betreff"])

        if blacklist_filter(mail["Von"]):
            print("Yes --- blacklist")
            continue
        if whitelist_filter(mail["Von"]):
            print("No --- whitelist")
            continue
        solution = bayes_spam_filter(training_data, mail)
        if solution == "NoSpam":
            print("No --- bayes")
            continue
        if solution == "Spam":
            print("Yes --- bayes")
            continue

        break


def bayes_spam_filter(training_data, mail):

    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform([mail["text"] for mail in training_data])

    classifier = MultinomialNB()
    targets = [mail["class"] for mail in training_data]

    classifier.fit(counts, targets)

    example = mail["text"]
    example_counts = vectorizer.transform([example])
    predictions = classifier.predict_proba(example_counts)

    index = 0
    coef_features_c1_c2 = []

    for feat, c1, c2 in zip(vectorizer.get_feature_names(), classifier.feature_count_[0], classifier.feature_count_[1]):
        coef_features_c1_c2.append(tuple([classifier.coef_[0][index], feat, c1, c2]))
        index += 1

    #for i in sorted(coef_features_c1_c2):
        #print(i)

    print(predictions[0][0], "---", predictions[0][1])
    if predictions[0][0] >= critical_value:
        return "NoSpam"
    else:
        return "Spam"



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
        if not(item.isalpha()):
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


def read_emails(path, spam):

    # TODO check if the file is actually a txt file

    mails = []
    for filename in os.listdir(path):
        finalDic = {}
        with open(path+filename, 'r') as text_file:
            header = []
            normalText = ""
            noHeader = False

            for line in text_file:
                # The first empty line marks the end of the header
                if line == "\n" and not(noHeader):
                    finalDic.update(get_header_dic(header))
                    noHeader = True

                if noHeader and line.strip() != "":
                    normalText = normalText + " " + (line.strip())
                else:
                    header.append(line)

            # if we wanna append something from the header into the normal text:
            # try: normalText = normalText + finalDic["An"]
            # except: pass

            # this is to remove all single characters in the string
            normalText = ' '.join([w for w in normalText.split() if len(w)>1])

            finalDic["text"] = normalText
            if spam:
                finalDic["class"] = "Spam"
            else:
                finalDic["class"] = "NoSpam"

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
