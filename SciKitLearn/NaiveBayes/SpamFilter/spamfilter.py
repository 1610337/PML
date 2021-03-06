import os
import re
from collections import Counter
import shutil

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

import spamfilter_params as par

current_path = os.path.dirname(os.path.abspath(__file__))


def main():

    global blacklist, whitelist, critical_value, words_to_remove, top_words_to_account, prio
    blacklist = open(par.filename_blacklist).read().split("\n")
    whitelist = open(par.filename_whitelist).read().split("\n")
    upper_treshold = par.upper_treshold
    lower_treshold = par.lower_treshold
    words_to_remove = list(par.char_replaces.keys())
    words_to_remove = words_to_remove + par.words_ignore
    top_words_to_account = par.top_words_to_account
    prio = par.prio

    # set up paths to folders
    path_to_spams = current_path + "/dir.spam/"
    path_to_nospams = current_path + "/dir.nospam/"
    path_to_inputs = current_path + "/dir.mail.input/"

    # read emails and append
    spams = read_emails(path_to_spams, True)
    nospams = read_emails(path_to_nospams, False)
    input = read_emails(path_to_inputs, None)
    training_data = spams + nospams

    final_operations()

    # train model
    classifier, vectorizer, model_df = train_model(training_data, spams, nospams)

    log = open(current_path+"\\dir.filter.results\\"+'log.txt', 'w')

    for mail in input:

        print("Is this mail a spam: ", mail["Betreff"], file=log)
        print("Bayes Value", bayes_filter2(mail, model_df), file=log)

        final_eveluation = ""
        for val in prio:
            if val == "blacklist":
                if blacklist_filter(mail["Von"]):
                    #print("Blacklist won")
                    final_eveluation += "XSPAM: Blacklist \n"
                    break
            if val == "whitelist":
                if whitelist_filter(mail["Von"]):
                    #print("Whitelist won")
                    final_eveluation += "XSPAM: Whitelist \n"
                    break
            if val == "bayes":
                bayes_val = bayes_filter2(mail, model_df)
                if bayes_val > upper_treshold:
                    #print("Bayes : Spam --> ", bayes_val)
                    final_eveluation += "XSPAM: spam \n"
                    break
                elif bayes_val < lower_treshold:
                    print("Bayes : Ham --> ", bayes_val, file=log)
                    final_eveluation += "XSPAM: ham \n"
                    break
                else:
                    print("Bayes : undetermined--> ", bayes_val,  file=log)
                    final_eveluation += "XSPAM: undetermined \n"

        final_eveluation += "XSPAM Proba: " + str(bayes_filter2(mail, model_df)) + "\n"
        final_eveluation += "*"*80 + "\n"

        copy_and_append_head(final_eveluation, mail["title"], mail)
    log.close()


def copy_and_append_head(evaluation, name, mail):

    #  copy  mail into outputfolder
    shutil.copy(current_path + "/dir.mail.input/" + name, current_path + "/dir.mail.output")

    # We open the file in WRITE mode
    src = open(current_path + "/dir.mail.output/" + name, "w")
    src.writelines(evaluation+mail["title"]+mail["Von"]+mail["Betreff"]+mail["text"])
    src.close()

    # copy word count into results folder
    shutil.copy(current_path + "\\dir.mail.output\\wordcount.csv", current_path + "/dir.filter.results/")
    # copy params into results folder
    shutil.copy(current_path + "\\spamfilter_params.py", current_path + "/dir.filter.results/")


def bayes_filter2(mail, model_df):

    df = get_word_df(mail)
    # merge of the current mail with df with all mails on the "word" col
    main_df = df.merge(model_df, on="Word")
    # create relative ham/spam count cols TODO integrate the relative value maybe
    main_df['SpamQuote'] = main_df['SpamCount'] / (main_df['SpamCount']+main_df['HamCount'])
    # main_df['RelativeSpamVal'] = main_df['WordCount'] * main_df['SpamCount']
    # main_df['RelativeHamVal'] = main_df['WordCount'] * main_df['HamCount']

    final_val = sum(main_df['SpamQuote']) / len(main_df['SpamQuote'])

    return final_val


def get_word_df(mail):
    # get a datagrame with words in one col and the number of how often the word occured in the other col
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform([mail["text"]])
    classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)  # class_prior=[1, 0]
    targets = [mail["class"]]
    classifier.fit(counts, targets)
    index = 0
    coef_features_c1_c2 = []
    for feat, c1, c2 in zip(vectorizer.get_feature_names(), classifier.feature_count_[0], classifier.feature_count_[1]):
        coef_features_c1_c2.append(tuple([classifier.coef_[0][index], feat, c1, c2]))
        index += 1
    df = pd.DataFrame(columns=["Word", "WordCount"])
    for ind, i in enumerate(sorted(coef_features_c1_c2)):
        df.loc[ind] = [i[1], i[2]]

    return df


def final_operations():
    #  delete folder
    shutil.rmtree(current_path+"/dir.mail.output/")

    #  create folder again
    if not os.path.exists(current_path+"/dir.mail.output/"):
        os.makedirs(current_path+"/dir.mail.output/")


def train_model(training_data, spam_mails, ham_mails):
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform([mail["text"] for mail in training_data])
    print(counts)
    classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)  # class_prior=[1, 0]
    targets = [mail["class"] for mail in training_data]

    print(targets)
    classifier.fit(counts, targets)

    # write word count into output file
    index = 0
    coef_features_c1_c2 = []
    for feat, c1, c2 in zip(vectorizer.get_feature_names(), classifier.feature_count_[0], classifier.feature_count_[1]):
        coef_features_c1_c2.append(tuple([classifier.coef_[0][index], feat, c1, c2]))
        index += 1
    # print(sorted(coef_features_c1_c2))
    df = pd.DataFrame(columns=["Word", "HamCount", "SpamCount"])
    for ind, i in enumerate(sorted(coef_features_c1_c2)):
        df.loc[ind] = [i[1], i[2], i[3]]

    # TODO
    # main_df['SpamQuote'] = main_df['SpamCount'] / (main_df['SpamCount']+main_df['HamCount'])
    # makes more sense probably
    df['Ratio Ham 0 - Spam 1'] = df['HamCount']/(df['HamCount']+df['SpamCount'])


    df['WordInHams'] = [word_in_mails(dfword, ham_mails) for dfword in df['Word']]
    df['WordInSpams'] = [word_in_mails(dfword, spam_mails) for dfword in df['Word']]

    df.to_csv(current_path + "/dir.mail.output/" + "wordcount.csv", index=False)

    return classifier, vectorizer, df


def word_in_mails(word, mails):
    count = 0
    for mail in mails:
        if str(word).lower() in str(mail["text"]).lower().split():
            count = count +1
    return count


def bayes_spam_filter(classifier, vectorizer, mail):

    example = mail["text"]
    example_counts = vectorizer.transform([example])
    predictions = classifier.predict_proba(example_counts)

    print(predictions[0][0], "---", predictions[0][1])
    if predictions[0][0] >= critical_value:
        return "NoSpam", predictions[0][0], predictions[0][1]
    else:
        return "Spam", predictions[0][0], predictions[0][1]


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
            finalDic["title"] = filename
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