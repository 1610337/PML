import os
import re

blacklist = ["123@hotmail.de"]
whitelist = ["professoren-bounces2@ml.hs-mannheim.de"]
def main():

    path_to_spams = os.path.dirname(os.path.abspath(__file__)) + "/dir.spam/"
    path_to_nospams = os.path.dirname(os.path.abspath(__file__)) + "/dir.nospam/"
    path_to_inputs = os.path.dirname(os.path.abspath(__file__)) + "/dir.mail.input/"

    spams = read_emails(path_to_spams)
    nospams = read_emails(path_to_nospams)
    input = read_emails(path_to_inputs)

    for mail in input:
        print("Is this mail a spam: ", mail["Betreff"], mail["Von"])

        if(blacklist_filter(mail["Von"])):
            print("Yes --- blacklist")
            continue
        if(whitelist_filter(mail["Von"])):
            print("No --- whitelist")
            continue


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