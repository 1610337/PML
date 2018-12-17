"""Parameter fuer spamfilter.py"""
prio = ["blacklist", "whitelist", "bayes"]
filename_blacklist="blacklist.txt"
top_words_to_account = 100
filename_whitelist="whitelist.txt"
filename_results="spamfilter.results"
filename_nbwordtable="nb.wordtable"
filename_logfile="spamfilter"
priorityorder="whitelist", "blacklist", "naive_bayes"
nb_wordtable={}
upper_treshold=0.67               #nb_level greater_or_equal is spam
lower_treshold=0.33             #nb_level loweror equal is nospam
                                 #in between is undetermined
nb_spam_class={"spam":(1.0,upper_treshold),
               "undetermined":(upper_treshold,lower_treshold),
               "nospam":(lower_treshold,0.0)}
char_replaces={'"':' ', "_":" ", ",":" ", "-":" ", "+":" ", "„":" ", "’":" ", "“":" ",
               "%":" ", ".":" ", "\t":" ", "[":" ", "]":" ", "<":" ", ">":" ", "/":" ", "=":" ",
               "(":" ", ")":" ", "…":" ", "  ":" "} #\n nicht rein

words_ignore=[":", "*", "#", "!", "&", "/", ";", "?", "@", "|", "©", "®", "´", "·"]
