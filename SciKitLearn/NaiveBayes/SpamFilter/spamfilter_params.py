"""Parameter fuer spamfilter.py"""
filename_blacklist="blacklist.txt"
filename_whitelist="whitelist"
filename_results="spamfilter.results"
filename_nbwordtable="nb.wordtable"
filename_logfile="spamfilter"+log_suffix
priorityorder="whitelist", "blacklist", "naive_bayes"
nb_wordtable={}
nb_spam_level=0.67               #nb_level greater_or_equal is spam
nb_nospam_level=0.33             #nb_level loweror equal is nospam
                                 #in between is undetermined
nb_spam_class={"spam":(1.0,nb_spam_level), 
               "undetermined":(nb_spam_level,nb_nospam_level), 
               "nospam":(nb_nospam_level,0.0)}
char_replaces={'"':' ', "\n":" ", "_":" ", ",":" ", "-":" ", "+":" ", "„":" ", "’":" ", "“":" ",
               "%":" ", ".":" ", "\t":" ", "[":" ", "]":" ", "<":" ", ">":" ", "/":" ", "=":" ",
               "(":" ", ")":" ", "…":" ", "  ":" "}
words_ignore=["", " ", ":", "*", "#", "!", "&", "/", ";", "?", "@", "|", "©", "®", "´", "·"]
