import parameters as par

import pandas as pd

'''
This Code is mainly a copy of the code we used in the workshop.
The only change is the method wrapping and the return value as a pandas df
'''

featurecols = par.featurecols
linefilter = par.linefilter
printseparator = par.printseparator
inputseparator = par.inputseparator
inputfilename = par.inputfilename
columnfilter = par.columnfilter
labelcolumn = par.labelcolumn
linefilter = par.linefilter
columnflen = par.columnflen
dnnrange = par.dnnrange
rpdigits = par.rpdigits
firstrelpred = par.firstrelpred


def get_filtered_data():
    # Header in Outputfile
    of = open(par.outputfilename, 'w')
    print('Particular Data Analysis (with kNN) [1.2]\n', file=of)
    print(f"selected features: {featurecols}", file=of)
    print(f"ignored lines: {linefilter}", file=of)
    print(printseparator, file=of)

    # Load Data
    filedata = [line.split(inputseparator) for line in open(inputfilename).read().split("\n") if line != '']

    # Filter Data
    rawdata = [[filedata[j][i] for i in range(len(filedata[j])) if i in columnfilter or i == labelcolumn] for j in
               range(len(filedata)) if eval('j not in ' + linefilter)]
    print('count of data records: ', len(rawdata), '\n', file=of)

    # Prepare Data
    for l in range(len(rawdata)):
        for i in range(columnflen): rawdata[l][i] = float(rawdata[l][i])

    # Function: distance of x and y
    def distance(x, y):
        return pow(sum([abs(x[i] - y[i]) for i in range(len(x))]), 0.5)

    dt = {}  # Working Data; Select features
    for i in range(len(rawdata)):
        dt[i] = {'features': [rawdata[i][k] for k in featurecols], 'label': rawdata[i][labelcolumn]}

    # Calc Distance-Tables, sort them and do a first analysis on predictions
    for i in range(len(dt)):
        dt[i]['dist'] = {j: distance(dt[i]['features'], dt[j]['features']) for j in range(len(dt)) if i != j}
        dt[i]['sorted'] = sorted([(e, dt[i]['label'] == dt[e]['label']) for e in dt[i]['dist']],
                                 key=lambda sd: dt[i]['dist'][sd[0]], reverse=False)
        dt[i]['relpredict'] = [(k, round([e[1] for e in dt[i]['sorted'][:k]].count(True) / float(k), rpdigits)) for k in
                               range(1, len(dt[i]['sorted']) + 1)]
        dt[i]['k_maxpred'] = [e[1] for e in dt[i]['relpredict']].count(1.0)

    # Show perdictions for nearest neighbours
    for i in dt: print(
        f"{i:3} : {dt[i]['features']} - {dt[i]['label']} - KMP: {dt[i]['k_maxpred']} - {dt[i]['relpredict'][:firstrelpred]}",
        file=of)
    print(printseparator, file=of)

    # Show nodes with different nearest neighbours
    print(f"Nodes with different first neighbours", file=of)
    for d in dnnrange:
        print(
            f"KMP {d}: {[dt[i]['k_maxpred'] for i in range(len(dt))].count(d)} {[i for i in range(len(dt)) if dt[i]['k_maxpred']==d]}",
            file=of)

        print(
            f"KMP {d}: {[dt[i]['k_maxpred'] for i in range(len(dt))].count(d)} {[i for i in range(len(dt)) if dt[i]['k_maxpred']==d]}")
    # Close up
    of.close()

    df = pd.DataFrame.from_records(rawdata, columns=par.col_names)

    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].apply(pd.to_numeric)

    df.to_csv("data_results.csv")

    return df
