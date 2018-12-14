# Params
inputfilename  = 'iris.data'                  #filename input data
outputfilename = inputfilename + '.erg'       #filename output data
inputseparator = ','                          #separator for csv columns
labelcolumn    = 4                            #column of label
columnfilter   = [0,1,2,3]                    #columns with features
columnflen     = len(columnfilter)            #count of features in data
featurecols    = [0,1]                        #selected Features
linefilter     = '[]'                         #linefilter: lines to ignore
firstrelpred   = 10                           #count of first neighbours in outputlist
rpdigits       = 4                            #relative prediction: number of digits after decimal point
dnnrange       = [0,1,2,3,4]                  #show different nearest neighbour range
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
printseparator = '*'*80+'\n'