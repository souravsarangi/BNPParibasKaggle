import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv(
    filepath_or_buffer='train.csv', 
    #header=None, 
    sep=',',
    low_memory=False)

test = pd.read_csv(
    filepath_or_buffer='test.csv', 
    #header=None, 
    sep=',',
    low_memory=False)
target = train['target']
train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
test = test.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

vs = pd.concat([train, test])
num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
            'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
            'v21', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
            'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50',
            'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
            'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84', 
            'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98', 
            'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
            'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']

def find_denominator(df, col):
    """
    Function that trying to find an approximate denominator used for scaling.
    So we can undo the feature scaling.
    """
    print type(df[col].dropna())
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax() 

for c in num_vars:
    if c not in train.columns:
        continue
    
    train.loc[train[c].round(5) == 0, c] = 0
    test.loc[test[c].round(5) == 0, c] = 0

    denominator = find_denominator(vs, c)
    train[c] *= 1/denominator
    test[c] *= 1/denominator

for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999


kf = KFold(train.shape[0], n_folds=3, random_state=1)
alg = GaussianNB()
 
#RandomForestClassifier(n_estimators=100,criterion= 'entropy')

predictions = []
for trainkf, test in kf:
	print trainkf
	# The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
	train_predictors = (train.iloc[trainkf,:])
	# The target we're using to train the algorithm.
	train_target = target.iloc[trainkf]
	print train_predictors.shape[0],train_predictors.shape[1]
	print train_target.shape[0]
	print train.iloc[test,:].shape[0],train.iloc[test,:].shape[1]
    # Training the algorithm using the predictors and target.
	alg.fit(train_predictors, train_target)
	# We can now make predictions on the test fold
	test_predictions = alg.predict(train.iloc[test,:])
	predictions.append(test_predictions)
	# The predictions are in three separate numpy arrays.  Concatenate them into one.  
    # We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

    # Map predictions to outcomes (only possible outcomes are 1 and 0)

accuracy=0


for i in range(len(predictions)):
    if int(predictions[i])==target.values[i]:
        accuracy+=1

accuracy = accuracy*1.0 / len(predictions)
print accuracy

