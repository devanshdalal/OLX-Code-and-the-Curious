import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math,pickle,json,sys,time
from sklearn.metrics import f1_score
import xgboost

t0 = time.time()
last_update = time.time()
def time_elapsed(t0):
	return (time.time()-t0)/60

def create_submission(is_duplicate_array):
	with open('../submission.csv', 'w') as f:
		f.write('test_id,is_duplicate\n')
		for i,x in enumerate(is_duplicate_array):
			f.write(str(i)+','+str(x)+'\n' )

train = pd.read_csv('../input/train.csv').fillna(0)# [['id', 'question1', 'question2', 'is_duplicate']]
# test = train #pd.read_csv('../input/test.csv')[['test_id', 'question1', 'question2']]
test = pd.read_csv('../input/test.csv').fillna(0) #[['test_id', 'question1', 'question2']]


def train_model(train_x,train_y,test_x,test_y):
	# print('train_x,train_y,test_x,test_y',train_x[0],train_y[0],test_x[0],test_y[0])
	print('train_x',train_x[0])
	print('train_y',train_y[0])
	print('test_x',test_x[0])
	clf = xgboost.XGBClassifier(max_depth=3,n_estimators=100,seed=27,subsample=0.8,colsample_bytree=0.8)
	clf.fit(train_x, train_y)
	Z = clf.predict(test_x)
	if len(test_y)!=0:
		print('f1_score',f1_score( Z,  test_y ) )
	return Z

train_x=train.as_matrix()
train_y=np.array(train['is_duplicate'])
test_x=test.as_matrix()
test_y = np.array([])

def predict(s):
	global train_x,train_y,test_x,test_y
	n = len(train_x)
	if s<1:
		train_x,test_x = train_x[:int(n*s)],train_x[int(n*s):]
		train_y,test_y = train_y[:int(n*s)],train_y[int(n*s):]
	print( 'lens',len(train_x),len(train_y),len(test_x),len(test_y) )
	Z = train_model(train_x,train_y,test_x,test_y)
	if(s==1):
		create_submission(Z)

predict(1)

save_cache(forced=True)