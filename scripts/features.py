import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math,pickle,json,sys,time
from nltk import word_tokenize
import nltk, string, traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import f1_score
import xgboost

from gensim import corpora
import gensim 
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.models import Word2Vec

# nltk.download('punkt') # if necessary...
stop_words = nltk.corpus.stopwords.words('english')
cache_use = True if sys.argv[1]=='1' else False
_cache = json.load(open("../data/cache.txt"))
if not cache_use:
	if 'stem' in _cache:
		_cache = {  'stem':_cache['stem'] }
	else:
		_cache = {  'stem':{} }

t0 = time.time()
last_update = time.time()
def time_elapsed(t0):
	return (time.time()-t0)/60


def save_cache(forced=False):
	global _cache,last_update
	if(forced or time_elapsed(last_update)>5):
		print(time_elapsed(t0), "minutes elapsed!!")
		last_update = time.time()
		json.dump(_cache, open("../data/cache.txt",'w'))


def create_submission(is_duplicate_array):
	with open('../submission.csv', 'w') as f:
		f.write('test_id,is_duplicate\n')
		for i,x in enumerate(is_duplicate_array):
			f.write(str(i)+','+str(x)+'\n' )

corpus = {}
text2id = {}

stemmer = nltk.stem.porter.PorterStemmer()
def stem_tokens(tokens):
	result=[]
	for item in tokens:
		if item not in _cache['stem']:
			_cache['stem'][item]=stemmer.stem(item)
		result.append(_cache['stem'][item])
	return result

remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def normalize(text):
	try:
		res = stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
	except Exception as e:
		print('text',text,str(e))
		return []
	return res

def lsi_preprocess(text):
	try:
		text = text.lower()
		doc = word_tokenize(text)
		doc = [word for word in doc if word not in stop_words]
		doc = [word for word in doc if word.isalpha()]
		return doc
	except Exception as e:
		print('lsi error',text,str(e))
		exit(-1)
		return []

def brute_cosine_sim(text1, text2):
	vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
	try:
		tfidf = vectorizer.fit_transform([text1, text2])
		return ((tfidf * tfidf.T).A)[0,1]
	except Exception as e:
		print('brute_cosine_sim, text1:',text1,'text2:',text2)
		return 0

train = pd.read_csv('../input/train.csv').fillna('')	# [['id', 'question1', 'question2', 'is_duplicate']]
# test = train #pd.read_csv('../input/test.csv')[['test_id', 'question1', 'question2']]
test = pd.read_csv('../input/test.csv').fillna('')[:100] #[['test_id', 'question1', 'question2']]

corpus_tfidf = ''
lsi = ''
word2vec_model = ''

def initialize_corpus():
	global corpus,train,text2id,corpus_tfidf,lsi,word2vec_model
	def process_text(text):
		save_cache()
		return { 'nz':normalize(text), 'lp':lsi_preprocess(text) }
	corpus = {x:{} for x in range(2*len(train)+2*len(test))}
	last_id = 0
	for i,row in train.iterrows():
		print('il',i)
		qid1,qid2 = row['qid1'],row['qid2']
		q1,q2 = row['question1'],row['question2']
		corpus[ qid1 ] =  process_text( q1 )
		corpus[ qid2 ] =  process_text( q2 )
		# text2id[q1] , text2id[q2] = qid1,qid2
		last_id = max(max(qid1,qid2),last_id)
	for i,row in test.iterrows():
		print('it',i)
		q1,q2 = row['question1'],row['question2']
		last_id+=1
		corpus[ last_id ] =  process_text( q1 )
		text2id[q1] = last_id
		last_id+=1
		corpus[ last_id ] =  process_text( q2 )
		text2id[q2] = last_id

	for _,ind in enumerate(corpus):
		if 'nz' not in corpus[ind]:
			corpus[ind]['nz']=['']

	dictionary = corpora.Dictionary( corpus[ind]['nz'] for ind in corpus if 'nz' in corpus[ind] )
	corpus_gensim = [dictionary.doc2bow( corpus[ind]['nz']) for ind in corpus]
	tfidf = TfidfModel(corpus_gensim)
	corpus_tfidf = tfidf[corpus_gensim]
	lsi = LsiModel(corpus_tfidf, id2word=dictionary)

	# pre_trained_w2v = '../data/GoogleNews-vectors-negative300.bin.gz'
	# word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_w2v, binary=True)
	# word2vec_model.init_sims(replace=True)

initialize_corpus()

def document_vector(qid):
	doc = corpus[qid]['lp']
	# remove out-of-vocabulary words
	doc_vec = [word for word in doc if word in word2vec_model.vocab]
	return 0 if len(doc_vec)==0 else np.mean(word2vec_model[doc_vec], axis=0)

def wmdistance(qid1, qid2):
    return word2vec_model.wmdistance(corpus[qid1]['lp'], corpus[qid2]['lp'])


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

def add_features(raw_vectors, train_side=True):
	feature_vectors = []
	# print('raw_vectors',raw_vectors)
	for i,row in raw_vectors.iterrows():
		print('lf' if train_side else 'tf',i)
		save_cache()

		q1,q2 = row['question1'],row['question2']
		qid1,qid2=0,0
		if train_side:
			qid1,qid2 = row['qid1'],row['qid2']
		else:
			qid1,qid2 = text2id[q1],text2id[q2]

		feature_vectors.append( []  )

		# brute_cosine_similarity			
		# feature_vectors[-1].append(  brute_cosine_sim( q1 , q2 ) )

		# lengths as feature
		feature_vectors[-1].append( min(len(q1),len(q2))  )
		feature_vectors[-1].append( max(len(q1),len(q2))  )

		# Latent semantic Indexing
		sim = gensim.matutils.cossim( lsi[corpus_tfidf[qid1]], lsi[corpus_tfidf[qid2]])
		feature_vectors[-1].append( sim )
		# print('final_sim',sim)

		# #Centroid_similarity
		# centroid_sim=0
		# try:
		# 	dv1,dv2 = document_vector(qid1),document_vector(qid2)
		# 	centroid_sim = cosine_similarity( np.array(dv1).reshape(1,-1), np.array(dv2).reshape(1,-1) );
		# 	# print('csim',centroid_sim)
		# except Exception as e:
		# 	print('centriod error',qid1,qid2,q1,q2)
		# feature_vectors[-1].append( centroid_sim )


		# #wm_distance
		# wmd = 0 
		# try:
		# 	wmd = wmdistance(qid1,qid2)
		# 	# print('wmd',wmd)
		# except Exception as e:
		# 	print('wmd error',qid1,qid2,q1,q2)
		# feature_vectors[-1].append( wmd  )

	return np.array(feature_vectors)

train_x=add_features(train)
train_y=np.array(train['is_duplicate'])
test_x=add_features(test,train_side=False)
test_y = np.array([])


save_cache(forced=True)