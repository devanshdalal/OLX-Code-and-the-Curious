import pandas as pd
import numpy as np
import sys,csv
import time
import dill
import math
from heapq import *
import operator

curr_time=time.time()
ADID_LIMIT = 2922042
USER_LIMIT = 15068
CATEG_LIMIT = 10
categories=[800,815,806,859,811,853,881,888,887,362]

converter={'user_id':np.int64,'ad_id':np.int64}
# 'images_count':np.int64,'ad_impressions':np.int64,'ad_views':np.int64,'ad_messages':np.int64}
ud=pd.read_csv('data/user_data.csv',dtype=converter).fillna(0)
ad=pd.read_csv('data/ads_data.csv',dtype=converter).fillna(0)
umt=pd.read_csv('data/user_messages_test.csv').fillna(0)
um=pd.read_csv('data/user_messages.csv').fillna(0)

print('pds loaded', time.time()-curr_time)
curr_time=time.time()


a2c = [-1]*ADID_LIMIT
with open('data/a2c.pkl', 'rb') as f:
	a2c = dill.load(f)
# for i,row in ad.iterrows():
# 	ad_id,c_id=row['ad_id'],row['category_id']
# 	a2c[ad_id]=i
# with open('data/a2c.pkl', 'wb') as f:
# 	dill.dump(a2c,f)
print('a2c computed', time.time()-curr_time)
# for x in ud.columns:
# 	y=np.array(ud[x].unique())
# 	print(x,y,len(y))

# print('-----------------------------------------------------------------------------------')
# for x in ad.columns:
# 	y=np.array(ad[x].unique())
# 	print(x,y,len(y))



def create_submission(values,user_messages,name):
	submission = user_messages.assign(ads=pd.Series(values))
	submission.to_csv('submissions/'+name)


def popularity_based(user_data,ad_data,user_messages):	
	frequency = [0]*ADID_LIMIT
	for i,row in user_data.iterrows():
		ad_id,ad_views=row['ad_id'],row['ad_views']	
		frequency[ad_id] = max(ad_views,frequency[ad_id])

	categories_={x:[] for x in category_ids}
	for i,row in ad_data.iterrows():
		ad_id,c_id=row['ad_id'],row['category_id']
		a2c[ad_id]=c_id
		categories_[c_id].append(ad_id)

	for x in category_ids:
		categories_[x] = nlargest(10, categories_[x],key=lambda y: frequency[y])

	values=[[] for i in range(10)]
	for i,row in user_messages.iterrows():
		u_id,c_id=row['user_id'],row['category_id']
		for x in range(1,10):
			if c_id not in category_ids:
				values[x].append( '[]' )
			else:
				values[x].append( str(categories_[c_id][:x]) )
	for x in range(1,10):
		create_submission(values[x],umt,str(x)+'.csv')

# curr_time=time.time()
# popularity_based(ud[:100],ad,umt)
# print('popularity_based done', time.time()-curr_time)

# curr_time=time.time()
# table={}
# cc=0
# for i,x in um.iterrows():
# 	dt=(x['user_id'],x['category_id'])
# 	table[dt]=1+(table[dt] if dt in table else 0)
# for i,x in umt.iterrows():
# 	dt=(x['user_id'],x['category_id'])
# 	if dt in table:
# 		cc+=1

# print('stats',cc,len(um),len(umt))
# print('stats done', time.time()-curr_time)
def f7(seq,seen):
	
	# res=[]
	# for x in seq:
	# 	rw=ad.ix[a2c[x]]
	# 	if rw['enabled']==0 or rw['origin']=='notification_center':
	# 		continue
    return [x for x in seq if (x not in seen)]

def filter_best(seq,u_ind):
	urow=None
	valid=False
	if(u_ind!=-1):
		urow = ud.ix[u_ind]
		valid=True
	#      sim , distance
	h=[]
	for i,x in enumerate(seq):
		rw = ad.ix[a2c[x]]
		if (rw['enabled']==0): 
			continue
		if(valid and (rw['creation_time']<=urow['event_time'])):
			continue
		dist = math.sqrt((rw['lat']-urow['user_lat']).item()**2+(rw['long']-urow['user_long']).item()**2) if valid else 300
		heappush(h,(dist,x))
	# print('uind',h,'u_ind',u_ind)
	values=[]
	while h:
		# print('h top',h[0],)
		values.append(heappop(h)[1])
	# print('vals',values)
	return values

def user_item_collaborative(saved,user_data,user_messages,user_messages_test):
	
	user_sim,user_all,user_info,user_seen=None,None,None,None
	if saved:
		with open('data/user_item.pkl', 'rb') as f:
			user_sim = dill.load(f)
			user_all = dill.load(f)
			user_info = dill.load(f)
			user_seen = dill.load(f)
	else:
		user_info = [-1]*USER_LIMIT
		user_all = [ {j:set() for j in categories} for i in range(USER_LIMIT)]
		user_seen = [ {j:set() for j in categories} for i in range(USER_LIMIT)]
		for i,row in user_data.iterrows():
			u_id,ad_id=row['user_id'],row['ad_id']
			rr=ad.ix[a2c[ad_id]]
			c_id=rr['category_id']
			user_info[u_id]=i
			user_all[u_id][c_id].add(ad_id)
			if row['origin']=='notification_center' or row['origin']=='home' or row['event']=='first_message':
				user_seen[u_id][c_id].add(ad_id)
			# print('uall',i,u_id,len(user_all[u_id][c_id]))

		for i,row in user_messages.iterrows():
			u_id,ads,c_id=row['user_id'],eval(row['ads']),row['category_id']
			# print('um',i,u_id)
			user_all[u_id][c_id]=user_all[u_id][c_id].union(set(ads))
			# print('user_ad[',u_id,']',user_ad[u_id])

		user_sim ={j:[ {} for j in range(USER_LIMIT) ] for j in categories}

		for i in range( min(USER_LIMIT,len(ad_data),len(user_data)) ):
			print('user_sim',i)
			for j in range(i+1,USER_LIMIT):
				for k in categories:
					ins=user_all[i][k].intersection(user_all[j][k])
					if len(ins)>0:
						sim = len(ins)*1.0/len(user_all[i][k].union(user_all[j][k]))
						user_sim[k][i][j]=sim
						user_sim[k][j][i]=sim
				# if(len(user_sim[k][i])>0):
				# 	print('u',i,len(user_sim[k][i]))
		# print('user_ad[',u_id,']',user_ad[u_id])	 
		with open('data/user_item.pkl', 'wb') as f:
			dill.dump(user_sim,f)
			dill.dump(user_all,f)
			dill.dump(user_info,f)
			dill.dump(user_seen,f)
	

	values=[]
	for i,row in user_messages_test.iterrows():
		if(i%10==0):
			print('umt',i)
		u_id,c_id=row['user_id'],row['category_id']

		# print('user_sim[u_id]',user_sim[u_id])
		Y = nlargest(10,list(user_sim[c_id][u_id]),key=lambda y:user_sim[c_id][u_id][y])
		if(Y==[]):
			values.append( '[]' )
			continue
		# print('Y',Y)
		accu = []
		for x in Y:
			print('sim users',user_sim[c_id][u_id][x],)
			for xx in user_all[x][c_id]:
				accu.append(xx)
				# print('accu',accu)
		accu=f7(accu,user_seen[u_id][c_id])
		accu=filter_best(accu,user_info[u_id])
		if len(accu):
			print('found',i,len(accu),'Y',Y)
		values.append( str(accu[:10]) )
	create_submission(values,user_messages_test,'s.csv')

curr_time=time.time()
# user_item_collaborative(1,ud.ix[:400],um.ix[:100],umt.ix[:100])
user_item_collaborative(1,ud.ix[:5000],um.ix[:1000],umt.ix[:100])
print('user_item_collaborative computed', time.time()-curr_time)

# def item_item_collaborative(saved,user_data,user_messages,user_messages_test):
# 	item_sim,item_all,user_items=None,None,None
# 	if saved:
# 		with open('data/item_item.pkl', 'rb') as f:
# 			item_sim = dill.load(f)
# 			item_all = dill.load(f)
# 			user_items = dill.load(f)
# 	else:
# 		item_all = {j:[{} for i in range(ADID_LIMIT)] for j in categories}
# 		item_all = {j:[{} for i in range(USER_LIMIT)] for j in categories}
# 		for i,row in user_data.iterrows():
# 			u_id,ad_id=row['user_id'],row['ad_id']
# 			c_id=a2c[ad_id]
# 			if c_id==-1:
# 				continue
# 			item_all[c_id][ad_id].add(u_id)
# 			user_items[c_id][u_id].add(ad_id)
# 			# print('uall',i,ad_id,len(user_all[ad_id][c_id]))

# 		item_sim ={j:[ {} for j in range(ADID_LIMIT) ] for j in categories}

# 		for k in categories:
# 			for i in range(min(ADID_LIMIT,len(user_data)) ):
# 				print('item_sim',i)
# 				for j in range(i+1,ADID_LIMIT):
# 					ins=item_all[k][i].intersection(item_all[k][j])
# 					if len(ins)>0:
# 						sim = len(ins)*1.0/len(item_all[k][i].union(item_all[k][i]))
# 						item_sim[k][i][j]=sim
# 						item_sim[k][j][i]=sim
# 				# if(len(user_sim[k][i])>0):
# 				# 	print('u',i,len(user_sim[k][i]))
# 		# print('user_ad[',u_id,']',user_ad[u_id])	 
# 		with open('data/item_item.pkl', 'wb') as f:
# 			dill.dump(item_sim,f)
# 			dill.dump(item_all,f)
# 			dill.dump(user_items,f)

# 	values=[]

