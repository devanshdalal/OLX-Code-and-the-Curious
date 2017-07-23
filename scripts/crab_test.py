2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
from pprint import pprint
import csv
 
from scikits.crab.models import MatrixPreferenceDataModel, MatrixBooleanPrefDataModel
from scikits.crab.metrics import pearson_correlation, euclidean_distances, jaccard_coefficient, cosine_distances, manhattan_distances, spearman_coefficient
from scikits.crab.similarities import ItemSimilarity, UserSimilarity
from scikits.crab.recommenders.knn import ItemBasedRecommender, UserBasedRecommender
from scikits.crab.recommenders.knn.neighborhood_strategies import NearestNeighborsStrategy
from scikits.crab.recommenders.knn.item_strategies import ItemsNeighborhoodStrategy
from scikits.crab.recommenders.svd.classes import MatrixFactorBasedRecommender
from scikits.crab.metrics.classes import CfEvaluator
 
"""
import random 
 
fieldnames = ['user_id', 'item_id', 'star_rating']
with open('dataset-recsys.csv', "w") as myfile: # writing data to new csv file
    writer = csv.DictWriter(myfile, delimiter = ',', fieldnames = fieldnames)    
    writer.writeheader()    
    
    for x in range(1, 21):
        items = random.sample(list(range(1, 41)), 20)
        random.randint(1,5)
        for item in items:        
            writer.writerow({'user_id': x, 'item_id': item, 'star_rating': random.randint(1, 5)})
"""
 
dataset = {}
with open('dataset-recsys.csv') as myfile:     
    reader = csv.DictReader(myfile, delimiter=',')    
    i = 0    
    for line in reader:            
        i += 1
        if (i == 1):
            continue    
        
        if (int(line['user_id']) not in dataset):
            dataset[int(line['user_id'])] = {}
            
        dataset[int(line['user_id'])][int(line['item_id'])] = float(line['star_rating'])
                    
 
model = MatrixPreferenceDataModel(dataset)
 
# User-based Similarity
 
#similarity = UserSimilarity(model, cosine_distances)
#neighborhood = NearestNeighborsStrategy()
#recsys = UserBasedRecommender(model, similarity, neighborhood)
 
# Item-based Similarity
 
similarity = ItemSimilarity(model, cosine_distances)
nhood_strategy = ItemsNeighborhoodStrategy()
recsys = ItemBasedRecommender(model, similarity, nhood_strategy, with_preference=False)
 
#recsys = MatrixFactorBasedRecommender(model=model, items_selection_strategy=nhood_strategy, n_features=10, n_interations=1)
 
evaluator = CfEvaluator()
 
#rmse = evaluator.evaluate(recsys, 'rmse', permutation=False)
#mae = evaluator.evaluate(recsys, 'mae', permutation=False)
#nmae = evaluator.evaluate(recsys, 'nmae', permutation=False)
#precision = evaluator.evaluate(recsys, 'precision', permutation=False)
#recall = evaluator.evaluate(recsys, 'recall', permutation=False)
#f1score = evaluator.evaluate(recsys, 'f1score', permutation=False)
 
#all_scores = evaluator.evaluate(recsys, permutation=False)
#all_scores = evaluator.evaluate(boolean_recsys, permutation=False)
 
result = evaluator.evaluate(recsys, None, permutation=False, at=10, sampling_ratings=0.7) 
 
# Cross Validation
#result = evaluator.evaluate_on_split(recsys, 'rmse', permutation=False, at=10, cv=5, sampling_ratings=0.7)
 
pprint (result)