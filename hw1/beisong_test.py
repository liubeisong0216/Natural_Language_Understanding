import itertools
from typing import Dict

import numpy as np

from embeddings import Embeddings
from test_analogies import cosine_sim, get_closest_words, load_analogies, \
    run_analogy_test

k = 1
# Load embeddings from a file
embeddings = Embeddings.from_file("data/glove_50d.txt")
# embeddings = Embeddings.from_file("data/glove_100d.txt")
# embeddings = Embeddings.from_file("data/glove_200d.txt")

test_data = load_analogies("data/analogies.txt")

result = {}

semantic_correct_cnt = 0
semantic_total_cnt = 0


syntatic_correct_cnt = 0
syntatic_total_cnt = 0

overal_correct_cnt = 0
overal_total_cnt = 0


for category, analogies in test_data.items():
    is_semantic = False if category.startswith('gram') else True
    correct_cnt = 0
    total = len(analogies)
    overal_total_cnt += total
    if is_semantic:
        semantic_total_cnt += total
    else:
        syntatic_total_cnt += total
    
    for analogy in analogies:
        # analogy: ('athens', 'greece', 'baghdad', 'iraq')
        word_vectors = embeddings[analogy]
        v_pred = word_vectors[1:2] - word_vectors[0:1] + word_vectors[2:3]
        v_true = np.array(analogy[3:4])
        closest_words = get_closest_words(embeddings, v_pred, k=k)[0]

        if v_true in closest_words:
            overal_correct_cnt += 1
            correct_cnt += 1
            if is_semantic:
                semantic_correct_cnt += 1
            else:
                syntatic_correct_cnt += 1

    accuracy = correct_cnt / total
    result[category] = [accuracy, correct_cnt, total]

for k,v in result.items():
    print(k,v)
print("semantic: ", semantic_correct_cnt, semantic_total_cnt, semantic_correct_cnt/semantic_total_cnt)
print("syntatic: ", syntatic_correct_cnt, syntatic_total_cnt, syntatic_correct_cnt/syntatic_total_cnt)
print("overall: ", overal_correct_cnt, overal_total_cnt, overal_correct_cnt/overal_total_cnt)





exit()
#####################################
result = {}

semantic_correct_cnt = 0
semantic_total_cnt = 0


syntatic_correct_cnt = 0
syntatic_total_cnt = 0

overal_correct_cnt = 0
overal_total_cnt = 0


for category, analogies in test_data.items():
    is_semantic = False if category.startswith('gram') else True
    correct_cnt = 0
    total = len(analogies)
    overal_total_cnt += total
    if is_semantic:
        semantic_total_cnt += total
    else:
        syntatic_total_cnt += total
    for a, b, c, d in analogies:
        if a not in embeddings or b not in embeddings or c not in embeddings or d not in embeddings:
            continue 
        v_target = embeddings[b] - embeddings[a] + embeddings[c]
        closest_words = get_closest_words(embeddings, np.array([v_target]), k=k)[0]
        if d in closest_words:
            overal_correct_cnt += 1
            correct_cnt += 1
            if is_semantic:
                semantic_correct_cnt += 1
            else:
                syntatic_correct_cnt += 1
    accuracy = correct_cnt / total
    result[category] = [accuracy, correct_cnt, total]

for k,v in result.items():
    print(k,v)
print("semantic: ", semantic_correct_cnt, semantic_total_cnt, semantic_correct_cnt/semantic_total_cnt)
print("syntatic: ", syntatic_correct_cnt, syntatic_total_cnt, syntatic_correct_cnt/syntatic_total_cnt)
print("overall: ", overal_correct_cnt, overal_total_cnt, overal_correct_cnt/overal_total_cnt)

# GloVe 50:
# semantic:  3545 8869 0.3997068440635923
# syntatic:  2945 10675 0.2758782201405152
# overall:  6490 19544 0.3320712239050348

# GloVe 100:
# semantic:  3945 8869 0.4448077573570865
# syntatic:  2965 10675 0.27775175644028105
# overall:  6910 19544 0.3535611952517397

# GloVe 200:
# semantic:  2809 8869 0.31672116360356295
# syntatic:  2319 10675 0.21723653395784542
# overall:  5128 19544 0.26238231682357754

# in paper
# CBOW 300 783M 15.5 53.1 36.1
# Skip-gram 300 783M 50.0 55.9 53.3


# for k = 2

# GloVe 50:
# semantic:  5019 8869 0.5659037095501184
# syntatic:  5727 10675 0.5364871194379391
# overall:  10746 19544 0.5498362668849774

# GloVe 100:
# semantic:  5894 8869 0.664561957379637
# syntatic:  7039 10675 0.6593911007025761
# overall:  12933 19544 0.6617376176831764

# GloVe 200:
# semantic:  6253 8869 0.705040027060548
# syntatic:  7172 10675 0.6718501170960187
# overall:  13425 19544 0.6869115841178879



    
