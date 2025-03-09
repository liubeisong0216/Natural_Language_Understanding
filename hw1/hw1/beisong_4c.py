import itertools
from typing import Dict

import numpy as np

from embeddings import Embeddings
from test_analogies import cosine_sim, get_closest_words, load_analogies, \
    run_analogy_test

k = 2
# Load embeddings from a file
embeddings1 = Embeddings.from_file("data/glove_50d.txt")
embeddings2 = Embeddings.from_file("data/glove_100d.txt")
embeddings3 = Embeddings.from_file("data/glove_200d.txt")

# test_data = load_analogies("data/analogies.txt")
for embeddings in [embeddings1, embeddings2, embeddings3]:
    print('-------------------')
    test_data = [('france', 'paris', 'italy'), 
                ('france', 'paris', 'japan'), 
                ('france', 'paris', 'florida'), 
                ('big', 'bigger', 'small'),
                ('big', 'bigger', 'cold'),
                ('big', 'bigger', 'quick')]

    for analogy in test_data:
        word_vectors = embeddings[analogy]
        v_pred = word_vectors[1:2] - word_vectors[0:1] + word_vectors[2:3]
        closest_words = get_closest_words(embeddings, v_pred, k=k)[0]
        result = closest_words[1] if analogy[2] == closest_words[0] else closest_words[0]
        print(analogy, result)

# glove 50
# ('france', 'paris', 'italy') rome
# ('france', 'paris', 'japan') tokyo
# ('france', 'paris', 'florida') miami
# ('big', 'bigger', 'small') larger
# ('big', 'bigger', 'cold') warmer
# ('big', 'bigger', 'quick') quicker

# glove 100
# ('france', 'paris', 'italy') rome
# ('france', 'paris', 'japan') tokyo
# ('france', 'paris', 'florida') miami
# ('big', 'bigger', 'small') larger
# ('big', 'bigger', 'cold') cooler
# ('big', 'bigger', 'quick') quicker

# glove 200
# ('france', 'paris', 'italy') rome
# ('france', 'paris', 'japan') tokyo
# ('france', 'paris', 'florida') miami
# ('big', 'bigger', 'small') smaller
# ('big', 'bigger', 'cold') colder
# ('big', 'bigger', 'quick') quicker