import pickle

with open("train_results_without_bitfit.p", "rb") as f:
    results = pickle.load(f)

print(results)