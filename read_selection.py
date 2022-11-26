import pickle

with open('block_selection', 'rb') as f:
    l = pickle.load(f)

sorted_l = sorted(l, key=lambda tup: tup[1], reverse=True)
print(sorted_l)

for name, score in l:
    if score == 1:
        print(name)