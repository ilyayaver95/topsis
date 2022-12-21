import pandas as pd
import numpy as np
from collections import defaultdict

attributes = np.array(["style", "reliability", "fuel_consumption", "price"])
candidates = np.array(["honda", "ford", "mazda", "subaru"])
raw_data = np.array([
    [7, 9,  9,  8],
    [8, 7,  8,  7],
    [9, 6,  8,  9],
    [6, 7,  8, 6]
])

data = [{'style': 7, 'reliability': 9, 'fuel_consumption': 9, 'price': 9},
        {'style': 8, 'reliability': 7, 'fuel_consumption': 8, 'price': 7},
        {'style': 9, 'reliability': 6, 'fuel_consumption': 8, 'price': 9},
        {'style': 6, 'reliability': 7, 'fuel_consumption': 8, 'price': 6}]


# Creates padas DataFrame by passing
# Lists of dictionaries and row index.
df = pd.DataFrame(data, index=["honda", "ford", "mazda", "subaru"])

# Extract only the ranking columns
vote_cols = df.columns
df = df[vote_cols]

# Calculate total number of candidates
candidates = set()
for candidate in df.index:
        candidates.add(candidate)
n_candidates = len(candidates)
print(n_candidates, 'candidates')

# Get how many ranks were available to give
n_ranks = len(vote_cols)
print(n_ranks, 'ranks')

print('='*20)


scores = defaultdict(int)
for i, cand  in enumerate(candidates):
    # Assuming first column is highest rank
    attributes = df.loc[cand].index

    points = 0
    for attr in attributes:
        points = points + df.loc[cand][attr]

    scores[cand] += points

# Sort candidates by score, descending
ranking = sorted(scores.items(), key=lambda i: i[1], reverse=True)
for cand, score in ranking:
    print(cand, score)
print('='*20)