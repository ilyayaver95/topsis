import numpy as np               # for linear algebra
import pandas as pd              # for tabular output
from scipy.stats import rankdata # for ranking the candidates

# topsis example:

def rank_according_to(data):
    ranks = rankdata(data).astype(int)
    ranks -= 1
    return candidates[ranks][::-1]


attributes = np.array(["style", "reliability", "fuel_consumption", "price"])
candidates = np.array(["honda", "ford", "mazda", "subaru"])
raw_data = np.array([
    [7, 9,  9,  8],
    [8, 7,  8,  7],
    [9, 6,  8,  9],
    [6, 7,  8, 6]
])

weights = np.array([0.1, 0.4, 0.3, 0.2])

# The indices of the attributes (zero-based) that are considered beneficial.
# Those indices not mentioned are assumed to be cost attributes.
benefit_attributes = set([0, 1, 2, 3])

# Display the raw data we have
d = pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
print(d)
print("")

# Step 1 - Normalizing the ratings
m = len(raw_data)
n = len(attributes)
divisors = np.empty(n)
for j in range(n):
    column = raw_data[:,j]
    divisors[j] = np.sqrt(column @ column)

raw_data = raw_data / divisors

columns = ["X_{%d}" % j for j in range(n)]
dd = pd.DataFrame(data=raw_data, index=candidates, columns=columns)
print(dd)
print("")

# Step 2 - Calculating the Weighted Normalized Ratings

raw_data *= weights
ddd = pd.DataFrame(data=raw_data, index=candidates, columns=columns)
print(ddd)
print("")

# Step 3 - Identifying PIS ( A∗ ) and NIS ( A− )
a_pos = np.zeros(n)
a_neg = np.zeros(n)
for j in range(n):
    column = raw_data[:, j]
    max_val = np.max(column)
    min_val = np.min(column)

    # See if we want to maximize benefit or minimize cost (for PIS)
    if j in benefit_attributes:
        a_pos[j] = max_val
        a_neg[j] = min_val
    else:
        a_pos[j] = min_val
        a_neg[j] = max_val

dddd = pd.DataFrame(data=[a_pos, a_neg], index=["A^*", "A^-"], columns=columns)
print(dddd)
print(" ")

# Step 4 and 5 - Calculating Separation Measures and Similarities to PIS

sp = np.zeros(m)
sn = np.zeros(m)
cs = np.zeros(m)

for i in range(m):
    diff_pos = raw_data[i] - a_pos
    diff_neg = raw_data[i] - a_neg
    sp[i] = np.sqrt(diff_pos @ diff_pos)
    sn[i] = np.sqrt(diff_neg @ diff_neg)
    cs[i] = sn[i] / (sp[i] + sn[i])

ddddd = pd.DataFrame(data=zip(sp, sn, cs), index=candidates, columns=["S^*", "S^-", "C^*"])
print(ddddd)
print(" ")

# Step 6 - Ranking the candidates/alternatives
cs_order = rank_according_to(cs)
sp_order = rank_according_to(sp)
sn_order = rank_according_to(sn)

dddddd = pd.DataFrame(data=zip(cs_order, sp_order, sn_order), index=range(1, m + 1), columns=["C^*", "S^*", "S^-"])
print(dddddd)
print(" ")

print("The best candidate/alternative according to C* is " + cs_order[0])
print("The preferences in descending order are " + ", ".join(cs_order) + ".")
