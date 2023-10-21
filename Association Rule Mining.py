#### Association Rule Mining
# Demo using mlxtend
# Using Apriori Algorithm

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

dataset = [
    ["Milk", "Onion", "Nutmeg", "Kidney Beans", "Eggs", "Yogurt"],
    ["Dill", "Onion", "Nutmeg", "Kidney Beans", "Eggs", "Yogurt"],
    ["Milk", "Apple", "Kidney Beans", "Eggs"],
    ["Milk", "Unicorn", "Corn", "Kidney Beans", "Yogurt"],
    ["Corn", "Onion", "Onion", "Kidney Beans", "Ice cream", "Eggs"]
]

# The * symbol is used to print the list elements in a single line with space.
# print in new line
print(*dataset, sep="\n")

### Apriori Algorithm

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
#print(*te_ary, sep="\n")
print(te.columns_)

df = pd.DataFrame(te_ary, columns=te.columns_)
print(df.head())

frequent_itemsets = apriori(df, min_support = 0.6, use_colnames = True)
# when use_colnames = False, it will use column index as itemsets
print(frequent_itemsets)

### Association Rules Mining
res = association_rules(frequent_itemsets, metric = "confidence", min_threshold=0.7)
#print(res)

res1 = res[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print(res1.head())

res2 = res[res['confidence'] >= 1]
print(res2.head())