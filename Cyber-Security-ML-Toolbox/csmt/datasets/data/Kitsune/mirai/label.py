import numpy as np

data = "Mirai_labels" + ".csv"
out = "labels.npy"

f = open(data, "r")

label = []
for i in f:
    s = i.strip().split(',')[1]
    if s == '"x"':
        continue
    label.append(int(s))

print(len(label))
print(label[:20])

np.save(out, label)

