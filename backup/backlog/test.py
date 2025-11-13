import json
with open("data/dataset.json") as f:
    data = json.load(f)
print(len(data))
print(data[0].keys())
print(data[0])