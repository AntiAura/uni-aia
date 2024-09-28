import csv
import os
import tqdm
import pickle

from deepface import DeepFace

""" # Load dataset
DATA_PATH = "data"
files = []
with open(os.path.join(DATA_PATH, "train.csv"), "r", encoding="utf8") as f:
    reader = csv.reader(f)
    reader.__next__()  # Skip header
    for row in reader:
        files.append(row)

results = []
for file in tqdm.tqdm(files[:5000]):
    result = {
        "labels": file[1:],
    }
    try:
        prediction = DeepFace.analyze(os.path.join(DATA_PATH, file[0]), silent=True)[0]
        result["predictions"] = prediction
        result["error"] = False
    except Exception as e:
        result["error"] = True
    results.append(result)

with open("results.pkl", "wb") as f:
    pickle.dump(results, f) """

# Load results
with open("results.pkl", "rb") as f:
    results = pickle.load(f)
