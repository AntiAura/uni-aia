import csv
import os
import tqdm
import pandas as pd
from deepface import DeepFace

# Load dataset
DATA_PATH = "data"
files = []
with open(os.path.join(DATA_PATH, "train.csv"), "r", encoding="utf8") as f:
    reader = csv.reader(f)
    reader.__next__()  # Skip header
    for row in reader:
        files.append(row)

results = []
for file in tqdm.tqdm(files[:20000]):
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

# Filter errors from data
results = [result for result in results if not result["error"]]

# Create final dataframe
df = pd.DataFrame()

# Extract true labels and predictions for race
y_true_race = [result["labels"][2] for result in results]
y_pred_race = [result["predictions"]["dominant_race"] for result in results]

# Map true values to equivalent values in the dataset (predictions)
old = y_true_race
y_true_race = []
for i in old:
    if i == "Black":
        y_true_race.append("black")
    elif i == "White":
        y_true_race.append("white")
    elif i == "East Asian":
        y_true_race.append("asian")
    elif i == "Indian":
        y_true_race.append("indian")
    elif i == "Middle Eastern":
        y_true_race.append("middle eastern")
    elif i == "Latino_Hispanic":
        y_true_race.append("latino hispanic")
    elif i == "Southeast Asian":
        y_true_race.append("asian")
    else:
        raise ValueError("Unknown value")

# Add true and predicted race to the dataframe
df["race_true"] = y_true_race
df["race_pred"] = y_pred_race

# Add sex to the dataframe
df["sex_true"] = [
    "female" if result["labels"][1] == "Female" else "male" for result in results
]
df["sex_pred"] = [
    "female" if result["predictions"]["dominant_gender"] == "Woman" else "male"
    for result in results
]

# Age for labels is one of ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
# while predictions are exact ages, so we need to map the predictions to the labels for uniformity
df["age_true"] = [result["labels"][0] for result in results]
df["age_pred"] = [
    (
        "0-2"
        if 0 <= result["predictions"]["age"] <= 2
        else (
            "3-9"
            if 3 <= result["predictions"]["age"] <= 9
            else (
                "10-19"
                if 10 <= result["predictions"]["age"] <= 19
                else (
                    "20-29"
                    if 20 <= result["predictions"]["age"] <= 29
                    else (
                        "30-39"
                        if 30 <= result["predictions"]["age"] <= 39
                        else (
                            "40-49"
                            if 40 <= result["predictions"]["age"] <= 49
                            else (
                                "50-59"
                                if 50 <= result["predictions"]["age"] <= 59
                                else (
                                    "60-69"
                                    if 60 <= result["predictions"]["age"] <= 69
                                    else "more than 70"
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    for result in results
]

# Add predicted emotions to the dataframe
df["emotion_pred"] = [result["predictions"]["dominant_emotion"] for result in results]

# Save the dataframe
df.to_csv("df.csv", index=False)
