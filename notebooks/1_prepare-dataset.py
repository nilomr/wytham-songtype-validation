# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyrootutils
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs, link_project_data

from src.io import read_labels

# ──── SETTINGS ────────────────────────────────────────────────────────────────


DATASET_ID = "great-tit-hits"
PROJECT_ROOT = pyrootutils.find_root()
# DATA_LOCATION = Path("/media/nilomr/SONGDATA/wytham-great-tit")
DATA_LOCATION = Path("/data/zool-songbird/shil5293/projects/great-tit-hits-setup/data")

# Create symlink from project to data if it doesn't exist already:
if not (PROJECT_ROOT / "data").exists():
    link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")

# Create a ProjDirs object for the project
RAW_DATA = PROJECT_ROOT / "data" / "raw" / DATASET_ID
RAW_DATA.mkdir(exist_ok=True, parents=True)
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, DATASET_ID, mkdir=True)


# ──── LOAD DATASET ────────────────────────────────────────────────────────────

# Open an existing dataset
out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(out_dir, DIRS)


# create a copy of the dataset (only the database, pointers remain the same)
DATASET_ID_N = "great-tit-songtypes"

# Define the new file path
out_dir_n = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID_N}.db"
# Copy the file to the new location
shutil.copy(DIRS.DATASET, out_dir_n)

# Load the dataset from the new file
dataset = load_dataset(out_dir_n, DIRS)

# NOTE: DIRS now points to the new dataset


# ──── FIX MISLABELLED SONGS ──────────────────────────────────────────────────

# Manually fix wrong labels (detected during the manual check) in class_id
# column (dataset.data is a pandas dataframe)

# class_20221W81_4_20221W81_20220406_050000_48505408 → class_20221W81_9
# class_20211B72_3 → class_20211B72_2
# merge class_20221O56_0 & class_20221O56_1 (name = 20221O56_0)

dataset.data.loc["20221W81_20220406_050000_48505408", "class_id"] = "20221W81_9"

dataset.data["class_id"] = dataset.data["class_id"].replace(
    {
        "20211B72_3": "20211B72_2",
        "20221O56_1": "20221O56_0",
        "20211SW70_3": "20211SW70_0",
    }
)

# ──── ASSIGN NEW LABELS ──────────────────────────────────────────────────────


# Read in manually assigned labels
lfile = DIRS.RESOURCES / "manual_labels_raw.txt"
labels_dict = read_labels(lfile)

# Run some basic checks
existing_labels = set(dataset.data.class_id.unique())
wrong_labels = (
    set(label for labels in labels_dict.values() for label in labels)
    - existing_labels
)
if wrong_labels:
    raise ValueError(
        f"Labels in labels_dict that are not in dataset.data.class_id: {wrong_labels}"
    )

new_labels = [label for labels in labels_dict.values() for label in labels]
old_labs = dataset.data.class_id.unique().tolist()
missing_labels = [l for l in old_labs if l not in new_labels]

for l in missing_labels:
    if not ("20211O115" in l or "20221MP32" in l):
        raise ValueError(f"Label {l} is missing in the manual labels.")

# Sort and plot
labels_dict = dict(sorted(labels_dict.items(), key=lambda item: len(item[1])))
pd.Series([len(v) for v in labels_dict.values()]).plot(kind="bar")


# create a dataframe with the class label for each unique class_id
class_labels = (
    dataset.data.groupby("class_id").first()["class_label"].reset_index()
)


# save the class labels in a CSV file in DIRS.RESOURCES
class_labels.to_csv(DIRS.RESOURCES / "manual_labels.csv", index=False)


# ──── ADD LABELS TO DATASET ──────────────────────────────────────────────────

# Open manually assigned cluster labels and add them to the dataset
df_labs = pd.read_csv(DIRS.RESOURCES / "manual_labels.csv")

# map class_id to class_label
if "class_label" in dataset.data.columns:
    dataset.data.drop(columns=["class_label"], inplace=True)
dataset.data.insert(2, "class_label", np.nan)
for class_id, class_labels in labels_dict.items():
    dataset.data.loc[
        dataset.data.class_id.isin(class_labels), "class_label"
    ] = class_id

# print n of unique class_label
print(f"Number of unique class_label: {dataset.data.class_label.nunique()}")

# save dataset to disk
dataset.save_to_disk()
