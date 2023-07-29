# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyrootutils
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs, link_project_data

# ──── SETTINGS ────────────────────────────────────────────────────────────────


DATASET_ID = "great-tit-hits"
PROJECT_ROOT = pyrootutils.find_root()
DATA_LOCATION = Path("/media/nilomr/SONGDATA/wytham-great-tit")
# DATA_LOCATION = Path("/data/zool-songbird/shil5293/data/wytham-great-tit")

# Create symlink from project to data if it doesn't exist already:
if not (PROJECT_ROOT / "data").exists():
    link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")

# Create a ProjDirs object for the project
RAW_DATA = PROJECT_ROOT / "data" / "raw" / DATASET_ID
RAW_DATA.mkdir(exist_ok=True)
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
    {"20211B72_3": "20211B72_2", "class_20221O56_1": "20221O56_0"}
)

# ──── ASSIGN NEW LABELS ──────────────────────────────────────────────────────

# create 200 fake classes and assign all the rows in dataset.data to one of them,
# following a broken stick distribution and based on the class_id column

n_classes = 200
stick = np.random.beta(1, np.arange(1, n_classes + 1))
stick /= stick.sum()

# assign each class_id in dataset.data to one of the classess so that the
# frequency of each class is proportional to the broken stick distribution
class_ids = dataset.data.class_id.unique()
class_labels = pd.DataFrame(
    {
        "class_id": class_ids,
        "class_label": np.random.choice(
            np.arange(1, n_classes + 1), size=len(class_ids), p=stick
        ),
    }
)

# plot the frequency of the new labels
class_labels.class_label.value_counts().plot(kind="bar")

# convert to strings prefixed with "class_"
class_labels.class_label = class_labels.class_label.apply(
    lambda x: f"class_{x}"
)

# save the class labels in a CSV file in DIRS.RESOURCES
class_labels.to_csv(DIRS.RESOURCES / "manual_labels.csv", index=True)


# ──── ADD LABELS TO DATASET ──────────────────────────────────────────────────

# Open manually assigned cluster labels and add them to the dataset
df_labs = pd.read_csv(DIRS.RESOURCES / "manual_labels.csv", index_col=0)

# map class_id to class_label
dataset.data.insert(
    2,
    "class_label",
    dataset.data["class_id"].map(df_labs.set_index("class_id")["class_label"]),
)

# save dataset to disk
dataset.save_to_disk()
