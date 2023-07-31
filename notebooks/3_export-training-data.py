# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import pyrootutils
from pykanto.utils.io import load_dataset, save_subset
from pykanto.utils.paths import ProjDirs
from sklearn.model_selection import train_test_split

# ──── SETTINGS ─────────────────────────────────────────────────────────────────

DATASET_ID = "great-tit-songtypes"
BASE_DATASET_ID = "great-tit-hits"
PROJECT_ROOT = pyrootutils.find_root()

# Create a ProjDirs object for the project
S_DATA = PROJECT_ROOT / "data" / "segmented" / BASE_DATASET_ID
DIRS = ProjDirs(PROJECT_ROOT, S_DATA, BASE_DATASET_ID)


# ──── LOAD DATASET ────────────────────────────────────────────────────────────

# Open an existing dataset
out_dir = DIRS.DATASET.parent / f"{DATASET_ID}.db"
dataset = load_dataset(out_dir, DIRS)

# Minimum cluster size to include a song type in the model:
min_cluster_size = 3
sample = 3


# ──── SUBSAMPLE DATASET FOR MODEL TRAINING ─────────────────────────────────────

"""
This will create a unique song class label for each vocalisation in the dataset
(a combination of the ID and the label).
"""

# from dataset.data get only the 'class_label' for which there is at least
# min_cluster_size 'class_id'.

df = dataset.data.groupby("class_label").filter(
    lambda x: len(x["class_id"].unique()) >= min_cluster_size
)

# randomly sample 'sample' songs per class_id
df_sub = df.groupby(["class_id"]).sample(sample, random_state=42, replace=True)


df = dataset.data.groupby(["class_label"]).filter(
    lambda x: len(x) >= min_cluster_size
)


# count number of rows per class_id in df_sub
dataset.data["class_label"].value_counts()
df_sub["class_label"].value_counts()
df_sub["class_id"].value_counts()


# Add spectrogram files
df_sub["spectrogram"] = dataset.files["spectrogram"]

# Print info
n_rem = len(set(dataset.data["class_label"].dropna())) - len(
    set(df_sub["class_label"].dropna())
)
print(
    f"Removed {n_rem} song types (songs types with < {min_cluster_size} songs)"
)


# ──── TRAIN / TEST SPLIT AND EXPORT ────────────────────────────────────────────

train, test = train_test_split(
    df_sub,
    test_size=0.3,
    shuffle=True,
    stratify=df_sub["class_label"],
    random_state=42,
)

train["class_label"].value_counts()

out_dir = dataset.DIRS.RESOURCES / "ML"
train_dir, test_dir = out_dir / "train", out_dir / "test"

for dset, dname in zip([train, test], ["train", "test"]):
    to_export = (
        dset.groupby("class_label")["spectrogram"]  # type: ignore
        .apply(list)
        .to_dict()
        .items()
    )
    save_subset(train_dir, test_dir, dname, to_export)
