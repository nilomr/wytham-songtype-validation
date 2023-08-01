# ──── DESCRIPTION ────────────────────────────────────────────────────────────

"""
Merge unstable manual clusters as identified by the ResNet-based classifier.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import pandas as pd
import pyrootutils
from pykanto.utils.io import load_dataset, save_subset
from pykanto.utils.paths import ProjDirs
from sklearn.model_selection import train_test_split

# ──── SETTINGS ────────────────────────────────────────────────────────────────

DATASET_ID = "great-tit-songtypes"
BASE_DATASET_ID = "great-tit-hits"
PROJECT_ROOT = pyrootutils.find_root()

# Create a ProjDirs object for the project
S_DATA = PROJECT_ROOT / "data" / "segmented" / BASE_DATASET_ID
DIRS = ProjDirs(PROJECT_ROOT, S_DATA, BASE_DATASET_ID)


# ──── LOAD DATASET ────────────────────────────────────────────────────────────

df_labs = pd.read_csv(DIRS.RESOURCES / "manual_labels.csv")
nan_mask = df_labs.isna().any(axis=1)
df_labs = df_labs[~nan_mask]


def replace_labels(df_labs, numbers):
    # Find the rows where the class_label column contains the given numbers
    mask = df_labs["class_label"].str.contains(
        "|".join(str(n) for n in numbers)
    )

    # Extract the numbers from the class_label column and find the lowest number
    lowest_number = (
        df_labs.loc[mask, "class_label"]
        .str.extract(f"({'|'.join(str(n) for n in numbers)})", expand=False)
        .astype(int)
        .astype(str)
        .min()
    )

    # Find the label with the lowest number
    lowest_label = df_labs.loc[
        df_labs["class_label"].str.contains(lowest_number), "class_label"
    ].iloc[0]
    df_labs.loc[mask, "class_label"] = lowest_label

    return df_labs


df_labs = replace_labels(df_labs, [455, 377, 454])
df_labs = replace_labels(df_labs, [514, 453, 515, 554])

df_labs.to_csv(DIRS.RESOURCES / "checked_manual_labels.csv", index=False)
df_labs.to_csv(
    DIRS.PROJECT.parent
    / "birdsong-demography"
    / "data"
    / "derived"
    / "manual_labels.csv",
    index=False,
)
