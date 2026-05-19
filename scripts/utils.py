from math import sqrt
import os
import random
from pathlib import Path
import yaml
import re

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config.yaml") -> dict:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


def get_absolute_path(relative_path: str) -> str:
    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent
    return str(root_dir / relative_path)


def duplicate_specific_quotes(text):
    # (?<! )   -> Not preceded by a space
    # (?<!^ )  -> Not at the very start of the string
    # '        -> The single quote to match
    # (?=[^ ]) -> Followed by a character that is NOT a space
    #             (This also ensures it's not the end of the string)

    pattern = r"(?<! )(?<!^)'(?=[^ ])"
    return re.sub(pattern, "''", text)


def precision_recall(n_truth: int, n_retrieved: int, inter: int):
    # Precision
    if n_retrieved == 0:
        precision = 1.0
    else:
        precision = inter / n_retrieved

    # Recall
    if n_truth == 0:
        recall = 1.0
    else:
        recall = inter / n_truth

    return round(precision, 2), round(recall, 2)


def save_dataframe(df, file_path):
    file_exists = os.path.isfile(file_path)

    # mode='a' appends to the file
    # header=not file_exists ensures the column names are only written once
    df.to_csv(file_path, mode="a", index=False, header=not file_exists)


def stdev_by_dim(m, n, dim):
    """
    Predicted standard deviation of the similarity for row retrieval.
    """
    # return sqrt((4 * m + 2 * n + 9) / (2 * m * dim))
    return sqrt(
        (8 * dim * m - 8 * dim + 8 * m + 4 * dim * n + n - 8) / (4 * m * dim * dim)
    )


def predicted_by_dim(m, n, dim):
    """
    Advised threshold value tau(m,n,dim) for equality predicates.
    """
    return round(sqrt(n / m) - stdev_by_dim(m, n, dim), 2)
