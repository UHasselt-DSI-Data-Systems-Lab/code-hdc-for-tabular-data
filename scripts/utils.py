from pathlib import Path
import yaml
import re
import os


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
