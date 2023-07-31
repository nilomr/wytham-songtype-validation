import re
from typing import Dict, List


def read_labels(file_path: str) -> Dict[str, List[str]]:
    """Reads in manually assigned labels from a file and returns a dictionary of labels.

    Args:
        file_path (str): The path to the file containing the labels.

    Returns:
        dict: A dictionary of labels, where the keys are the IDs and the values
            are lists of labels.
    """
    data_dict = {}
    current_key = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_key(line):
                current_key = line
                if current_key in data_dict.keys():
                    raise ValueError(f"Key already exists: {current_key}")
                data_dict[current_key] = []
            elif is_label(line):
                if current_key is not None:
                    data_dict[current_key].append(line)
                else:
                    raise ValueError(f"Invalid value: {line}")
            else:
                raise ValueError(f"Invalid value: {line}")

    return data_dict


def is_key(line: str) -> bool:
    """Checks if a line is a valid key.

    Args:
        line (str): The line to check.

    Returns:
        bool: True if the line is a valid key, False otherwise.
    """
    return bool(re.match(r"\d+-[A-Z\d]+(-[A-Z\d]+)*", line))


def is_label(line: str) -> bool:
    """Checks if a line is a valid label.

    Args:
        line (str): The line to check.

    Returns:
        bool: True if the line is a valid label, False otherwise.
    """
    return bool(
        re.match(r"202[0-2][12][A-Z]{0,2}\w{1,3}(_?[A-Z]?\d{1,2})?", line)
    )
