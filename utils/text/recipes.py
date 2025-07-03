from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import tqdm

from .files import get_files

DEFAULT_SPEAKER_NAME = "default_speaker"


def read_metadata(
    path: Path, metafile: str, format: str, n_workers: Optional[int] = 1
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Reads metadata from different dataset formats and returns text and speaker dictionaries.

    Args:
        path: The base directory path of the dataset.
        metafile: The metadata file name.
        format: The format of the dataset ('ljspeech', 'ljspeech_multi', 'vctk', 'pandas').
        n_workers: Number of worker processes to use (for 'vctk' format).

    Returns:
        A tuple containing:
            - text_dict: A dictionary mapping file IDs to text.
            - speaker_dict: A dictionary mapping file IDs to speaker names.

    Raises:
        ValueError: If an unsupported format is specified.
    """
    if format == "ljspeech":
        return read_ljspeech_format(path / metafile, multispeaker=False)
    elif format == "ljspeech_multi":
        return read_ljspeech_format(path / metafile, multispeaker=True)
    elif format == "vctk":
        return read_vctk_format(path, n_workers=n_workers or 1)
    elif format == "pandas":
        return read_pandas_format(path / metafile)
    else:
        raise ValueError(
            f"Unsupported format '{format}'. Supported formats are: "
            f"'ljspeech', 'ljspeech_multi', 'vctk', 'pandas'."
        )


def read_ljspeech_format(
    path: Path, multispeaker: bool = False
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Reads metadata in LJSpeech format.

    Args:
        path: Path to the metadata file.
        multispeaker: Whether the dataset includes multiple speakers.

    Returns:
        A tuple containing:
            - text_dict: A dictionary mapping file IDs to text.
            - speaker_dict: A dictionary mapping file IDs to speaker names.

    Raises:
        ValueError: If the metadata file does not exist.
    """
    if not path.is_file():
        raise ValueError(
            f"Could not find metafile: {path}. "
            "Please make sure that you set the correct path and metafile name!"
        )
    text_dict = {}
    speaker_dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            split = line.split("|")
            if multispeaker and len(split) > 2:
                speaker_name = split[-2]
            else:
                speaker_name = DEFAULT_SPEAKER_NAME
            file_id, text = split[0], split[-1]
            text_dict[file_id] = text.strip()
            speaker_dict[file_id] = speaker_name
    return text_dict, speaker_dict


def read_vctk_format(
    path: Path, n_workers: int = 1, extension: str = ".txt"
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Reads metadata in VCTK format.

    Args:
        path: The base directory path containing the text files.
        n_workers: Number of worker processes to use.
        extension: File extension of text files.

    Returns:
        A tuple containing:
            - text_dict: A dictionary mapping file IDs to text.
            - speaker_dict: A dictionary mapping file IDs to speaker names.
    """
    files = get_files(path, extension=extension)
    text_dict = {}
    speaker_dict = {}

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.imap_unordered(read_line, files)
            for file, text in tqdm.tqdm(results, total=len(files)):
                text_id = file.stem
                speaker_id = file.parent.stem
                text_dict[text_id] = text.strip()
                speaker_dict[text_id] = speaker_id
    else:
        for file in tqdm.tqdm(files, total=len(files)):
            _, text = read_line(file)
            text_id = file.stem
            speaker_id = file.parent.stem
            text_dict[text_id] = text.strip()
            speaker_dict[text_id] = speaker_id
    return text_dict, speaker_dict


def read_pandas_format(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Reads metadata from a pandas DataFrame stored in a file.

    Args:
        path: Path to the metadata file.

    Returns:
        A tuple containing:
            - text_dict: A dictionary mapping file IDs to text.
            - speaker_dict: A dictionary mapping file IDs to speaker names.

    Raises:
        ValueError: If the metadata file does not exist.
    """
    if not path.is_file():
        raise ValueError(
            f"Could not find metafile: {path}. "
            "Please make sure that you set the correct path and metafile name!"
        )
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    text_dict = pd.Series(df["text"].values, index=df["file_id"]).to_dict()
    speaker_dict = pd.Series(df["speaker_id"].values, index=df["file_id"]).to_dict()
    return text_dict, speaker_dict


def read_line(file: Path) -> Tuple[Path, str]:
    """
    Reads the first line from a text file.

    Args:
        file: Path to the text file.

    Returns:
        A tuple containing:
            - file: The Path object of the file.
            - line: The first line of the file.
    """
    with open(file, encoding="utf-8") as f:
        line = f.readline()
    return file, line
