"""The PlasmoCount dataset contains annotations for red blood cells in Giemsa-stained
blood smear images of malaria infected blood. The smears are mostly from *P. falciparum*
infected human blood, and were contributed by six different research centers.

The dataset contains 399 brightfield images with roughly 37,500 annotated cells, labelled
as 'uninfected', 'infected' or 'not_sure'. The annotations are *bounding boxes*, not
pixel-wise segmentation masks. The boxes are rasterized into label images here, so that the
data can be used with the segmentation functionality of `torch_em`: each box is filled as a
rectangle, either with a unique instance id ('instances') or with its category id
('semantic'). Note that these labels are only a coarse approximation of the actual cell
shapes.

The images vary in shape and are stored as either 8bit or 16bit RGB tif files.

In addition to the infection status, the repository contains parasite life stage annotations
for the infected cells in 'training_stages.json' and 'test_stages.json'. These are not used
for the labels here, because they contain the (partially disagreeing) labels of multiple
annotators per cell, with a label vocabulary that differs between the two splits. They remain
available in the downloaded data, together with the original box annotations in 'training.json'
and 'test.json'.

The dataset is located at https://data.mendeley.com/datasets/j55fyhtxn4/1.
This dataset is from the publication:
- Davidson et al. (2021): https://doi.org/10.1017/S2633903X21000015
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from warnings import warn
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


DOI, VERSION = "j55fyhtxn4", 1

URL = f"https://data.mendeley.com/public-api/datasets/{DOI}/files?folder_id=root&version={VERSION}"
"""The Mendeley Data repository does not provide an archive with all files, so the individual
files have to be downloaded. This url lists them, together with a sha256 checksum for each file,
which is used to verify the download. The listing is cached in the data folder after the first
download, so that the data can be used without an internet connection afterwards.
"""

CATEGORIES = ("uninfected", "infected", "not_sure")
"""The cell categories annotated in PlasmoCount. The semantic label id of a category is its
position in this tuple plus one, i.e. 1 for 'uninfected' and 3 for 'not_sure' (0 is background).
Cells are annotated as 'not_sure' if the annotators could not determine the infection status.
"""

ANNOTATION_FILES = {"train": "training.json", "test": "test.json"}
"""@private
"""

STAGE_ANNOTATION_FILES = {"train": "training_stages.json", "test": "test_stages.json"}
"""The files with the parasite life stage annotations for the infected cells. See the module
docstring for why they are not used for the labels.
"""

LABEL_CHOICES = ("instances", "semantic")
"""@private
"""


def _get_file_list(data_dir, download):
    """Get the list of files in the repository, from the local cache if it is available."""
    list_path = os.path.join(data_dir, "file_list.json")
    if os.path.exists(list_path):
        with open(list_path) as f:
            return json.load(f)

    if not download:
        raise RuntimeError(f"Cannot find the data at {data_dir}, but download was set to False")

    import requests

    response = requests.get(URL)
    response.raise_for_status()
    file_list = [
        {
            "filename": entry["filename"],
            "url": entry["content_details"]["download_url"],
            "checksum": entry["content_details"]["sha256_hash"],
        }
        for entry in response.json()
    ]

    with open(list_path, "w") as f:
        json.dump(file_list, f)

    return file_list


def _rasterize_annotations(objects, shape):
    """Rasterize the bounding box annotations of a single image into label images."""
    category_ids = {name: i + 1 for i, name in enumerate(CATEGORIES)}

    boxes, box_categories = [], []
    for obj in objects:
        bbox = obj["bounding_box"]
        boxes.append(
            [bbox["minimum"]["r"], bbox["minimum"]["c"], bbox["maximum"]["r"], bbox["maximum"]["c"]]
        )
        box_categories.append(category_ids[obj["category"]])

    boxes = np.array(boxes, dtype="int32").reshape(-1, 4)
    box_categories = np.array(box_categories, dtype="uint8")

    instances = np.zeros(shape, dtype="int32")
    semantic = np.zeros(shape, dtype="uint8")

    # Draw the boxes from large to small, so that small cells stay visible where boxes overlap.
    # The instance id of a box is its position in the annotation file plus one.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in np.argsort(-areas):
        min_r, min_c, max_r, max_c = boxes[i]
        instances[min_r:max_r, min_c:max_c] = i + 1
        semantic[min_r:max_r, min_c:max_c] = box_categories[i]

    return instances, semantic


def _preprocess(data_dir, split):
    """Rasterize the box annotations of a split into label tif files.

    The images are used as they are, so only the labels have to be created.
    """
    label_dirs = {choice: os.path.join(data_dir, "labels", split, choice) for choice in LABEL_CHOICES}
    for label_dir in label_dirs.values():
        os.makedirs(label_dir, exist_ok=True)

    with open(os.path.join(data_dir, ANNOTATION_FILES[split])) as f:
        annotations = json.load(f)

    n_missing = 0
    for annotation in tqdm(annotations, desc=f"Preprocessing PlasmoCount '{split}'"):
        image_path = os.path.join(data_dir, annotation["image"]["pathname"])
        if not os.path.exists(image_path):
            # A few of the images that are referenced in the annotations are not part of the repository.
            n_missing += 1
            continue

        fname = f"{os.path.splitext(os.path.basename(image_path))[0]}.tif"
        label_paths = {choice: os.path.join(label_dir, fname) for choice, label_dir in label_dirs.items()}
        if all(os.path.exists(label_path) for label_path in label_paths.values()):
            continue

        # The annotations don't contain the image shape, so we read it from the image header.
        shape = imageio.improps(image_path).shape[:2]
        labels = _rasterize_annotations(annotation["objects"], shape)
        for choice, label in zip(LABEL_CHOICES, labels):
            imageio.imwrite(label_paths[choice], label, compression="zlib")

    if n_missing > 0:
        warn(
            f"{n_missing} of the {len(annotations)} images that are annotated in the '{split}' split are "
            "not part of the PlasmoCount repository. They are skipped."
        )

    return label_dirs


def get_plasmocount_data(
    path: Union[os.PathLike, str], split: Literal["train", "test"] = "train", download: bool = False,
) -> str:
    """Download the PlasmoCount dataset and rasterize the box annotations.

    The data is downloaded file by file, because the repository does not provide an archive
    with all files. This means that the download of the roughly 8.4 GB of data takes a while.
    It can be interrupted and will be resumed on the next call.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the downloaded and preprocessed data.
    """
    assert split in ANNOTATION_FILES, \
        f"'{split}' is not a valid split. Choose either 'train' or 'test'."

    data_dir = os.path.join(str(path), "plasmocount")
    os.makedirs(data_dir, exist_ok=True)

    file_list = _get_file_list(data_dir, download)
    for entry in tqdm(file_list, desc="Download PlasmoCount data"):
        util.download_source(
            path=os.path.join(data_dir, entry["filename"]),
            url=entry["url"],
            download=download,
            checksum=entry["checksum"],
        )

    _preprocess(data_dir, split)

    return data_dir


def get_plasmocount_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PlasmoCount data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'test'.
        label_choice: The type of labels to use. Either 'instances' or 'semantic'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert label_choice in LABEL_CHOICES, \
        f"'{label_choice}' is not a valid label choice. Choose either 'instances' or 'semantic'."

    data_dir = get_plasmocount_data(path, split, download)

    label_paths = natsorted(glob(os.path.join(data_dir, "labels", split, label_choice, "*.tif")))
    assert len(label_paths) > 0, f"No labels were found for the split '{split}'."

    # The images of both splits are stored in the same folder, so we match them to the
    # labels of this split via the filename.
    raw_paths = [os.path.join(data_dir, os.path.basename(p)) for p in label_paths]

    return raw_paths, label_paths


def get_plasmocount_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    offsets: Union[List[List[int]], None] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> Dataset:
    """Get the PlasmoCount dataset for segmentation of malaria infected blood cells.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        label_choice: The type of labels to use. Either 'instances' for one id per annotated
            cell, or 'semantic' for the cell categories, see `CATEGORIES` for the label ids.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_plasmocount_paths(path, split, label_choice, download)

    if label_choice == "semantic":
        assert not any((offsets is not None, boundaries, binary)), \
            "'offsets', 'boundaries' and 'binary' are only supported for instance labels."
    else:
        kwargs, _ = util.add_instance_label_transform(
            kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
        )

    # The images are RGB and vary in shape, so we use an image collection dataset. It loads the
    # data lazily, which also avoids running out of file handles for the images of this dataset.
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", False)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_plasmocount_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    offsets: Union[List[List[int]], None] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the PlasmoCount dataloader for segmentation of malaria infected blood cells.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        label_choice: The type of labels to use. Either 'instances' for one id per annotated
            cell, or 'semantic' for the cell categories, see `CATEGORIES` for the label ids.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_plasmocount_dataset(
        path, patch_shape, split=split, label_choice=label_choice, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
