"""The BBBC041 dataset contains annotations for blood cells in Giemsa-stained
blood smear images of *P. vivax* (malaria) infected human blood.

The dataset consists of 1,328 brightfield images with roughly 80,000 annotated cells.
The annotations are *bounding boxes* with a class label for each cell, not pixel-wise
segmentation masks. The boxes are rasterized into label images here, so that the data
can be used with the segmentation functionality of `torch_em`: each box is filled as a
rectangle, either with a unique instance id ('instances') or with its category id
('semantic'). Note that these labels are only a coarse approximation of the actual cell
shapes. The original box annotations remain available in the 'training.json' and
'test.json' files of the downloaded data, so that it can also be used for detection.

The images are split into 1,208 training images (1200 x 1600 pixels) and 120 test images
(1383 x 1944 pixels), following the official split shipped with the data.

The dataset is located at https://bbbc.broadinstitute.org/BBBC041.
This dataset is from the following publications:
- Hung et al. (2020): https://doi.org/10.1002/cyto.a.23964
- Ljosa et al. (2012): https://doi.org/10.1038/nmeth.2083
Please cite them if you use this dataset in your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip"
CHECKSUM = None  # The BBBC does not publish a checksum for this archive.

CATEGORIES = (
    "red blood cell",
    "leukocyte",
    "gametocyte",
    "ring",
    "trophozoite",
    "schizont",
    "difficult",
)
"""The cell categories annotated in BBBC041. The semantic label id of a category is its
position in this tuple plus one, i.e. 1 for 'red blood cell' and 7 for 'difficult'
(0 is background). Cells are annotated as 'difficult' if their state is unclear.
"""

INFECTED_CATEGORIES = ("gametocyte", "ring", "trophozoite", "schizont")
"""The categories that correspond to infected cells, i.e. the life stages of the parasite.
"""

ANNOTATION_FILES = {"train": "training.json", "test": "test.json"}
"""@private
"""

LABEL_CHOICES = ("instances", "semantic")
"""@private
"""


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

    for annotation in tqdm(annotations, desc=f"Preprocessing BBBC041 '{split}'"):
        # The pathnames in the annotation files are given relative to the data directory,
        # but start with a separator, e.g. '/images/<uuid>.png'.
        image_path = os.path.join(data_dir, annotation["image"]["pathname"].lstrip("/"))
        fname = f"{os.path.splitext(os.path.basename(image_path))[0]}.tif"
        label_paths = {choice: os.path.join(label_dir, fname) for choice, label_dir in label_dirs.items()}
        if all(os.path.exists(label_path) for label_path in label_paths.values()):
            continue

        shape = annotation["image"]["shape"]
        labels = _rasterize_annotations(annotation["objects"], (shape["r"], shape["c"]))
        for choice, label in zip(LABEL_CHOICES, labels):
            imageio.imwrite(label_paths[choice], label, compression="zlib")

    return label_dirs


def get_bbbc041_data(
    path: Union[os.PathLike, str], split: Literal["train", "test"] = "train", download: bool = False,
) -> str:
    """Download the BBBC041 dataset and rasterize the box annotations.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the downloaded and preprocessed data.
    """
    assert split in ANNOTATION_FILES, \
        f"'{split}' is not a valid split. Choose either 'train' or 'test'."

    data_dir = os.path.join(str(path), "malaria")
    if not os.path.exists(data_dir):
        os.makedirs(str(path), exist_ok=True)
        zip_path = os.path.join(str(path), "malaria.zip")
        util.download_source(zip_path, URL, download, checksum=CHECKSUM)
        util.unzip(zip_path, str(path))

    _preprocess(data_dir, split)

    return data_dir


def get_bbbc041_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the BBBC041 data.

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

    data_dir = get_bbbc041_data(path, split, download)

    label_paths = natsorted(glob(os.path.join(data_dir, "labels", split, label_choice, "*.tif")))
    assert len(label_paths) > 0, f"No labels were found for the split '{split}'."

    # The images of both splits are stored in the same folder, and the extension differs
    # between the splits, so we match the images to the labels via the filename.
    images = {
        os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(data_dir, "images", "*"))
    }
    raw_paths = [images[os.path.splitext(os.path.basename(p))[0]] for p in label_paths]

    return raw_paths, label_paths


def get_bbbc041_dataset(
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
    """Get the BBBC041 dataset for segmentation of malaria infected blood cells.

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
    raw_paths, label_paths = get_bbbc041_paths(path, split, label_choice, download)

    if label_choice == "semantic":
        assert not any((offsets is not None, boundaries, binary)), \
            "'offsets', 'boundaries' and 'binary' are only supported for instance labels."
    else:
        kwargs, _ = util.add_instance_label_transform(
            kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
        )

    # The images are RGB and the two splits have different shapes, so we use an image
    # collection dataset. It loads the data lazily, which also avoids running out of file
    # handles for the more than 1000 images of the training split.
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", False)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_bbbc041_loader(
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
    """Get the BBBC041 dataloader for segmentation of malaria infected blood cells.

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
    dataset = get_bbbc041_dataset(
        path, patch_shape, split=split, label_choice=label_choice, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
