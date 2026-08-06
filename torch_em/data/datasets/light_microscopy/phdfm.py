"""The PHDFM dataset contains annotations of the developmental zones in fluorescence
microscopy images of *A. thaliana* root tissue.

The dataset contains 601 2D images of root tissue samples that were stained with the
ratiometric fluorescent indicator HPTS (8-hydroxypyrene-1,3,6-trisulfonic acid trisodium
salt). The images are the brightfield channel for excitation at 405 nm and are all
512 x 512 pixels. The annotations were created manually by plant biologists and assign
each pixel to one of five *regions*, see `CLASSES` for the class ids. The three
developmental zones are sub-regions of the root: the root tissue that is not part of an
annotated zone is labelled as 'root tissue', so the full root corresponds to all classes
except the background. Most images contain only one of the three zones.

The images are stored as OME-TIFF files that hold the image and the annotations as two
channels. They are split into separate tif files for the image and the annotations here, so
that the data can be loaded lazily. Storing them as one file per image with the image and the
annotations as internal datasets, e.g. as hdf5, would keep a file handle open for each of the
601 images and thereby exceed the limit for open files on many systems.

Since this is a semantic segmentation dataset, the labels are the class ids and not
instance ids. Pass `label_dtype=torch.int64` to the dataset or loader if you want to train
with a loss that expects integer class indices, such as `torch.nn.CrossEntropyLoss`.

The dataset is located at https://zenodo.org/records/5841376.
This dataset is from the publication:
- Wanner et al. (2024): https://doi.org/10.1017/qpb.2024.11
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

import imageio.v3 as imageio
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/5841376/files/root-seg-dataset.tar.gz"
CHECKSUM = "f4a5c94a178b618efe503c3a0090bb74066cf62149536252170a565557c1ba29"

CLASSES = (
    "background",
    "root tissue",
    "early elongation zone",
    "late elongation zone",
    "meristematic zone",
)
"""The regions annotated in PHDFM. The semantic label id of a region is its position in this
tuple, i.e. 0 for 'background' and 4 for 'meristematic zone'. Note that the background is
part of the annotations here, in contrast to the other datasets.
"""

ZONE_CLASSES = ("early elongation zone", "late elongation zone", "meristematic zone")
"""The developmental zones of the root, which are the regions of interest of this dataset.
"""

LABEL_CHOICES = ("semantic", "binary")
"""@private
"""


def _get_image_id(path):
    """Get the image id of a preprocessed file, which is used to define the splits."""
    return os.path.splitext(os.path.basename(path))[0][len("image_"):]


def _preprocess(path):
    """Split the OME-TIFF files into separate tif files for the image and the annotations.

    Each OME-TIFF contains the image and the annotations as two channels. The values of both
    channels are integers, but they are stored as float64, so they are converted to uint8 in
    order to keep the label ids exact and to save disk space.
    """
    image_paths = natsorted(glob(os.path.join(path, "PHDFM_dataset", "*.ome.tif")))
    assert len(image_paths) > 0, f"Could not find the PHDFM images in {path}."

    data_dir = os.path.join(path, "preprocessed")
    image_dir = os.path.join(data_dir, "images")
    label_dirs = {choice: os.path.join(data_dir, "labels", choice) for choice in LABEL_CHOICES}
    for dir_ in (image_dir, *label_dirs.values()):
        os.makedirs(dir_, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Preprocessing PHDFM"):
        # The filenames are of the form 'image_<id>.ome.tif'.
        fname = f"{os.path.basename(image_path)[:-len('.ome.tif')]}.tif"
        out_paths = {
            choice: os.path.join(label_dir, fname) for choice, label_dir in label_dirs.items()
        }
        out_paths["image"] = os.path.join(image_dir, fname)
        if all(os.path.exists(out_path) for out_path in out_paths.values()):
            continue

        image = imageio.imread(image_path)
        assert image.shape[0] == 2, f"Expected two channels in {image_path}, got {image.shape[0]}."
        semantic = image[1].astype("uint8")
        assert semantic.max() < len(CLASSES), f"Unexpected label id in {image_path}: {semantic.max()}."

        data = {"image": image[0].astype("uint8"), "semantic": semantic}
        # The zones are sub-regions of the root, so the root is the union of all classes
        # except the background.
        data["binary"] = (semantic > 0).astype("uint8")

        for name, out_path in out_paths.items():
            imageio.imwrite(out_path, data[name], compression="zlib")

    return data_dir


def _get_split_ids(image_ids, split):
    """Get the image ids of a split, reproducing the split of the original publication.

    The publication splits the data into 80% train, 10% val and 10% test, by applying
    `train_test_split` with a fixed random state to the image ids sorted as strings.
    """
    train_ids, val_test_ids = train_test_split(sorted(image_ids), test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=42)

    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    assert split in split_map, f"'{split}' is not a valid split. Choose from {list(split_map)}."

    return split_map[split]


def get_phdfm_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PHDFM dataset and split the images and annotations into separate files.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the preprocessed data.
    """
    data_dir = os.path.join(str(path), "PHDFM_dataset")
    if not os.path.exists(data_dir):
        os.makedirs(str(path), exist_ok=True)
        tar_path = os.path.join(str(path), "root-seg-dataset.tar.gz")
        util.download_source(path=tar_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip_tarfile(tar_path=tar_path, dst=str(path))

    return _preprocess(str(path))


def get_phdfm_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = None,
    label_choice: Literal["semantic", "binary"] = "semantic",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PHDFM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val', 'test', or None to use all images.
        label_choice: The type of labels to use. Either 'semantic' or 'binary'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert label_choice in LABEL_CHOICES, \
        f"'{label_choice}' is not a valid label choice. Choose either 'semantic' or 'binary'."

    data_dir = get_phdfm_data(path, download)

    raw_paths = glob(os.path.join(data_dir, "images", "*.tif"))
    assert len(raw_paths) > 0, f"Could not find the preprocessed data in {data_dir}."

    image_ids = [_get_image_id(p) for p in raw_paths]
    if split is not None:
        image_ids = _get_split_ids(image_ids, split)
    image_ids = natsorted(image_ids)

    raw_paths = [os.path.join(data_dir, "images", f"image_{i}.tif") for i in image_ids]
    label_paths = [os.path.join(data_dir, "labels", label_choice, f"image_{i}.tif") for i in image_ids]

    return raw_paths, label_paths


def get_phdfm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    label_choice: Literal["semantic", "binary"] = "semantic",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the PHDFM dataset for segmentation of developmental zones in root tissue.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val', 'test', or None to use all images.
        label_choice: The type of labels to use. Either 'semantic' for the five regions, see
            `CLASSES` for the label ids, or 'binary' for the root tissue vs. the background.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_phdfm_paths(path, split, label_choice, download)

    # We use an image collection dataset, which loads the data lazily. This avoids keeping a
    # file handle open for each of the images, which would exceed the limit for open files.
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", False)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_phdfm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    label_choice: Literal["semantic", "binary"] = "semantic",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the PHDFM dataloader for segmentation of developmental zones in root tissue.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val', 'test', or None to use all images.
        label_choice: The type of labels to use. Either 'semantic' for the five regions, see
            `CLASSES` for the label ids, or 'binary' for the root tissue vs. the background.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_phdfm_dataset(
        path, patch_shape, split=split, label_choice=label_choice, download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
