"""The Plasmodium-v1 dataset contains annotations for blood cells in May Grunwald-Giemsa
stained thin blood smear images of blood infected with different hematozoa, mostly the four
*Plasmodium* species that cause malaria in humans.

The dataset contains 29,228 brightfield images from 645 thin blood smears of the same number
of patients, with roughly 2.4 million annotated cells. The smears were collected at six
French university hospitals. The annotations are *bounding boxes* with a class label for each
cell, not pixel-wise segmentation masks. The boxes are rasterized into label images here, so
that the data can be used with the segmentation functionality of `torch_em`: each box is
filled as a rectangle, either with a unique instance id ('instances') or with its category id
('semantic'). Note that these labels are only a coarse approximation of the actual cell
shapes, and that uninfected red blood cells make up roughly 94% of the annotations.
The original box annotations remain available in the YOLO format text files of the downloaded
data, so that it can also be used for detection.

The images are RGB jpegs and vary in shape, e.g. 3840 x 2160, 2560 x 1920 and 1920 x 1080
pixels. In addition to the three official splits, the data contains a 'test_zoom' split, which
holds digitally zoomed versions of the test images with their own annotations.

NOTE: Some of the images, in particular in the 'test_zoom' split, contain the actual field of
view within a large black border, which is not annotated. Consider using a sampler, e.g.
`torch_em.data.MinInstanceSampler`, to avoid sampling patches that only contain the border.

The patient-level diagnosis is available per smear in 'image_list.xlsx', together with the
hospital the smear came from. It can be used to select a subset of the data via the `species`
argument, see `SPECIES` for the available values.

NOTE: This is a very large dataset. The archive is 44 GB, and is split into five parts that
have to be downloaded individually, so the download takes a long time. Roughly 90 GB of free
disk space are needed while the data is extracted, and roughly 45 GB afterwards, plus the
space for the rasterized labels of the splits that are used.

The dataset is located at https://zenodo.org/records/8358829.
This dataset is from the publication:
- Guemas et al. (2024): https://doi.org/10.1128/spectrum.01440-23
Please cite it if you use this dataset in your research.
"""

import os
import bisect
import hashlib
import zipfile
from glob import glob
from warnings import warn
from natsort import natsorted
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/8358829/files/{}?download=1"

ARCHIVE_PARTS = {
    "Plasmodium-v1.zip.001": (10485760000, "de5ed551ac2162c2574b13c6b25d7444"),
    "Plasmodium-v1.zip.002": (10485760000, "76cbc00b44810cf3c02251636cb94331"),
    "Plasmodium-v1.zip.003": (10485760000, "8702a650475d13a2c7a301499f1184a3"),
    "Plasmodium-v1.zip.004": (10485760000, "06b97fd08051a9ce831bcf4e0860be06"),
    "Plasmodium-v1.zip.005": (2462723333, "9c7337d7e7471d5e1bd84988396cecb8"),
}
"""The parts of the split archive, with their size in bytes and their md5 checksum. The parts
are a byte-wise split of a single zip file, so they are read as one contiguous archive.
"""

CATEGORIES = (
    "WBC",
    "RBC",
    "Platelets",
    "P. falciparum",
    "P. ovale",
    "P. malariae",
    "P. vivax",
    "Babesia",
    "Trypanosoma brucei",
)
"""The cell categories annotated in Plasmodium-v1, in the order of the class ids used by the
annotation files. The semantic label id of a category is its position in this tuple plus one,
i.e. 1 for 'WBC' and 9 for 'Trypanosoma brucei' (0 is background). The categories from
'P. falciparum' onwards refer to red blood cells infected by the respective parasite; for
'Trypanosoma brucei' the parasite itself is annotated, as it is extracellular.
"""

PARASITE_CATEGORIES = ("P. falciparum", "P. ovale", "P. malariae", "P. vivax", "Babesia", "Trypanosoma brucei")
"""The categories that correspond to parasites, i.e. to infected cells or, in the case of
'Trypanosoma brucei', to the parasite itself.
"""

SPECIES = (
    "Uninfected",
    "P. falciparum",
    "P. ovale",
    "P. malariae",
    "P. vivax",
    "Babesia",
    "Trypanosoma brucei",
)
"""The patient-level diagnoses of the smears, which can be used to select a subset of the data.
"""

SPLIT_FOLDERS = {"train": "train", "val": "validation", "test": "test", "test_zoom": "test-zoom"}
"""@private
"""

LABEL_CHOICES = ("instances", "semantic")
"""@private
"""

METADATA_FILE = "image_list.xlsx"
"""@private
"""


class _SplitArchive:
    """A read-only file-like object that presents the parts of the split archive as one file.

    The parts are a byte-wise split of a single zip file, so they can be read as a contiguous
    stream. Doing this on the fly avoids writing another 44 GB copy of the archive to disk in
    order to concatenate the parts before extraction.
    """
    def __init__(self, paths):
        self._files = [open(p, "rb") for p in paths]
        self._sizes = [os.fstat(f.fileno()).st_size for f in self._files]
        # The start offset of each part within the full archive, for locating a read position.
        self._starts = [sum(self._sizes[:i]) for i in range(len(self._sizes))]
        self._size = sum(self._sizes)
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self._pos = offset
        elif whence == os.SEEK_CUR:
            self._pos += offset
        elif whence == os.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"Invalid value for whence: {whence}")
        return self._pos

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._size - self._pos

        chunks = []
        while size > 0 and self._pos < self._size:
            # Find the part that contains the current position and read from it up to its end.
            i = bisect.bisect_right(self._starts, self._pos) - 1
            file_ = self._files[i]
            file_.seek(self._pos - self._starts[i])
            chunk = file_.read(min(size, self._starts[i] + self._sizes[i] - self._pos))
            if not chunk:
                break
            chunks.append(chunk)
            self._pos += len(chunk)
            size -= len(chunk)

        return b"".join(chunks)

    def close(self):
        for file_ in self._files:
            file_.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _check_md5(path, checksum):
    """Verify the md5 checksum of a downloaded part.

    Zenodo publishes md5 checksums, whereas `util.download_source` expects sha256, so the
    checksum is verified here. It is also computed incrementally, because the parts are too
    large to be read into memory at once.
    """
    md5 = hashlib.md5()
    size = os.path.getsize(path)
    with open(path, "rb") as f, tqdm(
        total=size, unit="B", unit_scale=True, desc=f"Verify checksum of {os.path.basename(path)}"
    ) as pbar:
        for chunk in iter(lambda: f.read(32 * 1024 * 1024), b""):
            md5.update(chunk)
            pbar.update(len(chunk))

    if md5.hexdigest() != checksum:
        raise RuntimeError(
            f"The checksum of {path} does not match the expected checksum. "
            f"Expected: {checksum}, got: {md5.hexdigest()}. "
            "Please delete the file so that it is downloaded again on the next call."
        )


def _download_archive_parts(path, download):
    """Download the parts of the split archive, verifying each of them."""
    part_paths = []
    for name, (size, checksum) in ARCHIVE_PARTS.items():
        part_path = os.path.join(path, name)
        part_paths.append(part_path)

        if os.path.exists(part_path):
            if os.path.getsize(part_path) == size:
                continue
            # The download does not support resuming, so an incomplete part from an interrupted
            # run has to be downloaded again. Otherwise it would be treated as complete.
            warn(f"{part_path} is incomplete, probably due to an interrupted download. It is downloaded again.")
            os.remove(part_path)

        util.download_source(path=part_path, url=URL.format(name), download=download)
        _check_md5(part_path, checksum)

    return part_paths


def _extract(part_paths, path):
    """Extract the split archive, skipping files that were already extracted."""
    with _SplitArchive(part_paths) as archive, zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extract Plasmodium-v1 data"):
            out_path = os.path.join(path, member.filename)
            # Skip files that are already extracted completely, so that the extraction can be
            # resumed if it was interrupted.
            if os.path.isfile(out_path) and os.path.getsize(out_path) == member.file_size:
                continue
            zf.extract(member, path)


def _rasterize_annotations(annotation_path, shape):
    """Rasterize the bounding box annotations of a single image into label images.

    The annotations are given in the YOLO format, i.e. one box per line, with the class id
    followed by the center coordinates and the size of the box, all normalized to [0, 1].
    """
    with open(annotation_path) as f:
        lines = [line.split() for line in f.read().split("\n") if line.strip()]

    instances = np.zeros(shape, dtype="int32")
    semantic = np.zeros(shape, dtype="uint8")
    if len(lines) == 0:  # A few of the images do not contain any annotations.
        return instances, semantic

    annotations = np.array(lines, dtype="float64")
    box_categories = annotations[:, 0].astype("uint8") + 1  # 0 is used for the background.

    # Convert the normalized center coordinates and sizes to pixel coordinates.
    height, width = shape
    center = annotations[:, [2, 1]] * [height, width]
    size = annotations[:, [4, 3]] * [height, width]
    boxes = np.round(np.concatenate([center - size / 2, center + size / 2], axis=1)).astype("int32")

    # Some of the boxes extend beyond the image border, so they are clipped to it. The boxes are
    # kept at least one pixel wide, so that none of the cells is lost due to the rounding.
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, height)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, width)
    boxes[:, 2] = np.maximum(boxes[:, 2], np.minimum(boxes[:, 0] + 1, height))
    boxes[:, 3] = np.maximum(boxes[:, 3], np.minimum(boxes[:, 1] + 1, width))

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

    The images are used as they are, so only the labels have to be created. The folder
    structure of the annotations, which groups the images by smear, is preserved.
    """
    folder = SPLIT_FOLDERS[split]
    annotation_paths = natsorted(glob(os.path.join(data_dir, "labels", folder, "*", "*.txt")))
    assert len(annotation_paths) > 0, f"No annotations were found for the split '{split}'."

    n_missing = 0
    for annotation_path in tqdm(annotation_paths, desc=f"Preprocessing Plasmodium-v1 '{split}'"):
        smear, fname = annotation_path.split(os.sep)[-2:]
        stem = os.path.splitext(fname)[0]

        image_path = os.path.join(data_dir, "images", folder, smear, f"{stem}.jpg")
        if not os.path.exists(image_path):
            n_missing += 1
            continue

        label_paths = {
            choice: os.path.join(data_dir, "rasterized_labels", folder, choice, smear, f"{stem}.tif")
            for choice in LABEL_CHOICES
        }
        if all(os.path.exists(label_path) for label_path in label_paths.values()):
            continue

        # The annotations are normalized to the image size, so we read the shape from the image header.
        shape = imageio.improps(image_path).shape[:2]
        labels = _rasterize_annotations(annotation_path, shape)
        for choice, label in zip(LABEL_CHOICES, labels):
            os.makedirs(os.path.dirname(label_paths[choice]), exist_ok=True)
            imageio.imwrite(label_paths[choice], label, compression="zlib")

    if n_missing > 0:
        warn(
            f"{n_missing} of the {len(annotation_paths)} annotated images of the '{split}' split are "
            "not part of the Plasmodium-v1 archive. They are skipped."
        )


def _filter_by_species(data_dir, split, smears_and_stems, species):
    """Restrict the images of a split to the smears with the given patient-level diagnoses."""
    if isinstance(species, str):
        species = [species]

    invalid = [name for name in species if name not in SPECIES]
    assert len(invalid) == 0, f"{invalid} are not valid species. Choose from {list(SPECIES)}."

    import pandas as pd

    metadata = pd.read_excel(os.path.join(data_dir, METADATA_FILE))
    # The metadata does not list the zoomed test images, but they correspond to the test images.
    set_name = "test" if split == "test_zoom" else SPLIT_FOLDERS[split]
    metadata = metadata[(metadata.set_name == set_name) & (metadata.species.isin(species))]

    selected = set(zip(metadata.smear_name, metadata.image_name.str.replace(".jpg", "", regex=False)))
    filtered = [item for item in smears_and_stems if item in selected]
    assert len(filtered) > 0, f"No images were found for the species {species} in the split '{split}'."

    return filtered


def get_plasmodium_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test", "test_zoom"] = "train",
    download: bool = False,
    remove_archive: bool = True,
) -> str:
    """Download the Plasmodium-v1 dataset and rasterize the box annotations.

    The archive is split into five parts, which are downloaded individually and then extracted
    as one contiguous archive. The download of the roughly 44 GB of data takes a long time.
    It can be interrupted and will be resumed on the next call. The full archive is extracted,
    but the annotations are only rasterized for the given split.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val', 'test' or 'test_zoom'.
        download: Whether to download the data if it is not present.
        remove_archive: Whether to delete the parts of the archive after the data was extracted.
            They take up 44 GB of disk space and are not needed anymore afterwards.

    Returns:
        The filepath to the folder with the downloaded and preprocessed data.
    """
    assert split in SPLIT_FOLDERS, \
        f"'{split}' is not a valid split. Choose from {list(SPLIT_FOLDERS)}."

    data_dir = os.path.join(str(path), "Plasmodium-v1")
    # The extraction takes a while, so it is marked as complete to avoid checking all files again.
    complete_path = os.path.join(data_dir, "extraction_complete")
    if not os.path.exists(complete_path):
        os.makedirs(str(path), exist_ok=True)
        part_paths = _download_archive_parts(str(path), download)
        _extract(part_paths, str(path))

        with open(complete_path, "w") as f:
            f.write("The data was extracted completely.\n")

        if remove_archive:
            for part_path in part_paths:
                os.remove(part_path)

    _preprocess(data_dir, split)

    return data_dir


def get_plasmodium_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test", "test_zoom"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    species: Optional[Union[str, Sequence[str]]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Plasmodium-v1 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val', 'test' or 'test_zoom'.
        label_choice: The type of labels to use. Either 'instances' or 'semantic'.
        species: The patient-level diagnoses to restrict the data to, see `SPECIES` for the
            available values. By default all images of the split are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert label_choice in LABEL_CHOICES, \
        f"'{label_choice}' is not a valid label choice. Choose either 'instances' or 'semantic'."

    data_dir = get_plasmodium_data(path, split, download)

    folder = SPLIT_FOLDERS[split]
    label_paths = natsorted(
        glob(os.path.join(data_dir, "rasterized_labels", folder, label_choice, "*", "*.tif"))
    )
    assert len(label_paths) > 0, f"No labels were found for the split '{split}'."

    # The images are grouped by smear, so they are matched to the labels via the smear and the filename.
    smears_and_stems = [
        (p.split(os.sep)[-2], os.path.splitext(os.path.basename(p))[0]) for p in label_paths
    ]
    if species is not None:
        smears_and_stems = _filter_by_species(data_dir, split, smears_and_stems, species)

    raw_paths = [
        os.path.join(data_dir, "images", folder, smear, f"{stem}.jpg") for smear, stem in smears_and_stems
    ]
    label_paths = [
        os.path.join(data_dir, "rasterized_labels", folder, label_choice, smear, f"{stem}.tif")
        for smear, stem in smears_and_stems
    ]

    return raw_paths, label_paths


def get_plasmodium_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test", "test_zoom"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    species: Optional[Union[str, Sequence[str]]] = None,
    download: bool = False,
    offsets: Union[List[List[int]], None] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Plasmodium-v1 dataset for segmentation of blood cells in infected blood smears.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val', 'test' or 'test_zoom'.
        label_choice: The type of labels to use. Either 'instances' for one id per annotated
            cell, or 'semantic' for the cell categories, see `CATEGORIES` for the label ids.
        species: The patient-level diagnoses to restrict the data to, see `SPECIES` for the
            available values. By default all images of the split are used.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_plasmodium_paths(path, split, label_choice, species, download)

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


def get_plasmodium_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test", "test_zoom"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    species: Optional[Union[str, Sequence[str]]] = None,
    download: bool = False,
    offsets: Union[List[List[int]], None] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Plasmodium-v1 dataloader for segmentation of blood cells in infected blood smears.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val', 'test' or 'test_zoom'.
        label_choice: The type of labels to use. Either 'instances' for one id per annotated
            cell, or 'semantic' for the cell categories, see `CATEGORIES` for the label ids.
        species: The patient-level diagnoses to restrict the data to, see `SPECIES` for the
            available values. By default all images of the split are used.
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
    dataset = get_plasmodium_dataset(
        path, patch_shape, split=split, label_choice=label_choice, species=species, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
