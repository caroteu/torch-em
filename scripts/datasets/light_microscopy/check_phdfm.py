import os
import sys

from torch_em.data.datasets import get_phdfm_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_phdfm():
    from util import ROOT

    for split in ["train", "val", "test"]:
        loader = get_phdfm_loader(
            path=os.path.join(ROOT, "phdfm"),
            batch_size=1,
            patch_shape=(512, 512),
            split=split,
            download=True,
        )
        check_loader(loader, 4, plt=True, save_path=f"phdfm_{split}.png")


if __name__ == "__main__":
    check_phdfm()
