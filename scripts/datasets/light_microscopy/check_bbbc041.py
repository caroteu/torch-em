import os
import sys

from torch_em.data.datasets import get_bbbc041_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_bbbc041():
    from util import ROOT

    for split in ["train", "test"]:
        loader = get_bbbc041_loader(
            path=os.path.join(ROOT, "bbbc041"),
            batch_size=1,
            patch_shape=(512, 512),
            split=split,
            download=True,
        )
        check_loader(loader, 4, instance_labels=True, plt=True, save_path=f"bbbc041_{split}.png")


if __name__ == "__main__":
    check_bbbc041()
