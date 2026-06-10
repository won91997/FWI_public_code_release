#!/usr/bin/env python3
import argparse
import os
import pathlib
from typing import List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Generate train/val/test index from split_by_inline.npy")
    p.add_argument("--data-root", required=True, help="dataset root containing seismic_full and vmodel_full")
    p.add_argument("--split-npy", default="split_by_inline.npy")
    p.add_argument("--sample-inline-npy", default="sample_inline.npy")
    p.add_argument("--out-dir", default="benchmark/generated_split")
    return p.parse_args()


def load_counts(data_root: str) -> Tuple[List[int], List[pathlib.Path], List[pathlib.Path]]:
    seis_dir = pathlib.Path(data_root) / "seismic_full"
    vm_dir = pathlib.Path(data_root) / "vmodel_full"
    seis_files = sorted(seis_dir.glob("seismic*.npy"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
    vm_files = sorted(vm_dir.glob("vmodel*.npy"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
    if len(seis_files) == 0 or len(vm_files) == 0 or len(seis_files) != len(vm_files):
        raise RuntimeError("Cannot match seismic_full/vmodel_full npy files")

    counts = []
    for s, v in zip(seis_files, vm_files):
        ns = np.load(s, mmap_mode="r").shape[0]
        nv = np.load(v, mmap_mode="r").shape[0]
        if ns != nv:
            raise RuntimeError(f"Count mismatch: {s.name}={ns}, {v.name}={nv}")
        counts.append(int(ns))
    return counts, seis_files, vm_files


def write_index_file(path: pathlib.Path, indices: np.ndarray):
    with path.open("w", encoding="utf-8") as f:
        for x in indices.tolist():
            f.write(f"{x}\n")


def main():
    args = parse_args()
    data_root = pathlib.Path(args.data_root)
    split = np.load(data_root / args.split_npy)
    sample_inline = np.load(data_root / args.sample_inline_npy)
    if split.shape[0] != sample_inline.shape[0]:
        raise RuntimeError("split and sample_inline size mismatch")

    counts, seis_files, vm_files = load_counts(str(data_root))
    total = sum(counts)
    if total != split.shape[0]:
        raise RuntimeError(f"total samples mismatch: npy={total}, split={split.shape[0]}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_index_file(out_dir / "train_idx.txt", np.where(split == 0)[0])
    write_index_file(out_dir / "val_idx.txt", np.where(split == 1)[0])
    write_index_file(out_dir / "test_idx.txt", np.where(split == 2)[0])

    # optional global mapping for adapters
    with (out_dir / "global_map.csv").open("w", encoding="utf-8") as f:
        f.write("global_idx,file_idx,in_file_idx,seismic_path,vmodel_path,inline_id,split\n")
        g = 0
        for file_idx, n in enumerate(counts):
            for j in range(n):
                f.write(
                    f"{g},{file_idx},{j},{seis_files[file_idx]},{vm_files[file_idx]},{int(sample_inline[g])},{int(split[g])}\n"
                )
                g += 1

    print(f"Generated split files in: {out_dir}")
    print(f"Train={int((split==0).sum())}, Val={int((split==1).sum())}, Test={int((split==2).sum())}")


if __name__ == "__main__":
    main()

