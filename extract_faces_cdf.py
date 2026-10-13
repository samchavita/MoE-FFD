#!/usr/bin/env python3
"""
extract_faces.py - face crops laid out for MoE-FFD's dataset.py

Two modes, chosen with --dataset:

  ffpp (default)  FaceForensics++ c23, train/valid/test by the official split files
  cdf             Celeb-DF v2, only the videos in List_of_testing_videos.txt

--- Celeb-DF v2 (--dataset cdf) ----------------------------------------------------
Reads the official layout:
    <src>/Celeb-real/id0_0000.mp4
    <src>/YouTube-real/00170.mp4
    <src>/Celeb-synthesis/id0_id1_0000.mp4
    <src>/List_of_testing_videos.txt
Writes 20 crops for each listed video:
    <dst>/Celeb-real/id0_0000/<frame>.png      (same for the other two folders)
The loader treats every entry in <dst> as a class folder, so the log is written
next to <dst> (as <dst>_extract_log.csv), not inside it.

--- FaceForensics++ (--dataset ffpp) -----------------------------------------------
Reads the official FF++ download layout:
    <src>/original_sequences/youtube/c23/videos/000.mp4
    <src>/manipulated_sequences/<Method>/c23/videos/000_003.mp4

Writes the layout dataset.py / train.py expect:
    <dst>/train/original/<id>/<frame>.png
    <dst>/train/Deepfakes/<id_id>/<frame>.png      (same for the other methods)
    <dst>/valid/...
    <dst>/test/...

Only the frames the loader will read are extracted:
    80 per original video in train (dataset.py uses 4*frame there)
    20 per video everywhere else
at evenly spaced positions i * (total_frames // n), the same rule dataset.py applies.
Because each folder then holds exactly n images, the loader's interval is 1 and it
reads every one of them.

The train/valid/test assignment comes from the official split files
(train.json, val.json, test.json in dataset/splits of the FaceForensics repo).

Requires: opencv-python, numpy, torch, facenet-pytorch, pillow
"""

import argparse
import csv
import json
import os
import sys

import cv2

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
SPLIT_FILES = {"train": "train.json", "valid": "val.json", "test": "test.json"}
EXPECTED_PAIRS = {"train": 360, "valid": 70, "test": 70}
README_EXAMPLES = {"239": "train", "579": "valid", "000": "test"}
IMG_SIZE = 224          # dataset.py resizes to 224x224, so store at that size
LOG_NAME = "extract_log.csv"
CDF_LIST = "List_of_testing_videos.txt"
CDF_EXPECTED = {"real": 178, "fake": 340}     # Table I of the CFM paper
LOG_FIELDS = ["split", "method", "video", "total_frames", "wanted", "saved", "shifted", "note"]


# ----------------------------------------------------------------------------- splits

def load_splits(splits_dir):
    splits = {}
    for split, fn in SPLIT_FILES.items():
        path = os.path.join(splits_dir, fn)
        if not os.path.isfile(path):
            sys.exit(f"Missing split file: {path}")
        with open(path) as f:
            splits[split] = [(str(a), str(b)) for a, b in json.load(f)]
    return splits


def check_splits(splits):
    seen = {}
    for split, pairs in splits.items():
        if len(pairs) != EXPECTED_PAIRS[split]:
            print(f"WARNING: {split} has {len(pairs)} pairs, expected {EXPECTED_PAIRS[split]}")
        for pair in pairs:
            for vid in pair:
                if vid in seen:
                    print(f"WARNING: id {vid} appears in both {seen[vid]} and {split}")
                seen[vid] = split
    print(f"Splits: {len(seen)} original ids "
          + ", ".join(f"{s}={2 * len(p)}" for s, p in splits.items()))
    for vid, expected in README_EXAMPLES.items():
        got = seen.get(vid, "absent")
        flag = "matches" if got == expected else "DIFFERS from"
        print(f"  id {vid} -> {got} ({flag} the repo README example, which shows it in {expected})")


def build_jobs(src, comp, splits, base_frames):
    """One job per video: (split, method, video_id, video_path, n_frames)."""
    jobs = []
    orig_dir = os.path.join(src, "original_sequences", "youtube", comp, "videos")
    for split, pairs in splits.items():
        n_orig = 4 * base_frames if split == "train" else base_frames
        for a, b in pairs:
            for vid in (a, b):
                jobs.append((split, "original", vid, os.path.join(orig_dir, vid + ".mp4"), n_orig))
            for m in METHODS:
                m_dir = os.path.join(src, "manipulated_sequences", m, comp, "videos")
                for vid in (f"{a}_{b}", f"{b}_{a}"):
                    jobs.append((split, m, vid, os.path.join(m_dir, vid + ".mp4"), base_frames))
    return jobs


def build_cdf_jobs(src, list_path, frames):
    """One job per line of the official test list, e.g. '1 YouTube-real/00170.mp4'."""
    if not os.path.isfile(list_path):
        sys.exit(f"Missing test list: {list_path}")
    jobs, seen, per_folder = [], set(), {}
    with open(list_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            rel = parts[-1].replace("\\", "/")           # the path is the last field
            if "/" not in rel:
                print(f"WARNING: skipping unrecognised line: {line.strip()}")
                continue
            folder, fn = rel.split("/")[-2:]
            vid = os.path.splitext(fn)[0]
            if (folder, vid) in seen:
                continue
            seen.add((folder, vid))
            per_folder[folder] = per_folder.get(folder, 0) + 1
            jobs.append(("test", folder, vid, os.path.join(src, folder, fn), frames))
    real = sum(n for d, n in per_folder.items() if "real" in d)   # the loader's own rule
    fake = len(jobs) - real
    print(f"Test list: {len(jobs)} videos, real={real}, fake={fake}  "
          + ", ".join(f"{d}={n}" for d, n in sorted(per_folder.items())))
    if (real, fake) != (CDF_EXPECTED["real"], CDF_EXPECTED["fake"]):
        print(f"WARNING: expected {CDF_EXPECTED['real']} real and {CDF_EXPECTED['fake']} fake")
    return jobs


# ----------------------------------------------------------------------------- detection

def make_detector(device, min_prob):
    """Returns detect(rgb) -> (x1, y1, x2, y2) of the largest confident face, or None."""
    from facenet_pytorch import MTCNN
    from PIL import Image

    mtcnn = MTCNN(keep_all=True, device=device)

    def detect(rgb):
        boxes, probs = mtcnn.detect(Image.fromarray(rgb))
        if boxes is None:
            return None
        good = [b for b, p in zip(boxes, probs) if p is not None and p >= min_prob]
        if not good:
            return None
        return max(good, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    return detect


def crop_face(bgr, detect, scale):
    box = detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if box is None:
        return None
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * scale / 2          # square box, enlarged by `scale`
    xa, ya = max(int(round(cx - half)), 0), max(int(round(cy - half)), 0)
    xb, yb = min(int(round(cx + half)), w), min(int(round(cy + half)), h)
    if xb - xa < 2 or yb - ya < 2:
        return None
    # INTER_LINEAR is what dataset.py's alb.Resize would apply to a raw crop
    return cv2.resize(bgr[ya:yb, xa:xb], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- extraction

def extract_video(video_path, out_dir, n_frames, detect, scale, max_shift):
    """
    Decodes the video once, front to back, and saves a crop at each target position.
    If no face is found on a target frame, the following frames are tried, up to
    max_shift frames later and never past the next target.
    Files are named by their real frame index, e.g. 125.png.
    Returns (total_frames, wanted, saved, shifted, note).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, n_frames, 0, 0, "cannot open"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0, n_frames, 0, 0, "no frames reported"

    interval = max(total // n_frames, 1)
    targets = [i * interval for i in range(n_frames) if i * interval < total]

    saved = shifted = 0
    slot, idx = 0, -1
    while slot < len(targets):
        if not cap.grab():
            break
        idx += 1
        target = targets[slot]
        if idx < target:
            continue
        nxt = targets[slot + 1] if slot + 1 < len(targets) else total
        last_try = min(target + max_shift, nxt - 1)
        ok, bgr = cap.retrieve()
        crop = crop_face(bgr, detect, scale) if ok else None
        if crop is not None:
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{idx}.png"), crop)
            saved += 1
            shifted += int(idx != target)
            slot += 1
        elif idx >= last_try:
            slot += 1
    cap.release()

    note = ""
    if len(targets) < n_frames:
        note = f"video has only {total} frames"
    elif slot < len(targets):
        note = f"stream ended at frame {idx}"
    return total, n_frames, saved, shifted, note


def clear_pngs(out_dir):
    if os.path.isdir(out_dir):
        for fn in os.listdir(out_dir):
            if fn.endswith(".png"):
                os.remove(os.path.join(out_dir, fn))


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) if os.path.isdir(path) else 0


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ffpp", "cdf"], default="ffpp", help="which layout to read (default ffpp)")
    ap.add_argument("--src", required=True, help="dataset root as downloaded")
    ap.add_argument("--dst", required=True, help="output root for the crops")
    ap.add_argument("--splits", help="ffpp: folder holding the official train.json, val.json, test.json")
    ap.add_argument("--list", help=f"cdf: path to the test list (default <src>/{CDF_LIST})")
    ap.add_argument("--log", help="per-video log file (default: inside <dst> for ffpp, next to <dst> for cdf)")
    ap.add_argument("--comp", default="c23", help="compression folder name (default c23)")
    ap.add_argument("--frames", type=int, default=20, help="frames per video; ffpp train originals get 4x this (default 20)")
    ap.add_argument("--scale", type=float, default=1.3, help="enlargement of the square face box (default 1.3)")
    ap.add_argument("--min-prob", type=float, default=0.9, help="minimum MTCNN confidence (default 0.9)")
    ap.add_argument("--max-shift", type=int, default=5, help="later frames to try when no face is found (default 5)")
    ap.add_argument("--limit", type=int, default=0, help="trial run: only N videos per folder")
    ap.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if available)")
    args = ap.parse_args()

    dst = os.path.normpath(args.dst)
    if args.dataset == "cdf":
        jobs = build_cdf_jobs(args.src, args.list or os.path.join(args.src, CDF_LIST), args.frames)
        out_dir_of = lambda split, method, vid: os.path.join(dst, method, vid)
        log_path = args.log or dst + "_" + LOG_NAME
    else:
        if not args.splits:
            ap.error("--splits is required for --dataset ffpp")
        splits = load_splits(args.splits)
        check_splits(splits)
        jobs = build_jobs(args.src, args.comp, splits, args.frames)
        out_dir_of = lambda split, method, vid: os.path.join(dst, split, method, vid)
        log_path = args.log or os.path.join(dst, LOG_NAME)
    full_images = sum(j[4] for j in jobs)

    missing = [j[3] for j in jobs if not os.path.isfile(j[3])]
    if missing:
        print(f"WARNING: {len(missing)} of {len(jobs)} videos not found, e.g. {missing[0]}")
        jobs = [j for j in jobs if os.path.isfile(j[3])]

    if args.limit:
        count, kept = {}, []
        for j in jobs:
            key = (j[0], j[1])
            count[key] = count.get(key, 0) + 1
            if count[key] <= args.limit:
                kept.append(j)
        jobs = kept

    os.makedirs(dst, exist_ok=True)
    done = set()
    if os.path.isfile(log_path):
        with open(log_path, newline="") as f:
            done = {(r["split"], r["method"], r["video"]) for r in csv.DictReader(f)}
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"Videos: {len(jobs)} selected, {len(jobs) - len(todo)} already logged, {len(todo)} to do")
    print(f"A full run produces up to {full_images:,} images")
    if not todo:
        return

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    detect = make_detector(args.device, args.min_prob)

    new_log = not os.path.isfile(log_path)
    images = size = short = empty = 0
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_log:
            writer.writeheader()
        for n, (split, method, vid, path, n_frames) in enumerate(todo, 1):
            out_dir = out_dir_of(split, method, vid)
            clear_pngs(out_dir)                       # leftovers from an interrupted run
            total, wanted, saved, shifted, note = extract_video(
                path, out_dir, n_frames, detect, args.scale, args.max_shift)
            writer.writerow(dict(zip(LOG_FIELDS, [split, method, vid, total, wanted, saved, shifted, note])))
            f.flush()
            images += saved
            size += dir_bytes(out_dir)
            short += int(0 < saved < wanted)
            empty += int(saved == 0)
            if n % 50 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {split}/{method}/{vid}: {saved}/{wanted}  "
                      f"total {images:,} images, {size / 1e9:.2f} GB", flush=True)

    print("\nDone.")
    print(f"  images written this run : {images:,} ({size / 1e9:.2f} GB)")
    if images:
        avg = size / images
        print(f"  average per image       : {avg / 1e3:.1f} KB")
        print(f"  projected full run      : {avg * full_images / 1e9:.1f} GB for {full_images:,} images")
    print(f"  videos short of target  : {short}  (the loader pads these by repeating frames)")
    print(f"  videos with no face     : {empty}  (no folder created, so the loader never sees them)")
    print(f"  per-video log           : {log_path}")


if __name__ == "__main__":
    main()