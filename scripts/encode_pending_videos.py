#!/usr/bin/env python
"""Recover deferred LeRobot v3 videos into a separate dataset copy.

Stop recording first. Inspect with --dry-run, then encode with --output-root.
Only complete, unencoded trailing episodes are supported. Uses this checkout's
private batch encoder to preserve LeRobot video metadata.
"""

import argparse
import json
import math
import shutil
from pathlib import Path


def inspect_pending(root):
    import pandas as pd

    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Only this checkout's LeRobot v3.0 datasets are supported.")
    keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    if not keys:
        raise ValueError("No video features: this is an image-mode dataset, not deferred video encoding.")
    paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not paths:
        raise ValueError("No saved episode metadata found.")
    episodes = pd.concat([pd.read_parquet(path) for path in paths]).sort_values("episode_index")
    total = info["total_episodes"]
    if episodes["episode_index"].tolist() != list(range(total)):
        raise ValueError("Episode metadata is incomplete or duplicated; cannot safely encode.")
    pending = []
    for episode in episodes.to_dict("records"):
        index = int(episode["episode_index"])
        encoded = []
        for key in keys:
            fields = [f"videos/{key}/{name}" for name in
                      ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
            present = [field in episode and pd.notna(episode[field]) for field in fields]
            if any(present) and not all(present):
                raise ValueError(f"Episode {index}, {key}: partially written video metadata.")
            encoded.append(all(present))
            if all(present):
                path = root / info["video_path"].format(
                    video_key=key, chunk_index=int(episode[fields[0]]), file_index=int(episode[fields[1]])
                )
                if not path.is_file():
                    raise ValueError(f"Metadata references missing video: {path}")
                start, end = float(episode[fields[2]]), float(episode[fields[3]])
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    raise ValueError(f"Episode {index}, {key}: invalid video timestamps.")
        if any(encoded) and not all(encoded):
            raise ValueError(f"Episode {index}: only some cameras were encoded; manual recovery required.")
        if all(encoded):
            if pending:
                raise ValueError("Unencoded episodes are not a contiguous tail; manual recovery required.")
            continue
        for key in keys:
            directory = root / "images" / key / f"episode-{index:06d}"
            frames = sorted(path.name for path in directory.glob("frame-*.png"))
            expected = [f"frame-{i:06d}.png" for i in range(int(episode["length"]))]
            if not expected or frames != expected:
                raise ValueError(f"Missing/extra/non-contiguous frames: {directory}")
        pending.append(index)
    return keys, pending, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, help="New dataset directory; must not exist")
    parser.add_argument("--dry-run", action="store_true", help="Inspect only; write nothing")
    args = parser.parse_args()
    root = args.root.resolve()
    keys, pending, total = inspect_pending(root)
    print(f"Saved episodes: {total}; video cameras: {', '.join(keys)}")
    print(f"Pending episodes: {len(pending)}" + (f" ({pending[0]}..{pending[-1]})" if pending else ""))
    if args.dry_run or not pending:
        return
    if args.output_root is None:
        parser.error("Encoding requires --output-root; the original dataset is preserved.")
    output = args.output_root.resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        parser.error("--output-root must be a new directory outside the source dataset.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class LocalRecoveryDataset(LeRobotDataset):
        def _check_cached_episodes_sufficient(self):
            # Recovery deliberately allows absent videos, but requires frame data.
            return set(range(self.meta.total_episodes)).issubset(
                set(self.hf_dataset.unique("episode_index"))
            ) and len(self.hf_dataset) == self.meta.total_frames

        def download(self, *args, **kwargs):
            raise RuntimeError("Local frame data is incomplete; recovery will not download from the Hub.")

    print(f"Copying dataset to {output} (requires space for images and encoded videos)...", flush=True)
    shutil.copytree(root, output)
    dataset = LocalRecoveryDataset(args.repo_id, root=output, video_backend="pyav")
    try:
        dataset._batch_save_episode_video(pending[0], total)
    finally:
        dataset.finalize()
    _, remaining, _ = inspect_pending(output)
    if remaining:
        raise RuntimeError(f"Encoding incomplete: {remaining}")
    print(f"Done. LeRobot dataset: {output}")


if __name__ == "__main__":
    main()
