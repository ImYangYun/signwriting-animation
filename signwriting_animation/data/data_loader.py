import os
import math
import random
from typing import Literal, Optional
import copy
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.pose import Pose
from pose_format.utils.generic import reduce_holistic
from pose_anonymization.data.normalization import normalize_mean_std
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image
from transformers import CLIPProcessor


def normalize_pose_with_global_stats(pose: Pose, mean_std: dict):
    """
    Normalize a Pose object using an externally provided mean/std dict.
    This replaces the default per-file normalization for global consistency.
    """
    pose = copy.deepcopy(pose)
    mean = torch.tensor(mean_std["mean"]).float()
    std = torch.tensor(mean_std["std"]).float()

    data = torch.from_numpy(pose.body.data).float()
    data = (data - mean) / std
    pose.body.data = data.numpy()

    return pose

def _coalesce_maybe_nan(x) -> Optional[int]:
    """Return None if value is NaN/None/empty; else return the value."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


class DynamicPosePredictionDataset(Dataset):
    """
    A PyTorch Dataset for dynamic sampling of normalized pose sequences,
    conditioned on SignWriting images and optional scalar metadata.

    Changes:
    - Optional `reduce_holistic`: reduce keypoints (for testing/CI speedups).
      We apply reduction *before* normalization.
    - Do NOT call `.zero_filled()` here; rely on `zero_pad_collator` to pad.
    - Safe pivot in [1, total_frames-1] to ensure non-empty past/future.
    """
    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        num_past_frames: int = 60,
        num_future_frames: int = 30,
        with_metadata: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        split: Literal["train", "dev", "test"] = "train",
        use_reduce_holistic: bool = True,
    ):
        super().__init__()
        assert split in ["train", "dev", "test"]

        self.data_dir = data_dir
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.with_metadata = with_metadata
        self.use_reduce_holistic = use_reduce_holistic

        self.mean_std = None

        df_records = pd.read_csv(csv_path)
        df_records = df_records[df_records["split"] == split].reset_index(drop=True)
        self.records = df_records.to_dict(orient="records")

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)


    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        pose_path = os.path.join(self.data_dir, rec["pose"])
        if not pose_path.endswith(".pose"):
            pose_path += ".pose"

        start = _coalesce_maybe_nan(rec.get("start"))
        end   = _coalesce_maybe_nan(rec.get("end"))

        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"[ERR] Pose file not found: {pose_path}")

        with open(pose_path, "rb") as f:
            raw = Pose.read(f)

        total_frames = len(raw.body.data)
        if total_frames < 5:
            print(f"[SKIP SHORT FILE] idx={idx} | total_frames={total_frames} | "
                  f"file={os.path.basename(pose_path)}")
            return self.__getitem__((idx + 1) % len(self.records))

        if self.use_reduce_holistic:
            raw = reduce_holistic(raw)

        if hasattr(self, "mean_std") and self.mean_std is not None:
            pose = normalize_pose_with_global_stats(raw, self.mean_std)
        else:
            pose = normalize_mean_std(raw)

        total_frames = len(pose.body.data)

        if total_frames < 5:
            print(f"[SKIP SHORT CLIP] idx={idx} | total_frames={total_frames}")
            return self.__getitem__((idx + 1) % len(self.records))

        if total_frames <= (self.num_past_frames + self.num_future_frames + 2):
            # short clip: safely center
            pivot_frame = total_frames // 2
            input_start = max(0, pivot_frame - self.num_past_frames // 2)
            target_end = min(total_frames, input_start + self.num_past_frames + self.num_future_frames)
        else:
            pivot_min = self.num_past_frames
            pivot_max = total_frames - self.num_future_frames
            pivot_frame = random.randint(pivot_min, pivot_max)
            input_start = pivot_frame - self.num_past_frames
            target_end = pivot_frame + self.num_future_frames

        input_pose = pose.body[input_start:pivot_frame].torch()
        target_pose = pose.body[pivot_frame:target_end].torch()

        if idx < 3:
            print(f"[DEBUG SPLIT] idx={idx} | total={total_frames} | pivot={pivot_frame} | "
                f"input={input_start}:{pivot_frame} ({input_pose.data.shape[0]}f) | "
                f"target={pivot_frame}:{target_end} ({target_pose.data.shape[0]}f) | "
                f"file={os.path.basename(pose_path)}")

        input_data  = input_pose.data
        target_data = target_pose.data
        input_mask  = input_pose.data.mask
        target_mask = target_pose.data.mask

        pil_img = signwriting_to_clip_image(rec.get("text", ""))
        sign_img = self.clip_processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)

        sample = {
            "data": target_data,  # future window
            "conditions": {
                "input_pose": input_data,   # past window
                "input_mask": input_mask,
                "target_mask": target_mask,
                "sign_image": sign_img,
            },
            "id": rec.get("id", os.path.basename(rec["pose"])),
        }

        if self.with_metadata:
            meta = {
                "total_frames": total_frames,
                "sample_start": pivot_frame,
                "sample_end": pivot_frame + len(target_data),
                "orig_start": start or 0,
                "orig_end": end or total_frames,
            }
            sample["metadata"] = {k: torch.tensor([int(v)], dtype=torch.long) for k, v in meta.items()}

        return sample

def get_num_workers():
    cpu_count = os.cpu_count()
    return 0 if cpu_count is None or cpu_count <= 1 else cpu_count


def main():
    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data_fixed.csv"

    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        use_reduce_holistic=True,
    )
    stats = torch.load("/data/yayun/pose_data/mean_std_586.pt")
    dataset.mean_std = stats

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=get_num_workers(),
        pin_memory=False,
    )

    batch = next(iter(loader))
    print("Batch:", batch["data"].shape)
    print("Input pose:", batch["conditions"]["input_pose"].shape)
    print("Input mask:", batch["conditions"]["input_mask"].shape)
    print("Target mask:", batch["conditions"]["target_mask"].shape)
    print("Sign image:", batch["conditions"]["sign_image"].shape)
    if "metadata" in batch:
        for k, v in batch["metadata"].items():
            print(f"Metadata {k}:", v.shape)


if __name__ == "__main__":
    main()
