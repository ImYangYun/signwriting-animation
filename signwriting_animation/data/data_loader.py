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
from pose_anonymization.data.normalization import pre_process_pose
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image
from transformers import CLIPProcessor


def _coalesce_maybe_nan(x) -> Optional[int]:
    """Return None if value is NaN/None/empty; else return the value."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


class DynamicPosePredictionDataset(Dataset):
    """
    A PyTorch Dataset for dynamic sampling of pose sequences,
    conditioned on SignWriting images and optional scalar metadata.
    
    🔧 最终修复版本：
    1. reduce_holistic: 减少关键点数量（与统计量一致）
    2. pre_process_pose: 数据预处理（与统计量一致）✅
    3. 不归一化: 返回原始数据，归一化在 LightningModule 中进行
    
    数据流：
    原始 pose → reduce_holistic → pre_process_pose → 返回
    （不归一化，避免重复归一化）
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

        # 保留 mean_std 属性（用于兼容性），但不使用
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
        raw = pre_process_pose(raw)
        pose = raw

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
            "data": target_data,  # future window - 原始数据（未归一化）
            "conditions": {
                "input_pose": input_data,   # past window - 原始数据（未归一化）
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
    data_dir = "home/yayun/data/pose_data"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"

    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        use_reduce_holistic=True,
    )
    
    # ✅ 不设置 mean_std，返回原始数据
    # dataset.mean_std = torch.load("/data/yayun/pose_data/mean_std_586.pt")

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

    data = batch["data"]
    if hasattr(data, "tensor"):
        data = data.tensor
    print(f"\nData statistics:")
    print(f"  Min: {data.min().item():.4f}")
    print(f"  Max: {data.max().item():.4f}")
    print(f"  Mean: {data.mean().item():.4f}")
    print(f"  Std: {data.std().item():.4f}")
    
    if abs(data.mean().item()) < 0.1 and abs(data.std().item() - 1.0) < 0.2:
        print("  ⚠️  Warning: Data appears normalized (should be raw)")
    else:
        print("  ✓ Data is in raw range (correct)")
    
    if "metadata" in batch:
        for k, v in batch["metadata"].items():
            print(f"Metadata {k}:", v.shape)


if __name__ == "__main__":
    main()