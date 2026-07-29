#!/usr/bin/env python3
"""Extract SegDINO3D-compatible Grounding DINO + SAM features for ScanNet.

The output format is compatible with the SegDINO3D ScanNet datasets:

  {scene}.pth                    list[4] of [num_points, 256]
  {scene}_query_feats.pth        [num_queries, 256]
  {scene}_query_3dctr.pth        [num_queries, 3]
  {scene}_query_clsdists.pth     [num_queries]

Grounding DINO supplies encoder feature maps, decoder object queries, boxes,
and confidence scores. SAM turns the boxes into masks. The masks are used only
to project each 2D object into 3D and obtain a robust medoid center.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


TOO_LARGE_SCENES = {
    "scene0465_00",
    "scene0465_01",
    "scene0002_01",
    "scene0233_00",
    "scene0054_00",
    "scene0233_01",
    "scene0727_00",
    "scene0757_00",
    "scene0736_00",
    "scene0331_01",
    "scene0000_02",
    "scene0000_01",
    "scene0613_00",
}

SCANNET20_CLASSES = [
    "bathtub",
    "bed",
    "bookshelf",
    "cabinet",
    "chair",
    "counter",
    "curtain",
    "desk",
    "door",
    "floor",
    "otherfurniture",
    "picture",
    "refrigerator",
    "shower curtain",
    "sink",
    "sofa",
    "table",
    "toilet",
    "wall",
    "window",
]


def _load_runtime(vendor_root: Path) -> dict[str, Any]:
    """Import heavy/optional dependencies only after CLI arguments are parsed."""
    grounding_root = vendor_root / "GroundingDINO"
    sam_root = vendor_root / "segment_anything"
    for path in (grounding_root, sam_root):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing bundled dependency: {path}. Restore the complete "
                "tools/grounding_dino_feature_extraction/third_party directory."
            )
        sys.path.insert(0, str(path))

    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F
        from groundingdino.models import build_model
        from groundingdino.util.inference import load_image, predict
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig
        from pycocotools import mask as mask_util
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "Feature-extraction dependencies are incomplete. Install the packages "
            "listed in requirements.txt (and build Grounding DINO's CUDA extension)."
        ) from exc

    return {
        "cv2": cv2,
        "np": np,
        "torch": torch,
        "F": F,
        "load_image": load_image,
        "build_model": build_model,
        "clean_state_dict": clean_state_dict,
        "SLConfig": SLConfig,
        "predict": predict,
        "mask_util": mask_util,
        "SamPredictor": SamPredictor,
        "sam_model_registry": sam_model_registry,
    }


def load_grounding_model(args: argparse.Namespace, rt: dict[str, Any]) -> Any:
    """Load Grounding DINO while allowing an explicit local BERT directory."""
    torch = rt["torch"]
    config = rt["SLConfig"].fromfile(str(args.grounding_config))
    config.device = args.device
    config.text_encoder_type = args.text_encoder
    model = rt["build_model"](config)
    checkpoint = torch.load(args.grounding_checkpoint, map_location="cpu")
    model.load_state_dict(
        rt["clean_state_dict"](checkpoint["model"]),
        strict=False,
    )
    model.to(args.device).eval()
    return model


def _sort_frames(paths: Iterable[Path]) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return sys.maxsize, path.name

    return sorted(paths, key=key)


def _load_classes(args: argparse.Namespace) -> list[str]:
    if args.classes_file:
        with open(args.classes_file, "r", encoding="utf-8") as handle:
            classes = [line.strip() for line in handle if line.strip()]
        if not classes:
            raise ValueError(f"No classes found in {args.classes_file}")
        return classes
    if args.class_set == "scannet20":
        return SCANNET20_CLASSES
    with open(args.scannet200_meta, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return list(metadata["classes"])


def _output_name(args: argparse.Namespace) -> str:
    base = (
        f"PointGDSAMFeature_gap_{args.frame_gap}_top{args.topk_views}_mean"
        f"_queries_FPS_{args.max_queries}_medoid_clsdist"
    )
    if args.class_set == "scannet20":
        base += "_scannet20"
    return base


def _scene_outputs(output_dir: Path, scene_id: str) -> dict[str, Path]:
    return {
        "points": output_dir / f"{scene_id}.pth",
        "queries": output_dir / f"{scene_id}_query_feats.pth",
        "centers": output_dir / f"{scene_id}_query_3dctr.pth",
        "scores": output_dir / f"{scene_id}_query_clsdists.pth",
    }


def _all_outputs_exist(output_dir: Path, scene_id: str) -> bool:
    return all(path.exists() for path in _scene_outputs(output_dir, scene_id).values())


def _boxes_cxcywh_to_xyxy(boxes: Any, width: int, height: int) -> Any:
    torch = sys.modules["torch"]
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=boxes.dtype)
    cx, cy, bw, bh = boxes.unbind(-1)
    xyxy = torch.stack(
        (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), dim=-1
    )
    scale = boxes.new_tensor([width, height, width, height])
    xyxy = xyxy * scale
    xyxy[:, 0::2].clamp_(0, max(width - 1, 0))
    xyxy[:, 1::2].clamp_(0, max(height - 1, 0))
    return xyxy


def _sam_masks_from_boxes(
    predictor: Any,
    image_rgb: Any,
    boxes_xyxy: Any,
    device: str,
    multimask: bool,
) -> Any:
    torch = sys.modules["torch"]
    if boxes_xyxy.numel() == 0:
        h, w = image_rgb.shape[:2]
        return torch.empty((0, h, w), dtype=torch.bool)

    predictor.set_image(image_rgb)
    boxes_device = boxes_xyxy.to(device)
    transformed = predictor.transform.apply_boxes_torch(
        boxes_device, image_rgb.shape[:2]
    )
    with torch.inference_mode():
        masks, quality, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=multimask,
        )
    if masks.ndim != 4:
        raise RuntimeError(f"Unexpected SAM mask shape: {tuple(masks.shape)}")
    if masks.shape[1] == 1:
        masks = masks[:, 0]
    else:
        best = quality.argmax(dim=1)
        masks = masks[torch.arange(masks.shape[0], device=masks.device), best]
    return masks.detach().cpu().bool()


def _encode_mask(mask: Any, mask_util: Any, np: Any) -> dict[str, Any]:
    encoded = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(encoded["size"]), "counts": encoded["counts"]}


def extract_frame(
    frame_path: Path,
    model: Any,
    sam_predictor: Any,
    text_prompt: str,
    args: argparse.Namespace,
    rt: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = rt["torch"]
    image_rgb, image_tensor = rt["load_image"](str(frame_path))
    boxes, scores, _, query_features, encoder_output = rt["predict"](
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        remove_combined=args.remove_combined,
        return_query_features=True,
        return_image_features=True,
        return_phrases=False,
        move_model=False,
        cache_text=True,
    )
    if query_features is None or encoder_output is None:
        raise RuntimeError("Customized Grounding DINO feature outputs are unavailable.")

    height, width = image_rgb.shape[:2]
    boxes_xyxy = _boxes_cxcywh_to_xyxy(boxes, width, height)
    masks = _sam_masks_from_boxes(
        sam_predictor,
        image_rgb,
        boxes_xyxy,
        args.device,
        args.sam_multimask,
    )
    if not (len(boxes_xyxy) == len(scores) == len(query_features) == len(masks)):
        raise RuntimeError("Grounding DINO boxes/queries and SAM masks lost alignment.")

    frame_queries: list[dict[str, Any]] = []
    for box, mask, hidden_state, score in zip(
        boxes_xyxy, masks, query_features, scores
    ):
        box_i = box.round().to(torch.int64)
        x1, y1, x2, y2 = (int(v) for v in box_i.tolist())
        clipped = mask.numpy()
        clipped[:y1, :] = False
        clipped[y2:, :] = False
        clipped[:, :x1] = False
        clipped[:, x2:] = False
        frame_queries.append(
            {
                "bbox": box.tolist(),
                "mask": _encode_mask(clipped, rt["mask_util"], rt["np"]),
                "hidden_state": hidden_state.detach().cpu(),
                "cls_dist": score.detach().cpu(),
            }
        )

    feature_dict = {
        "memory": encoder_output["memory"].detach().cpu(),
        "spatial_shapes": encoder_output["spatial_shapes"].detach().cpu(),
        "level_start_index": encoder_output["level_start_index"].detach().cpu(),
    }
    return feature_dict, frame_queries


def parse_multiscale_features(
    frame_features: Sequence[dict[str, Any]], device: str, torch: Any
) -> list[Any]:
    if not frame_features:
        raise ValueError("No frame features were extracted.")
    num_scales = int(frame_features[0]["level_start_index"].shape[0])
    scales: list[list[Any]] = [[] for _ in range(num_scales)]
    for frame in frame_features:
        for scale_id in range(num_scales):
            start = int(frame["level_start_index"][scale_id])
            h, w = (int(v) for v in frame["spatial_shapes"][scale_id])
            scale = frame["memory"][0, start : start + h * w].view(h, w, -1)
            scales[scale_id].append(scale)
    return [torch.stack(scale, dim=0).to(device) for scale in scales]


def load_scene_geometry(
    root_data: Path,
    rgbd_root: Path,
    scene_id: str,
    selected_frames: Sequence[Path],
    frame_features: Sequence[dict[str, Any]],
    device: str,
    rt: dict[str, Any],
) -> tuple[Any, list[Any], Any, Any, Any, int, int]:
    cv2, np, torch = rt["cv2"], rt["np"], rt["torch"]
    scene_root = rgbd_root / scene_id
    depth_dir = scene_root / "depth"
    pose_dir = scene_root / "pose"
    intrinsic_dir = scene_root / "intrinsic"
    intrinsic_depth_path = intrinsic_dir / "intrinsic_depth.txt"
    intrinsic_color_path = intrinsic_dir / "intrinsic_color.txt"

    intrinsic_depth_base = np.loadtxt(intrinsic_depth_path)[:3, :3]
    intrinsic_color_base = np.loadtxt(intrinsic_color_path)[:3, :3]
    depth_maps, intrinsics_depth, intrinsics_color, world_to_camera = [], [], [], []
    for frame in selected_frames:
        depth_path = depth_dir / f"{frame.stem}.png"
        pose_path = pose_dir / f"{frame.stem}.txt"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(depth_path)
        camera_to_world = np.loadtxt(pose_path)
        depth_maps.append(torch.from_numpy(depth).float())
        intrinsics_depth.append(torch.from_numpy(intrinsic_depth_base).float())
        intrinsics_color.append(torch.from_numpy(intrinsic_color_base).float())
        world_to_camera.append(torch.from_numpy(np.linalg.inv(camera_to_world)).float())

    points_path = root_data / "points" / f"{scene_id}.bin"
    points_np = np.fromfile(points_path, dtype=np.float32).reshape(-1, 6)[:, :3]
    first_image = cv2.imread(str(selected_frames[0]))
    if first_image is None:
        raise FileNotFoundError(selected_frames[0])
    color_height, color_width = first_image.shape[:2]

    return (
        torch.stack(depth_maps).to(device),
        parse_multiscale_features(frame_features, device, torch),
        torch.stack(intrinsics_depth).to(device),
        torch.stack(intrinsics_color).to(device),
        torch.stack(world_to_camera).to(device),
        torch.from_numpy(points_np).to(device),
        color_height,
        color_width,
    )


def points_to_views(
    points_world: Any,
    intrinsic_depth: Any,
    intrinsic_color: Any,
    world_to_camera: Any,
    torch: Any,
) -> tuple[Any, Any, Any]:
    num_points = points_world.shape[0]
    ones = torch.ones((num_points, 1), device=points_world.device)
    homogeneous = torch.cat([points_world, ones], dim=1)
    points_cam_h = torch.einsum("vij,nj->vni", world_to_camera, homogeneous)
    points_cam = points_cam_h[:, :, :3] / points_cam_h[:, :, 3:].clamp(min=1e-6)

    points_depth = torch.einsum("vij,vnj->vni", intrinsic_depth, points_cam)
    points_depth[:, :, :2] /= points_depth[:, :, 2:]
    points_color = torch.einsum("vij,vnj->vni", intrinsic_color, points_cam)
    points_color[:, :, :2] /= points_color[:, :, 2:]
    return (
        points_depth.permute(1, 0, 2),
        points_color.permute(1, 0, 2),
        points_cam.permute(1, 0, 2),
    )


def valid_projections(points_uvd: Any, depth_maps: Any, torch: Any) -> Any:
    height, width = depth_maps.shape[1:]
    valid = (
        (points_uvd[..., 0] >= 0)
        & (points_uvd[..., 0] < width)
        & (points_uvd[..., 1] >= 0)
        & (points_uvd[..., 1] < height)
        & (points_uvd[..., 2] > 0)
        & (points_uvd[..., 2] < 20)
    )
    safe = points_uvd.clone()
    safe[~valid] = 0
    x = safe[..., 0].long()
    y = safe[..., 1].long()
    view_ids = torch.arange(depth_maps.shape[0], device=depth_maps.device)
    view_ids = view_ids.view(1, -1).expand(points_uvd.shape[0], -1)
    observed_depth = depth_maps[view_ids, y, x] / 1000.0
    point_depth = safe[..., 2]
    return valid & (point_depth > 0) & (point_depth < observed_depth)


def nearest_visible_views(
    points_cam: Any, points_color: Any, visible: Any, topk: int, torch: Any
) -> tuple[Any, Any]:
    k = min(topk, visible.shape[1])
    distance = torch.norm(points_cam[..., :3], dim=-1)
    distance[~visible] = 1e10
    view_ids = torch.topk(-distance, k, dim=-1).indices
    coords = torch.gather(points_color, 1, view_ids[..., None].expand(-1, -1, 3))
    selected_visible = torch.gather(visible, 1, view_ids)
    view_ids[~selected_visible] = -1
    coords[~selected_visible] = -1
    return view_ids, coords


def sample_point_features(
    feature_maps: Any,
    view_ids: Any,
    uv: Any,
    batch_size: int,
    rt: dict[str, Any],
) -> Any:
    torch, F = rt["torch"], rt["F"]
    num_points = view_ids.shape[0]
    channels = feature_maps.shape[-1]
    output = torch.zeros((num_points, channels), device=view_ids.device)
    counts = torch.zeros(num_points, device=view_ids.device)
    maps = feature_maps.permute(0, 3, 1, 2)
    for neighbor in range(view_ids.shape[1]):
        ids = view_ids[:, neighbor]
        coords = uv[:, neighbor, :2]
        valid = ids >= 0
        valid_indices = torch.where(valid)[0]
        order = torch.argsort(ids[valid_indices], stable=True)
        grouped_indices = valid_indices[order]
        grouped_views = ids[grouped_indices]
        unique_views, group_sizes = torch.unique_consecutive(
            grouped_views, return_counts=True
        )
        offset = 0
        # Sorting avoids rescanning all points once for every source view.
        for view_id, group_size in zip(
            unique_views.tolist(), group_sizes.tolist()
        ):
            view_indices = grouped_indices[offset : offset + group_size]
            offset += group_size
            for start in range(0, view_indices.numel(), batch_size):
                indices = view_indices[start : start + batch_size]
                grid = (coords[indices] * 2 - 1).view(1, -1, 1, 2)
                sampled = F.grid_sample(
                    maps[view_id : view_id + 1],
                    grid,
                    mode="bilinear",
                    align_corners=True,
                )[0, :, :, 0].transpose(0, 1)
                output[indices] += sampled
                counts[indices] += 1
    output /= counts.clamp(min=1).unsqueeze(-1)
    output[counts == 0] = 0
    return output


def depth_to_world_points(
    depth_maps: Any,
    intrinsic_depth: Any,
    world_to_camera: Any,
    rt: dict[str, Any],
) -> Any:
    np, torch = rt["np"], rt["torch"]
    views, height, width = depth_maps.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    uv1 = np.stack((xx.reshape(-1), yy.reshape(-1), np.ones(height * width)))
    uv1 = torch.from_numpy(uv1).float().to(depth_maps.device)
    uv1 = uv1[None].expand(views, -1, -1)
    z = depth_maps.reshape(views, -1) / 1000.0
    xyz_camera = torch.bmm(torch.linalg.inv(intrinsic_depth), uv1) * z[:, None]
    camera_to_world = torch.linalg.inv(world_to_camera)
    xyz_world = torch.bmm(camera_to_world[:, :3, :3], xyz_camera)
    xyz_world += camera_to_world[:, :3, 3:]
    return xyz_world


def deterministic_fps(points: Any, count: int, torch: Any) -> Any:
    """Match PyTorch3D's default deterministic FPS start (index zero)."""
    num_points = points.shape[0]
    if count >= num_points:
        return torch.arange(num_points, device=points.device)
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    min_dist = torch.full((num_points,), float("inf"), device=points.device)
    current = torch.tensor(0, dtype=torch.long, device=points.device)
    for index in range(count):
        selected[index] = current
        distance = ((points - points[current]) ** 2).sum(dim=1)
        min_dist = torch.minimum(min_dist, distance)
        current = torch.argmax(min_dist)
    return selected


def batched_deterministic_fps(
    point_sets: Sequence[Any], count: int, torch: Any
) -> Any:
    """Run the same index-zero FPS for several variable-length point sets."""
    lengths = torch.tensor(
        [points.shape[0] for points in point_sets],
        device=point_sets[0].device,
    )
    batch_size = len(point_sets)
    max_points = int(lengths.max())
    padded = point_sets[0].new_zeros((batch_size, max_points, 3))
    for batch_id, points in enumerate(point_sets):
        padded[batch_id, : points.shape[0]] = points

    point_ids = torch.arange(max_points, device=padded.device)
    valid = point_ids[None] < lengths[:, None]
    selected = torch.empty(
        (batch_size, count), dtype=torch.long, device=padded.device
    )
    min_dist = padded.new_full((batch_size, max_points), float("inf"))
    min_dist[~valid] = -1
    current = torch.zeros(batch_size, dtype=torch.long, device=padded.device)
    batch_ids = torch.arange(batch_size, device=padded.device)

    for index in range(count):
        selected[:, index] = current
        current_points = padded[batch_ids, current]
        distance = ((padded - current_points[:, None]) ** 2).sum(dim=-1)
        distance[~valid] = -1
        min_dist = torch.minimum(min_dist, distance)
        current = torch.argmax(min_dist, dim=1)
    return padded[batch_ids[:, None], selected]


def medoid(points: Any, max_candidates: int, torch: Any) -> Any:
    if points.shape[0] > max_candidates:
        points = points[deterministic_fps(points, max_candidates, torch)]
    distances = torch.cdist(points, points)
    return points[distances.sum(dim=1).argmin()]


def batched_medoid(
    point_sets: Sequence[Any], max_candidates: int, torch: Any
) -> Any:
    """Compute medoids together for masks that require FPS subsampling."""
    candidates = batched_deterministic_fps(
        point_sets, max_candidates, torch
    )
    distances = torch.cdist(candidates, candidates)
    medoid_ids = distances.sum(dim=-1).argmin(dim=-1)
    batch_ids = torch.arange(len(point_sets), device=candidates.device)
    return candidates[batch_ids, medoid_ids]


def project_queries_to_3d(
    video_queries: Sequence[Sequence[dict[str, Any]]],
    depth_maps: Any,
    intrinsic_depth: Any,
    world_to_camera: Any,
    args: argparse.Namespace,
    rt: dict[str, Any],
) -> tuple[Any, Any, Any]:
    cv2, np, torch = rt["cv2"], rt["np"], rt["torch"]
    point_maps = depth_to_world_points(
        depth_maps, intrinsic_depth, world_to_camera, rt
    )
    _, depth_height, depth_width = depth_maps.shape
    records: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Any]] = []

    def flush_pending() -> None:
        if not pending:
            return
        pending_centers = batched_medoid(
            [points for _, points in pending],
            args.medoid_candidates,
            torch,
        )
        for (record, _), center in zip(pending, pending_centers):
            record["center"] = center
        pending.clear()

    for view_id, queries in enumerate(video_queries):
        for query in queries:
            decoded = rt["mask_util"].decode(query["mask"])
            resized = cv2.resize(
                decoded,
                (depth_width, depth_height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = torch.from_numpy(np.asarray(resized) > 0).to(point_maps.device)
            if not mask.any():
                continue
            object_points = point_maps[view_id][:, mask.flatten()].T
            record = {
                "center": None,
                "feature": query["hidden_state"],
                "score": query["cls_dist"],
            }
            records.append(record)
            if object_points.shape[0] > args.medoid_candidates:
                pending.append((record, object_points))
                if len(pending) >= args.medoid_batch_size:
                    flush_pending()
            else:
                record["center"] = medoid(
                    object_points, args.medoid_candidates, torch
                )
    flush_pending()

    centers, features, scores = [], [], []
    for record in records:
        center = record["center"]
        if center is None or not torch.isfinite(center).all():
            continue
        centers.append(center)
        features.append(record["feature"].to(center.device))
        scores.append(record["score"].to(center.device))

    if not centers:
        return (
            torch.empty((0, 3)),
            torch.empty((0, 256)),
            torch.empty((0,)),
        )
    centers_t = torch.stack(centers)
    features_t = torch.stack(features)
    scores_t = torch.stack(scores).reshape(-1)
    if centers_t.shape[0] > args.max_queries:
        indices = deterministic_fps(centers_t, args.max_queries, torch)
        centers_t = centers_t[indices]
        features_t = features_t[indices]
        scores_t = scores_t[indices]
    return centers_t.cpu(), features_t.cpu(), scores_t.cpu()


def process_scene(
    scene_id: str,
    root_data: Path,
    rgbd_root: Path,
    output_dir: Path,
    model: Any,
    sam_predictor: Any,
    text_prompt: str,
    args: argparse.Namespace,
    rt: dict[str, Any],
) -> None:
    torch = rt["torch"]
    paths = _scene_outputs(output_dir, scene_id)
    if not args.overwrite and _all_outputs_exist(output_dir, scene_id):
        print(f"[skip] {scene_id}: all outputs already exist")
        return

    color_dir = rgbd_root / scene_id / "color"
    frames = _sort_frames(color_dir.glob("*.jpg"))
    effective_gap = args.frame_gap
    if scene_id in TOO_LARGE_SCENES and not args.disable_large_scene_gap:
        effective_gap *= 2
    selected_frames = frames[::effective_gap]
    if args.max_frames is not None:
        selected_frames = selected_frames[: args.max_frames]
    if not selected_frames:
        raise RuntimeError(f"No RGB frames found for {scene_id} in {color_dir}")

    print(
        f"[scene] {scene_id}: {len(selected_frames)}/{len(frames)} frames, "
        f"effective gap={effective_gap}"
    )
    frame_features, video_queries = [], []
    for index, frame in enumerate(selected_frames, start=1):
        print(f"  [2D] {index:04d}/{len(selected_frames):04d} {frame.name}")
        feature, queries = extract_frame(
            frame, model, sam_predictor, text_prompt, args, rt
        )
        frame_features.append(feature)
        video_queries.append(queries)

    (
        depth_maps,
        multiscale_features,
        intrinsic_depth,
        intrinsic_color,
        world_to_camera,
        points_world,
        color_height,
        color_width,
    ) = load_scene_geometry(
        root_data,
        rgbd_root,
        scene_id,
        selected_frames,
        frame_features,
        args.device,
        rt,
    )

    points_depth, points_color, points_cam = points_to_views(
        points_world,
        intrinsic_depth,
        intrinsic_color,
        world_to_camera,
        torch,
    )
    visible = valid_projections(points_depth, depth_maps, torch)
    view_ids, uv = nearest_visible_views(
        points_cam, points_color, visible, args.topk_views, torch
    )
    uv[..., 0] /= max(color_width - 1, 1)
    uv[..., 1] /= max(color_height - 1, 1)

    query_centers, query_features, query_scores = project_queries_to_3d(
        video_queries,
        depth_maps,
        intrinsic_depth,
        world_to_camera,
        args,
        rt,
    )

    point_features = [
        sample_point_features(
            scale, view_ids, uv, args.point_batch_size, rt
        ).cpu()
        for scale in multiscale_features
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(point_features, paths["points"])
    torch.save(query_features, paths["queries"])
    torch.save(query_centers, paths["centers"])
    torch.save(query_scores, paths["scores"])
    empty_ratio = float((point_features[0].abs().sum(dim=-1) == 0).float().mean())
    print(
        f"[saved] {scene_id}: points={point_features[0].shape[0]}, "
        f"queries={query_features.shape[0]}, empty={empty_ratio:.4f}"
    )


def _select_scenes(args: argparse.Namespace, rgbd_root: Path) -> list[str]:
    if args.scene:
        scenes = list(dict.fromkeys(args.scene))
    elif args.scene_list:
        with open(args.scene_list, "r", encoding="utf-8") as handle:
            scenes = [line.strip() for line in handle if line.strip()]
    else:
        scenes = sorted(path.name for path in rgbd_root.iterdir() if path.is_dir())
    if args.num_groups > 1:
        scenes = scenes[args.group_id :: args.num_groups]
    if args.limit is not None:
        scenes = scenes[: args.limit]
    return scenes


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
    default_vendor = here / "third_party"
    checkpoint_root = here / "checkpoints"
    parser = argparse.ArgumentParser(
        description="Extract SegDINO3D Grounding DINO + SAM features for ScanNet."
    )
    parser.add_argument(
        "--root-data",
        type=Path,
        default=os.environ.get("SCANNET_ROOT"),
        required="SCANNET_ROOT" not in os.environ,
        help=(
            "ScanNet preparation root containing points/ and ExtractedRGBD_GAP1/. "
            "May also be set with SCANNET_ROOT."
        ),
    )
    parser.add_argument(
        "--rgbd-root",
        type=Path,
        default=None,
        help="Defaults to <root-data>/ExtractedRGBD_GAP1.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--class-set", choices=("scannet200", "scannet20"), default="scannet200"
    )
    parser.add_argument("--classes-file", type=Path)
    parser.add_argument(
        "--scannet200-meta",
        type=Path,
        default=project_root / "evaluation" / "scannet200_dataset_meta.json",
    )
    parser.add_argument("--scene", action="append", help="Repeat for multiple scenes.")
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--frame-gap", type=int, default=5)
    parser.add_argument("--topk-views", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=2048)
    parser.add_argument("--medoid-candidates", type=int, default=2048)
    parser.add_argument("--medoid-batch-size", type=int, default=4)
    parser.add_argument("--point-batch-size", type=int, default=1000)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Debug only; restrict extraction to the first N selected frames.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--remove-combined", action="store_true")
    parser.add_argument(
        "--sam-multimask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sam-model", default="vit_h")
    parser.add_argument(
        "--grounding-config",
        type=Path,
        default=default_vendor
        / "GroundingDINO"
        / "groundingdino"
        / "config"
        / "GroundingDINO_SwinB_cfg.py",
    )
    parser.add_argument(
        "--grounding-checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "GROUNDING_DINO_CHECKPOINT",
                checkpoint_root / "groundingdino_swinb_cogcoor.pth",
            )
        ),
        help=(
            "Grounding DINO Swin-B checkpoint. May also be set with "
            "GROUNDING_DINO_CHECKPOINT."
        ),
    )
    parser.add_argument(
        "--text-encoder",
        default=str(default_vendor / "bert-base-uncased"),
        help=(
            "Hugging Face model ID or a local bert-base-uncased directory. "
            "Use a local directory on offline compute nodes."
        ),
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "SAM_CHECKPOINT",
                checkpoint_root / "sam_vit_h_4b8939.pth",
            )
        ),
        help="SAM checkpoint. May also be set with SAM_CHECKPOINT.",
    )
    parser.add_argument("--vendor-root", type=Path, default=default_vendor)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-large-scene-gap", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.group_id < args.num_groups:
        parser.error("--group-id must be in [0, --num-groups).")
    if (
        args.frame_gap < 1
        or args.topk_views < 1
        or args.max_queries < 1
        or args.medoid_candidates < 1
        or args.medoid_batch_size < 1
        or args.point_batch_size < 1
    ):
        parser.error("Sampling parameters must be positive.")

    args.root_data = args.root_data.resolve()
    args.grounding_checkpoint = args.grounding_checkpoint.expanduser().resolve()
    args.sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    for label, path in (
        ("Grounding DINO checkpoint", args.grounding_checkpoint),
        ("SAM checkpoint", args.sam_checkpoint),
    ):
        if not path.is_file():
            parser.error(
                f"{label} not found: {path}. Pass its CLI option, set the "
                "corresponding environment variable, or run download_checkpoints.sh."
            )
    required_data = (args.root_data / "points",)
    for path in required_data:
        if not path.exists():
            parser.error(f"Required ScanNet path not found: {path}")
    rgbd_root = (
        args.rgbd_root.resolve()
        if args.rgbd_root
        else args.root_data / "ExtractedRGBD_GAP1"
    )
    if not rgbd_root.is_dir():
        parser.error(f"RGB-D frame root not found: {rgbd_root}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else args.root_data / _output_name(args)
    )
    classes = _load_classes(args)
    text_prompt = " . ".join(classes)
    scenes = _select_scenes(args, rgbd_root)
    if not scenes:
        parser.error("No scenes selected.")

    print(f"Classes: {args.class_set} ({len(classes)})")
    print(f"Scenes: {len(scenes)}")
    print(f"Output: {output_dir}")
    print(f"Grounding DINO: {args.grounding_checkpoint}")
    print(f"Text encoder: {args.text_encoder}")
    print(f"SAM: {args.sam_checkpoint}")

    rt = _load_runtime(args.vendor_root.resolve())
    model = load_grounding_model(args, rt)
    sam = rt["sam_model_registry"][args.sam_model](
        checkpoint=str(args.sam_checkpoint)
    )
    sam.to(device=args.device)
    sam_predictor = rt["SamPredictor"](sam)

    failures: list[tuple[str, str]] = []
    started = time.time()
    for scene_id in scenes:
        try:
            process_scene(
                scene_id,
                args.root_data,
                rgbd_root,
                output_dir,
                model,
                sam_predictor,
                text_prompt,
                args,
                rt,
            )
        except Exception as exc:
            failures.append((scene_id, repr(exc)))
            print(f"[failed] {scene_id}: {exc}", file=sys.stderr)
            if len(scenes) == 1:
                raise
    elapsed = (time.time() - started) / 60
    print(f"Finished {len(scenes) - len(failures)}/{len(scenes)} scenes in {elapsed:.1f} min")
    if failures:
        print("Failures:", file=sys.stderr)
        for scene_id, error in failures:
            print(f"  {scene_id}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
