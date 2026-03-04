import torch
import os
import argparse
import numpy as np
from torchvision.transforms import functional as F
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth import save_video
from diffsynth.utils.auxiliary import CameraTrajectory, load_video, homo_matrix_inverse
from PIL import Image

from pathlib import Path
from diffsynth.auxiliary_models.worldmirror.utils.save_utils import save_gs_ply

import cv2

# ================= 新增：快速二进制 PLY 保存函数 =================
def save_ply_binary(points, colors, filename):
    """高效保存点云为二进制 PLY 文件"""
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        vertex_data = np.empty(len(points), dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
        ])
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['r'] = colors[:, 0]
        vertex_data['g'] = colors[:, 1]
        vertex_data['b'] = colors[:, 2]
        f.write(vertex_data.tobytes())
# =============================================================

def concat_pil_images_horizontal(img_list):
    """将多个 PIL Image 横向并排拼接"""
    widths, heights = zip(*(i.size for i in img_list))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in img_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im

@torch.no_grad()
def generate_multiple_videos(pipe, input_video, prompt, negative_prompt, cam_traj_list,
                   output_paths, alpha_threshold=1.0, static_flag=False,
                   seed=42, cfg_scale=1.0, num_inference_steps=4):
    device = pipe.device
    height, width = input_video[0].size[1], input_video[0].size[0]
    views = {
        "img": torch.stack([F.to_tensor(image)[None] for image in input_video], dim=1).to(device),
        "is_target": torch.zeros((1, len(input_video)), dtype=torch.bool, device=device),
    }
    if static_flag:
        views["is_static"] = torch.ones((1, len(input_video)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.zeros((1, len(input_video)), dtype=torch.int64, device=device)
    else:
        views["is_static"] = torch.zeros((1, len(input_video)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.arange(0, len(input_video), dtype=torch.int64, device=device).unsqueeze(0)

    # ================= 阶段 1：只执行一次 4DGS 重建 =================
    if pipe.vram_management_enabled:
        pipe.reconstructor.to(device)

    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

    if pipe.vram_management_enabled:
        pipe.reconstructor.cpu()
        torch.cuda.empty_cache()

    gaussians = predictions["splats"]

    K_orig = predictions["rendered_intrinsics"][0]
    input_cam2world = predictions["rendered_extrinsics"][0]
    timestamps_orig = predictions["rendered_timestamps"][0]

    all_generated_seqs = []
    all_depth_seqs = []
    all_rgb_seqs = []

    # ================= 阶段 2：针对每个视角轨迹循环渲染与 Diffusion =================
    for idx, cam_traj in enumerate(cam_traj_list):
        output_path = output_paths[idx]
        base_path_idx, ext_idx = os.path.splitext(output_path)
        print(f"\n---> 开始生成第 {idx + 1}/{len(cam_traj_list)} 个视角，保存至: {output_path}")
        
        K = K_orig.clone()
        timestamps = timestamps_orig.clone()

        if static_flag:
            K = K[:1].repeat(len(cam_traj), 1, 1)
            timestamps = timestamps[:1].repeat(len(cam_traj))

        # 处理镜头缩放
        ratio = torch.linspace(1, cam_traj.zoom_ratio, K.shape[0], device=device)
        K_zoomed = K.clone()
        K_zoomed[:, 0, 0] *= ratio
        K_zoomed[:, 1, 1] *= ratio

        # 获取目标相机位姿
        target_cam2world = cam_traj.c2w.to(device)
        if cam_traj.mode == "relative" and not static_flag:
            target_cam2world = input_cam2world @ target_cam2world
        
        target_world2cam = homo_matrix_inverse(target_cam2world)
        
        # GS 渲染（Rasterization）
        target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
            gaussians,
            render_viewmats=[target_world2cam],
            render_Ks=[K_zoomed],
            render_timestamps=[timestamps],
            sh_degree=0, width=width, height=height,
        )
        target_mask = (target_alpha > alpha_threshold).float()
        
        # 兼容 cam_traj 的可选属性
        if getattr(cam_traj, "use_first_frame", False):
            target_rgb[0, 0] = views["img"][0, 0].permute(1, 2, 0)
            target_mask[0, 0] = 1.0
        
        # ================= 新增：RGB-D 投影到点云 (叠帧效果) =================
        print(f"  ---> 正在提取并生成叠帧点云...")
        accumulated_pts = []
        accumulated_colors = []
        
        T_frames = target_rgb.shape[1]
        # 创建像素网格 (uv)
        v, u = torch.meshgrid(torch.arange(height, device=device), 
                              torch.arange(width, device=device), indexing='ij')
        
        for t in range(T_frames):
            rgb_t = target_rgb[0, t]            # [H, W, 3]
            depth_t = target_depth[0, t, ..., 0] # [H, W]
            mask_t = target_mask[0, t, ..., 0]   # [H, W]
            K_t = K_zoomed[t]                    # [3, 3]
            c2w_t = target_cam2world[t]          # [4, 4]
            
            fx, fy = K_t[0, 0], K_t[1, 1]
            cx, cy = K_t[0, 2], K_t[1, 2]
            
            # 反投影到相机坐标系 (Camera coordinates)
            x_c = (u - cx) * depth_t / fx
            y_c = (v - cy) * depth_t / fy
            z_c = depth_t
            
            pts_cam = torch.stack([x_c, y_c, z_c], dim=-1) # [H, W, 3]
            
            # 过滤背景（基于 Alpha Mask 和最小深度）
            valid = (depth_t > 0.01) & (mask_t > 0.5)
            pts_cam_valid = pts_cam[valid] # [N, 3]
            colors_valid = rgb_t[valid]    # [N, 3]
            
            if pts_cam_valid.shape[0] > 0:
                # 转换到世界坐标系 (World coordinates)
                pts_cam_homo = torch.cat([pts_cam_valid, torch.ones_like(pts_cam_valid[..., :1])], dim=-1)
                pts_world = (c2w_t @ pts_cam_homo.T).T[:, :3]
                
                accumulated_pts.append(pts_world.cpu().numpy())
                accumulated_colors.append(colors_valid.cpu().numpy())

        if accumulated_pts:
            merged_pts = np.concatenate(accumulated_pts, axis=0)
            merged_cols = (np.concatenate(accumulated_colors, axis=0) * 255).clip(0, 255).astype(np.uint8)
            
            pc_out_path = f"{base_path_idx}_stacked_pc.ply"
            save_ply_binary(merged_pts, merged_cols, pc_out_path)
            print(f"  [完成] 当前轨迹的叠帧点云已保存至: {pc_out_path} (总点数: {len(merged_pts)})")
        # =============================================================

        wrapped_data = {
            "source_views": views,
            "target_rgb": target_rgb,
            "target_depth": target_depth,
            "target_mask": target_mask,
            "target_poses": target_cam2world.unsqueeze(0),
            "target_intrs": K_zoomed.unsqueeze(0),
        }
        
        # Diffusion 模型生成
        generated_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed, rand_device=pipe.device,
            height=height, width=width, num_frames=len(target_cam2world),
            cfg_scale=cfg_scale, num_inference_steps=num_inference_steps, tiled=False,
            **wrapped_data,
        )
        all_generated_seqs.append(generated_frames)
        
        # 2. 深度图提取
        depth_seq = target_depth[0].permute(0, 3, 1, 2)
        depth_seq_3ch = depth_seq.repeat(1, 3, 1, 1) 
        d_min, d_max = depth_seq_3ch.min(), depth_seq_3ch.max()
        depth_norm = (depth_seq_3ch - d_min) / (d_max - d_min + 1e-6)
        depth_uint8 = (depth_norm * 255).clamp(0, 255).to(torch.uint8).cpu()
        depth_frames = [F.to_pil_image(frame) for frame in depth_uint8]
        all_depth_seqs.append(depth_frames)

        # 3. Target RGB 提取 (4DGS 原生渲染, 形状 [B, T, H, W, 3])
        # 转换为 [T, 3, H, W] 并转为 uint8
        rgb_seq = target_rgb[0].permute(0, 3, 1, 2)
        rgb_uint8 = (rgb_seq * 255).clamp(0, 255).to(torch.uint8).cpu()
        rgb_frames = [F.to_pil_image(frame) for frame in rgb_uint8]
        all_rgb_seqs.append(rgb_frames)

        # 释放每轮生成的显存
        del wrapped_data, target_rgb, target_depth, target_alpha
        torch.cuda.empty_cache()

    # ================= 阶段 3：合并多视角视频帧 =================
    print("\n---> 正在合并多个视角的视频...")
    T = min(
        # len(all_generated_seqs[0]), 
        len(all_depth_seqs[0]), 
        len(all_rgb_seqs[0])
    )
    merged_generated_frames = []
    merged_depth_frames = []
    merged_rgb_frames = []
    alignment_checks = []
    edge_checks = []

    for t in range(T):
        # 取出所有视角在 t 时刻的帧，并横向拼接
        gen_t = [seq[t] for seq in all_generated_seqs]
        merged_generated_frames.append(concat_pil_images_horizontal(gen_t))

        depth_t = [seq[t] for seq in all_depth_seqs]
        merged_depth_frames.append(concat_pil_images_horizontal(depth_t))

        rgb_t = [seq[t] for seq in all_rgb_seqs]
        merged_rgb_frames.append(concat_pil_images_horizontal(rgb_t))

    # ================= 阶段 4：保存合并结果 =================
    base_path, ext = os.path.splitext(output_path)
    out_gen = f"{base_path}_merged_diffusion{ext}"
    out_depth = f"{base_path}_merged_depth{ext}"
    out_rgb = f"{base_path}_merged_target_rgb{ext}"

    save_video(merged_generated_frames, out_gen, fps=16)
    print(f"  [完成] 视角合并后的 Diffusion 视频已保存至: {out_gen}")

    save_video(merged_depth_frames, out_depth, fps=16)
    print(f"  [完成] 视角合并后的 深度图(Depth) 视频已保存至: {out_depth}")

    save_video(merged_rgb_frames, out_rgb, fps=16)
    print(f"  [完成] 视角合并后的 Target RGB 视频已保存至: {out_rgb}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="NeoVerse Unified Inference",
    )

    # Trajectory specification (mutually exclusive)
    traj_group = parser.add_mutually_exclusive_group(required=True)
    traj_group.add_argument("--trajectory", nargs='+',
                            choices=["pan_left", "pan_right", "tilt_up", "tilt_down",
                                     "move_left", "move_right", "push_in", "pull_out",
                                     "boom_up", "boom_down", "orbit_left", "orbit_right",
                                     "static"],
                            help="Predefined trajectory type")
    traj_group.add_argument("--trajectory_file", nargs='+',
                            help="Path to JSON trajectory file")

    # Predefined trajectory parameters
    parser.add_argument("--angle", type=float,
                        help="Override rotation angle for pan/tilt/orbit")
    parser.add_argument("--distance", type=float,
                        help="Override translation distance for move/push/pull/boom")
    parser.add_argument("--orbit_radius", type=float,
                        help="Override orbit radius")
    parser.add_argument("--traj_mode", choices=["relative", "global"], default="relative",
                        help="Trajectory mode (default: relative)")
    parser.add_argument("--zoom_ratio", type=float, default=1.0,
                        help="Zoom factor for zoom_in/zoom_out (default: 1.0)")

    # Validation only
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate trajectory file, don't run inference")

    # Input/output
    parser.add_argument("--input_path", help="Input video or image path")
    parser.add_argument("--output_path", default="outputs/inference.mp4",
                        help="Output video path (default: outputs/inference.mp4)")
    parser.add_argument("--prompt", default="A smooth video with complete scene content. Inpaint any missing regions or margins naturally to match the surrounding scene.",
                        help="Text prompt for generation")
    parser.add_argument("--negative_prompt", default="",
                        help="Negative text prompt")

    # Model parameters
    parser.add_argument("--model_path", default="models",
                        help="Model directory path (default: models)")
    parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt",
                        help="Path to reconstructor checkpoint")
    parser.add_argument("--disable_lora", action="store_true",
                        help="Skip distilled LoRA loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames (default: 81)")

    # Video loading
    parser.add_argument("--height", type=int, default=336,
                        help="Output height (default: 336)")
    parser.add_argument("--width", type=int, default=560,
                        help="Output width (default: 560)")
    parser.add_argument("--resize_mode", choices=["center_crop", "resize"],
                        default="center_crop",
                        help="Video resize mode (default: center_crop)")

    # Advanced
    parser.add_argument("--alpha_threshold", type=float, default=1.0,
                        help="Alpha mask threshold (0.0-1.0)")
    parser.add_argument("--static_scene", action="store_true",
                        help="Enable static scene mode")
    parser.add_argument("--vis_rendering", action="store_true",
                        help="Save intermediate rendering visualizations")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- LoRA / inference params ---
    use_lora = not args.disable_lora
    num_inference_steps = 4 if use_lora else 50
    cfg_scale = 1.0 if use_lora else 5.0

    lora_path = os.path.join(
        args.model_path,
        "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
    ) if use_lora else None

    # --- Validate-only mode ---
    if args.validate_only:
        if args.trajectory_file is None:
            print("Error: --validate_only requires --trajectory_file")
            return 1
        print(f"Validating trajectory file: {args.trajectory_file}")
        try:
            data = CameraTrajectory.validate_json(args.trajectory_file)
            fmt = "Keyframe operations" if "keyframes" in data else "Direct matrices"
            count = len(data.get("keyframes", data.get("trajectory", [])))
            print(f"  Format: {fmt}")
            print(f"  Entries: {count}")
            print(f"  Mode: {data.get('mode', 'relative')}")
            print("Validation passed!")
            return 0
        except ValueError as e:
            print(f"Validation failed: {e}")
            return 1

    # --- Normal inference mode ---
    if args.input_path is None:
        print("Error: --input_path is required for inference")
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build trajectories list
    cam_traj_list = []
    output_paths = []
    base_output, ext = os.path.splitext(args.output_path)

    if args.trajectory:
        for t_name in args.trajectory:
            cam_traj = CameraTrajectory.from_predefined(
                t_name,
                num_frames=args.num_frames,
                mode=args.traj_mode,
                angle=args.angle,
                distance=args.distance,
                orbit_radius=args.orbit_radius,
                zoom_ratio=args.zoom_ratio,
            )
            cam_traj_list.append(cam_traj)
            output_paths.append(f"{base_output}_{t_name}{ext}")
    else:
        for t_file in args.trajectory_file:
            cam_traj = CameraTrajectory.from_json(t_file)
            cam_traj_list.append(cam_traj)
            # 使用json文件名作为后缀    
            file_name = os.path.splitext(os.path.basename(t_file))[0]
            output_paths.append(f"{base_output}_{file_name}{ext}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        lora_path=lora_path,
        lora_alpha=1.0,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    print("Model loaded!")

    # Load video
    print(f"Loading video from {args.input_path}...")
    images = load_video(args.input_path, args.num_frames,
                        resolution=(args.width, args.height),
                        resize_mode=args.resize_mode,
                        static_scene=args.static_scene)

    # Run inference
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.vis_rendering:
        # Save rendering visualizations to a folder named after the output (without extension)
        vis_dir = os.path.splitext(output_path)[0]
        os.makedirs(vis_dir, exist_ok=True)
        pipe.save_root = vis_dir

    print(f"Generating with trajectory: {cam_traj.name} (mode={cam_traj.mode})")
    generate_multiple_videos(
        pipe=pipe,
        input_video=images,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cam_traj_list=cam_traj_list,
        output_paths=output_paths, # 传入多个保存路径
        alpha_threshold=args.alpha_threshold,
        static_flag=args.static_scene,
        seed=args.seed,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
    )
    print(f"Done! Output saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
