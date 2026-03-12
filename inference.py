import torch
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import functional as F
from scipy.spatial.transform import Rotation
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth import save_video
from diffsynth.utils.auxiliary import CameraTrajectory, load_video, homo_matrix_inverse
from diffsynth.utils.app import extract_point_cloud, build_scene_glb
from diffsynth.auxiliary_models.depth_anything_3.utils.gsply_helpers import export_ply

# ================= 核心新增：引入 Umeyama Sim(3) 对齐算法 =================
from diffsynth.auxiliary_models.depth_anything_3.utils.pose_align import align_poses_umeyama
# =======================================================================

def transform_gaussians_sim3(gaussians, R, T, scale):
    """
    将 Gaussians 对象参数经过 Sim(3) 变换 (旋转R, 平移T, 缩放scale)。
    """
    if isinstance(gaussians, list):
        return [transform_gaussians_sim3(g, R, T, scale) for g in gaussians]

    device = R.device

    def quat_mult(q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    R_np = R.detach().cpu().to(torch.float32).numpy()
    rot = Rotation.from_matrix(R_np)
    qx, qy, qz, qw = rot.as_quat()
    q_c2w = torch.tensor([qw, qx, qy, qz], device=device, dtype=torch.float32)

    kwargs = {}
    for k, v in gaussians.__dict__.items():
        if k == 'means':
            # Sim(3) 作用于坐标: X_new = scale * (R @ X) + T
            new_means = scale * torch.matmul(v, R.to(v.dtype).T) + T.to(v.dtype)
            kwargs['means'] = new_means
        elif k == 'scales':
            # 缩放高斯体的物理尺寸
            kwargs['scales'] = v * scale
        elif k == 'rotations':
            q_c2w_exp = q_c2w.unsqueeze(0).expand(v.shape[0], 4).to(v.dtype)
            new_quats = quat_mult(q_c2w_exp, v)
            new_quats = new_quats / new_quats.norm(dim=-1, keepdim=True)
            kwargs['rotations'] = new_quats
        elif isinstance(v, torch.Tensor):
            kwargs[k] = v.clone()
        else:
            kwargs[k] = v

    from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
    return Gaussians(**kwargs)

def apply_sim3_to_poses(poses, R, T, s):
    """
    将 Sim(3) 变换应用于 Camera-to-World 轨迹
    poses: [N, 4, 4]
    """
    out_poses = poses.clone()
    # 旋转: R_new = R @ R_old
    out_poses[:, :3, :3] = R @ poses[:, :3, :3]
    # 平移: T_new = s * (R @ T_old) + T
    out_poses[:, :3, 3] = s * (R @ poses[:, :3, 3].unsqueeze(-1)).squeeze(-1) + T
    return out_poses

def merge_and_filter_gaussians(gaussians_list, opacity_thresh=0.05, conf_thresh=0.0):
    """
    学习 Rasterizer 逻辑：对列表中的多个局部高斯体进行掩码过滤，并使用 torch.cat 合并为单一全局 Gaussians 对象。
    """
    if not gaussians_list:
        return None
    
    filtered_means, filtered_scales, filtered_rots = [], [], []
    filtered_harmonics, filtered_opacities, filtered_conf = [], [], []
    
    for gs in gaussians_list:
        # 1. 过滤逻辑：参考 forward 中的阈值 Mask
        mask = torch.ones_like(gs.opacities, dtype=torch.bool).squeeze(-1)
        if opacity_thresh >= 0:
            mask = mask & (gs.opacities.squeeze(-1) >= opacity_thresh)
        if conf_thresh >= 0 and getattr(gs, "confidences", None) is not None:
            mask = mask & (gs.confidences.squeeze(-1) > conf_thresh)
            
        if mask.sum() == 0:
            continue
            
        # 提取当前局部高斯中符合条件的点
        filtered_means.append(gs.means[mask])
        filtered_scales.append(gs.scales[mask])
        filtered_rots.append(gs.rotations[mask])
        filtered_harmonics.append(gs.harmonics[mask])
        filtered_opacities.append(gs.opacities[mask])
        if getattr(gs, "confidences", None) is not None:
            filtered_conf.append(gs.confidences[mask])
            
    if not filtered_means:
        return None
        
    # 2. 拼接逻辑：参考 rasterize_splats 中的 torch.cat
    from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
    
    merged_gaussians = Gaussians(
        means=torch.cat(filtered_means, dim=0),
        scales=torch.cat(filtered_scales, dim=0),
        rotations=torch.cat(filtered_rots, dim=0),
        harmonics=torch.cat(filtered_harmonics, dim=0),
        opacities=torch.cat(filtered_opacities, dim=0),
        confidences=torch.cat(filtered_conf, dim=0) if filtered_conf else None,
        timestamp=0 # 静态场景默认时间戳
    )
    
    return merged_gaussians

def save_single_gaussians_to_ply(merged_gaussians, out_path):
    """
    将单一的 Gaussians 对象直接导出为 PLY 格式文件。
    """
    if merged_gaussians is None or merged_gaussians.means.shape[0] == 0:
        print("No Gaussians to export.")
        return

    # 全部转移至 CPU 并转换为 float32 以供导出
    means = merged_gaussians.means.detach().cpu().float()
    scales = merged_gaussians.scales.detach().cpu().float()
    rotations = merged_gaussians.rotations.detach().cpu().float()
    harmonics = merged_gaussians.harmonics.detach().cpu().float()
    
    # 调整球谐函数维度以适配导出脚本
    if harmonics.ndim == 2:
        harmonics = harmonics.unsqueeze(-1)
    elif harmonics.ndim == 3 and harmonics.shape[2] != 1:
        harmonics = harmonics.transpose(1, 2)
        
    opacities = merged_gaussians.opacities.detach().cpu().float()
    # 3DGS 要求的逆 Sigmoid 转换
    inv_op = torch.log(opacities / (1.0 - opacities + 1e-7))

    from diffsynth.auxiliary_models.depth_anything_3.utils.gsply_helpers import export_ply
    from pathlib import Path
    
    export_ply(
        means=means, scales=scales,
        rotations=rotations, harmonics=harmonics,
        opacities=inv_op, path=Path(out_path),
        shift_and_scale=False, save_sh_dc_only=True, match_3dgs_mcmc_dev=False
    )
    print(f"✅ Successfully exported optimized GLOBAL Gaussians to {out_path} (Total Points: {means.shape[0]})")

@torch.no_grad()
def generate_video(pipe, input_video, prompt, negative_prompt, cam_traj: CameraTrajectory,
                   output_path="outputs/output.mp4", alpha_threshold=1.0, static_flag=False,
                   seed=42, cfg_scale=1.0, num_inference_steps=4, num_iterations=1):
    device = pipe.device
    height, width = input_video[0].size[1], input_video[0].size[0]
    
    current_video = input_video
    
    # 用于记录前一次生成的真实全局位姿，作为当前局部重建轨迹的匹配参考
    last_global_target_cam2world = None 
    all_global_gaussians = [] 
    global_time_offset = 0
    base_output_path = output_path
    name, ext = os.path.splitext(base_output_path)
    all_generated_frames, all_combined_frames = [], []
    
    for it in range(num_iterations):
        print(f"\n--- Iteration {it+1}/{num_iterations} ---")
        current_static_flag = static_flag if it == 0 else False
        
        views = {
            "img": torch.stack([F.to_tensor(image)[None] for image in current_video], dim=1).to(device),
            "is_target": torch.zeros((1, len(current_video)), dtype=torch.bool, device=device),
        }
        
        if current_static_flag:
            views["is_static"] = torch.ones((1, len(current_video)), dtype=torch.bool, device=device)
            views["timestamp"] = torch.zeros((1, len(current_video)), dtype=torch.int64, device=device)
        else:
            views["is_static"] = torch.zeros((1, len(current_video)), dtype=torch.bool, device=device)
            views["timestamp"] = torch.arange(
                global_time_offset, 
                global_time_offset + len(current_video), 
                dtype=torch.int64, 
                device=device
            ).unsqueeze(0)

        if pipe.vram_management_enabled: pipe.reconstructor.to(device)
        with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
            predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)
        if pipe.vram_management_enabled:
            pipe.reconstructor.cpu()
            torch.cuda.empty_cache()

        local_gaussians = predictions["splats"] 
        K = predictions["rendered_intrinsics"][0]
        local_cam2world = predictions["rendered_extrinsics"][0]
        timestamps = predictions["rendered_timestamps"][0]

        if current_static_flag:
            K = K[:1].repeat(len(cam_traj.c2w), 1, 1)
            timestamps = timestamps[:1].repeat(len(cam_traj.c2w))

        # ================= SE(3) 全局对齐 =================
        if it == 0:
            # 第一次迭代：定义局部坐标系为初始全局坐标系
            R_tensor = torch.eye(3, device=device)
            T_tensor = torch.zeros(3, device=device)
            s_val = 1.0
            current_transformed_gaussians = local_gaussians[0]
        else:
            c2w_ref = last_global_target_cam2world.detach().cpu().numpy().astype(np.float64)
            c2w_est = local_cam2world.detach().cpu().numpy().astype(np.float64)
            
            # 【纯旋转模式】相机中心几乎不动，Umeyama 会退化。
            # 策略：强制缩放 scale = 1.0，直接使用首帧的 SE(3) 相对位姿进行刚性对齐
            # 计算相对位姿: M_align = c2w_ref @ c2w_est^-1
            M_align = c2w_ref[0] @ np.linalg.inv(c2w_est[0])
            
            R_tensor = torch.tensor(M_align[:3, :3], device=device, dtype=torch.float32)
            T_tensor = torch.tensor(M_align[:3, 3], device=device, dtype=torch.float32)
            s_val = 1.0
            current_transformed_gaussians = transform_gaussians_sim3(local_gaussians[0], R_tensor, T_tensor, s_val)
        # ==========================================================
            
        all_global_gaussians.extend(current_transformed_gaussians)
        
        # 变焦缩放处理
        ratio = torch.linspace(1, cam_traj.zoom_ratio, K.shape[0], device=device)
        K_zoomed = K.clone()
        if K_zoomed.shape[0] < cam_traj.c2w.shape[0]:
            K_zoomed = torch.cat([K_zoomed, K_zoomed[-1:].repeat(cam_traj.c2w.shape[0] - K_zoomed.shape[0], 1, 1)], dim=0)
        elif K_zoomed.shape[0] > cam_traj.c2w.shape[0]:
            K_zoomed = K_zoomed[:cam_traj.c2w.shape[0]]
        K_zoomed[:, 0, 0] *= ratio; K_zoomed[:, 1, 1] *= ratio

        # 构建局部目标相机位姿
        local_target_cam2world = cam_traj.c2w.to(device)
        if cam_traj.mode == "relative" and not current_static_flag:
            base_pose = local_cam2world[-1:] 
            local_target_cam2world = base_pose @ local_target_cam2world

        # 构建对应的全局目标位姿 (为下一次迭代的 Umeyama 对齐做准备)
        global_target_cam2world = apply_sim3_to_poses(local_target_cam2world, R_tensor, T_tensor, s_val)
        last_global_target_cam2world = global_target_cam2world.clone()
        
        global_target_world2cam = homo_matrix_inverse(global_target_cam2world)
        
        ts = timestamps.clone()
        if ts.shape[0] < global_target_world2cam.shape[0]:
            ts = torch.cat([ts, ts[-1:].repeat(global_target_world2cam.shape[0] - ts.shape[0])], dim=0)
        else:
            ts = ts[:global_target_world2cam.shape[0]]

        # 在统一尺度下渲染全局高斯
        target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
            [all_global_gaussians],
            render_viewmats=[global_target_world2cam],
            render_Ks=[K_zoomed],
            render_timestamps=[ts],
            sh_degree=0, width=width, height=height,
        )
        

        target_mask = (target_alpha > alpha_threshold).float()
        if cam_traj.use_first_frame:
            target_rgb[0, 0] = views["img"][0, -1].permute(1, 2, 0)
            target_mask[0, 0] = 1.0
            
        wrapped_data = {
            "source_views": views,
            "target_rgb": target_rgb,
            "target_depth": target_depth,
            "target_mask": target_mask,
            "target_poses": local_target_cam2world.unsqueeze(0),
            "target_intrs": K_zoomed.unsqueeze(0),
        }
        
        generated_frames = pipe(
            prompt=prompt, negative_prompt=negative_prompt, seed=seed + it, 
            rand_device=pipe.device, height=height, width=width, num_frames=len(local_target_cam2world),
            cfg_scale=cfg_scale, num_inference_steps=num_inference_steps, tiled=False, **wrapped_data,
        )
        
        _target_rgb = target_rgb[0].detach().cpu().float().numpy()
        _target_depth = target_depth[0].detach().cpu().float().numpy()
        if _target_depth.ndim == 3: _target_depth = _target_depth[..., np.newaxis]
            
        _target_rgb = np.clip(_target_rgb * 255, 0, 255).astype(np.uint8)
        d_min, d_max = _target_depth.min(), _target_depth.max()
        if d_max > d_min: _target_depth = (_target_depth - d_min) / (d_max - d_min)
        else: _target_depth = np.zeros_like(_target_depth)
        _target_depth = np.clip(_target_depth * 255, 0, 255).astype(np.uint8)
        
        if _target_depth.shape[-1] == 1: _target_depth = np.repeat(_target_depth, 3, axis=-1)
            
        rgb_pil = [Image.fromarray(f) for f in _target_rgb]
        depth_pil = [Image.fromarray(f) for f in _target_depth]
        
        combined_iter_frames = []
        for r_img, d_img, g_img in zip(rgb_pil, depth_pil, generated_frames):
            tw = r_img.width + d_img.width + g_img.width
            th = max(r_img.height, d_img.height, g_img.height)
            comb = Image.new('RGB', (tw, th))
            comb.paste(r_img, (0, 0))
            comb.paste(d_img, (r_img.width, 0))
            comb.paste(g_img, (r_img.width + d_img.width, 0))
            combined_iter_frames.append(comb)
            
        if it == 0:
            all_generated_frames.extend(generated_frames)
            all_combined_frames.extend(combined_iter_frames) 
        else:
            all_generated_frames.extend(generated_frames[1:])
            all_combined_frames.extend(combined_iter_frames[1:])

        current_video = generated_frames
        global_time_offset += len(current_video) - 1

    final_output_path = f"{name}_full_concat{ext}"
    save_video(all_generated_frames, final_output_path, fps=16)
    print(f"\n>>> Saved fully concatenated continuous video to {final_output_path}")
    
    final_combined_path = f"{name}_full_combined{ext}"
    save_video(all_combined_frames, final_combined_path, fps=16)
    print(f">>> Saved fully concatenated combined video to {final_combined_path}")

    print("\n>>> Optimizing and Merging global 3D Gaussians list...")
    final_merged_gaussians = merge_and_filter_gaussians(
        all_global_gaussians, 
        opacity_thresh=0.05, 
        conf_thresh=0.0
    )
    
    global_ply_path = f"{name}_global_scene.ply"
    print(f">>> Saving Merged Global 3D Gaussians to {global_ply_path}...")
    
    save_single_gaussians_to_ply(final_merged_gaussians, global_ply_path)

def parse_args():
    parser = argparse.ArgumentParser(description="NeoVerse Unified Inference")
    traj_group = parser.add_mutually_exclusive_group(required=True)
    traj_group.add_argument("--trajectory", choices=["pan_left", "pan_right", "tilt_up", "tilt_down",
                                     "move_left", "move_right", "push_in", "pull_out",
                                     "boom_up", "boom_down", "orbit_left", "orbit_right", "static"], help="Predefined trajectory type")
    traj_group.add_argument("--trajectory_file", help="Path to JSON trajectory file")

    parser.add_argument("--angle", type=float, help="Override rotation angle for pan/tilt/orbit")
    parser.add_argument("--distance", type=float, help="Override translation distance for move/push/pull/boom")
    parser.add_argument("--orbit_radius", type=float, help="Override orbit radius")
    parser.add_argument("--traj_mode", choices=["relative", "global"], default="relative", help="Trajectory mode")
    parser.add_argument("--zoom_ratio", type=float, default=1.0, help="Zoom factor for zoom_in/zoom_out")
    parser.add_argument("--validate_only", action="store_true", help="Only validate trajectory file")

    parser.add_argument("--num_iterations", type=int, default=1, help="Number of continuous generative iterations")

    parser.add_argument("--input_path", help="Input video or image path")
    parser.add_argument("--output_path", default="outputs/inference.mp4", help="Output video path")
    parser.add_argument("--prompt", default="A smooth video with complete scene content. Inpaint any missing regions or margins naturally to match the surrounding scene.", help="Text prompt for generation")
    parser.add_argument("--negative_prompt", default="worst quality, low quality, blurry, pixelated, flickering, morphing, unnatural movement, temporal inconsistency, bad anatomy, deformed, poorly drawn face, poorly drawn hands, extra limbs, bad composition, out of frame, watermark, text, logo, artifacts, noise.", help="Negative text prompt")
    parser.add_argument("--model_path", default="models", help="Model directory path")
    parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt", help="Path to reconstructor checkpoint")
    parser.add_argument("--disable_lora", action="store_true", help="Skip distilled LoRA loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--height", type=int, default=336, help="Output height")
    parser.add_argument("--width", type=int, default=560, help="Output width")
    parser.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop", help="Video resize mode")
    parser.add_argument("--alpha_threshold", type=float, default=1.0, help="Alpha mask threshold (0.0-1.0)")
    parser.add_argument("--static_scene", action="store_true", help="Enable static scene mode")
    parser.add_argument("--vis_rendering", action="store_true", help="Save intermediate rendering visualizations")
    parser.add_argument("--low_vram", action="store_true", help="Enable low-VRAM mode")
    return parser.parse_args()

def main():
    args = parse_args()
    use_lora = not args.disable_lora
    num_inference_steps = 4 if use_lora else 50
    cfg_scale = 1.0 if use_lora else 5.0
    lora_path = os.path.join(
        args.model_path,
        "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
    ) if use_lora else None

    if args.validate_only:
        if args.trajectory_file is None:
            print("Error: --validate_only requires --trajectory_file")
            return 1
        data = CameraTrajectory.validate_json(args.trajectory_file)
        return 0

    if args.input_path is None: return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.trajectory:
        cam_traj = CameraTrajectory.from_predefined(
            args.trajectory, num_frames=args.num_frames, mode=args.traj_mode, angle=args.angle,
            distance=args.distance, orbit_radius=args.orbit_radius, zoom_ratio=args.zoom_ratio,
        )
    else:
        cam_traj = CameraTrajectory.from_json(args.trajectory_file)

    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=args.model_path, reconstructor_path=args.reconstructor_path,
        lora_path=lora_path, lora_alpha=1.0, torch_dtype=torch.bfloat16,
    ).to("cuda")

    images = load_video(args.input_path, args.num_frames, resolution=(args.width, args.height),
                        resize_mode=args.resize_mode, static_scene=args.static_scene)

    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if args.vis_rendering:
        vis_dir = os.path.splitext(output_path)[0]
        os.makedirs(vis_dir, exist_ok=True)
        pipe.save_root = vis_dir

    generate_video(
        pipe=pipe, input_video=images, prompt=args.prompt, negative_prompt=args.negative_prompt,
        cam_traj=cam_traj, output_path=output_path, alpha_threshold=args.alpha_threshold,
        static_flag=args.static_scene, seed=args.seed, cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps, num_iterations=args.num_iterations,
    )
    return 0

if __name__ == "__main__":
    exit(main())
