#!/usr/bin/env python3
"""
Export dataset to 4DGS multipleview format.

Converts from our organized format to 4DGS-compatible structure:
- Flat image structure: camXX/frame_XXXXX.jpg
- Ready for 4DGS's multipleviewprogress.sh script

Usage:
    python scripts/export_for_4dgs.py --scene 3
    python scripts/export_for_4dgs.py --scene 9 --output custom_path
"""

import argparse
import sys
from pathlib import Path
import json
import shutil
from PIL import Image
from tqdm import tqdm


def export_to_4dgs_format(scene_num: int, dataset_dir: str, output_dir: str = None):
    """
    Export organized dataset to 4DGS multipleview format.
    
    Args:
        scene_num: Scene number
        dataset_dir: Path to organized dataset (dataset/sceneX/)
        output_dir: Path to output (data/multipleview/sceneX_name/)
    """
    
    dataset_dir = Path(dataset_dir)
    scene_dir = dataset_dir / f"scene{scene_num}"
    
    # Check if scene exists
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    
    # Load metadata
    metadata_path = scene_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path("data") / "multipleview" / metadata['scene_name']
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"Exporting Scene {scene_num} to 4DGS Format")
    print("="*70)
    print(f"Source: {scene_dir}")
    print(f"Output: {output_dir}")
    print(f"Scene: {metadata['scene_name']}")
    print(f"Frames: {metadata['total_frames']}")
    print(f"Cameras: {len(metadata['cameras'])}")
    print()
    
    # Convert images to 4DGS format: camXX/frame_XXXXX.jpg
    total_images = metadata['total_frames'] * len(metadata['cameras'])
    
    with tqdm(total=total_images, desc="Converting images") as pbar:
        for cam_idx, cam_name in enumerate(metadata['cameras']):
            # Create camera directory (cam00, cam01, ...)
            cam_dir = output_dir / f"cam{cam_idx:02d}"
            cam_dir.mkdir(exist_ok=True)
            
            # Copy and convert images
            for frame_idx in range(metadata['total_frames']):
                # Source: dataset/scene3/images/cam0/00000.png
                src_path = scene_dir / "images" / cam_name / f"{frame_idx:05d}.png"
                
                # Destination: data/multipleview/scene3_collision/cam00/frame_00001.jpg
                # Note: 4DGS uses 1-indexed frames
                dst_path = cam_dir / f"frame_{frame_idx+1:05d}.jpg"
                
                if src_path.exists():
                    # Convert PNG to JPG
                    img = Image.open(src_path)
                    if img.mode == 'RGBA':
                        # Convert RGBA to RGB
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                        img = rgb_img
                    img.save(dst_path, 'JPEG', quality=95)
                else:
                    print(f"Warning: Missing {src_path}")
                
                pbar.update(1)
    
    print(f"\n✓ Converted {total_images} images")
    
    # Copy ground truth data for reference (not used by 4DGS, but useful for evaluation)
    print("\nCopying ground truth data...")
    
    shutil.copy(
        scene_dir / "camera_intrinsics.json",
        output_dir / "camera_intrinsics_gt.json"
    )
    print("  ✓ camera_intrinsics_gt.json")
    
    shutil.copy(
        scene_dir / "camera_extrinsics.json",
        output_dir / "camera_extrinsics_gt.json"
    )
    print("  ✓ camera_extrinsics_gt.json")
    
    shutil.copy(
        scene_dir / "metadata.json",
        output_dir / "metadata.json"
    )
    print("  ✓ metadata.json")
    
    # Create README with instructions
    readme_content = f"""# {metadata['scene_name']} - 4DGS Training Dataset

## Scene Information

**Scene Number:** {scene_num}
**Description:** {metadata['description']}
**Duration:** {metadata['duration']}s
**Frames:** {metadata['total_frames']}
**Cameras:** {len(metadata['cameras'])}
**Resolution:** {metadata['resolution'][0]}x{metadata['resolution'][1]}
**FPS:** {metadata['fps']}

## What This Scene Tests

{metadata['description']}

**Objects:** {', '.join(metadata['objects'])}

## Training with 4DGS

### Step 1: Run COLMAP Reconstruction

First, navigate to the 4DGaussians repository and run their multipleview processing script:
```bash
cd /path/to/4DGaussians
bash multipleviewprogress.sh {metadata['scene_name']}
```

This will generate:
- `sparse_/` - COLMAP reconstruction
- `points3D_multipleview.ply` - Initial point cloud
- `poses_bounds_multipleview.npy` - Camera poses

### Step 2: Train 4DGS
```bash
python train.py \\
    -s ../4dgs-synthetic-dataset/data/multipleview/{metadata['scene_name']} \\
    --port 6017 \\
    --expname "{metadata['scene_name']}" \\
    --configs arguments/hypernerf/default.py \\
    --iterations 30000
```

### Step 3: Render Results
```bash
python render.py \\
    -s ../4dgs-synthetic-dataset/data/multipleview/{metadata['scene_name']} \\
    --expname "{metadata['scene_name']}" \\
    --configs arguments/hypernerf/default.py
```

### Step 4: Evaluate
```bash
python metrics.py \\
    -s ../4dgs-synthetic-dataset/data/multipleview/{metadata['scene_name']} \\
    --expname "{metadata['scene_name']}"
```

## Ground Truth Data

This dataset includes ground truth data for quantitative evaluation:

- `camera_intrinsics_gt.json` - True camera intrinsics from MuJoCo
- `camera_extrinsics_gt.json` - True camera poses per frame
- `metadata.json` - Scene metadata and object trajectories

You can use this to:
1. Compare COLMAP reconstruction accuracy
2. Evaluate temporal consistency
3. Measure object tracking quality
4. Compute reconstruction metrics with known ground truth

## Expected Challenges

Based on the scene design, 4DGS is expected to struggle with:
"""

    if scene_num == 3:
        readme_content += """
- **Multi-object collision:** Three objects colliding simultaneously
- **Chaotic motion:** Unpredictable bounce trajectories
- **Gaussian boundary preservation:** Spheres in close proximity
- **Temporal consistency:** Motion before, during, and after collision
"""
    elif scene_num == 9:
        readme_content += """
- **Topology changes:** Objects splitting and merging
- **Fixed Gaussian count assumption:** Method assumes constant primitive count
- **Appearance changes:** Two separate objects vs. one merged object
- **Temporal smoothness:** Discontinuous events (split/merge moments)
"""
    
    readme_content += """
## Dataset Structure
```
{scene_name}/
├── cam00/
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ...
├── cam01/
│   └── ...
├── sparse_/              (generated by multipleviewprogress.sh)
├── points3D_multipleview.ply  (generated by multipleviewprogress.sh)
├── poses_bounds_multipleview.npy  (generated by multipleviewprogress.sh)
├── camera_intrinsics_gt.json
├── camera_extrinsics_gt.json
└── metadata.json
```

## Citation

If you use this dataset, please cite:
```bibtex
@dataset{{4dgs_synthetic_2024,
  title={{4DGS Synthetic Testing Dataset}},
  author={{Your Name}},
  year={{2024}},
  note={{Synthetic dynamic scenes for testing 4D Gaussian Splatting}}
}}
```

And cite the original 4DGS paper:
```bibtex
@article{{wu20234dgaussians,
  title={{4D Gaussian Splatting for Real-Time Dynamic Scene Rendering}},
  author={{Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang}},
  journal={{arXiv preprint arXiv:2310.08528}},
  year={{2023}}
}}
```
""".format(scene_name=metadata['scene_name'])
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print("  ✓ README.md")
    
    # Summary
    print("\n" + "="*70)
    print("Export Complete!")
    print("="*70)
    print(f"\nDataset ready for 4DGS training: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. cd /path/to/4DGaussians")
    print(f"  2. bash multipleviewprogress.sh {metadata['scene_name']}")
    print(f"  3. python train.py -s ../4dgs-synthetic-dataset/data/multipleview/{metadata['scene_name']} ...")
    print("\nSee README.md in the output directory for detailed instructions.")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Export dataset to 4DGS multipleview format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/export_for_4dgs.py --scene 3
  python scripts/export_for_4dgs.py --scene 9 --output custom/path
  python scripts/export_for_4dgs.py --scene 3 --dataset-dir my_dataset
        """
    )
    
    parser.add_argument('-s', '--scene', type=int, required=True,
                       help='Scene number to export (e.g., 3 or 9)')
    parser.add_argument('-d', '--dataset-dir', type=str, default='dataset',
                       help='Source dataset directory (default: dataset)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output directory (default: data/multipleview/sceneX_name)')
    
    args = parser.parse_args()
    
    try:
        export_to_4dgs_format(args.scene, args.dataset_dir, args.output)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()