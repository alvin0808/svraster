# SVRecon: Sparse Voxel Rasterization for Surface Reconstruction

![teaser](./asset/fig1_teaser.jpg)

### [Project page](https://jaesung-choe.github.io/svrecon/index.html) | [Arxiv](https://arxiv.org/abs/2511.17364) 

<details>
<summary>Paper abstract</summary>
We extend the recently proposed sparse voxel rasterization paradigm to the task of high-fidelity surface reconstruction by integrating Signed Distance Function (SDF), named SVRecon. Unlike 3D Gaussians, sparse voxels are spatially disentangled from their neighbors and have sharp boundaries, which makes them prone to local minima during optimization. Although SDF values provide a naturally smooth and continuous geometric field, preserving this smoothness across independently parameterized sparse voxels is nontrivial. To address this challenge, we promote coherent and smooth voxel-wise structure through (1) robust geometric initialization using a visual geometry model and (2) a spatial smoothness loss that enforces coherent relationships across parent-child and sibling voxel groups. Extensive experiments across various benchmarks show that our method achieves strong reconstruction accuracy while having consistently speedy convergence.
</details>

**Updates:**
- This project is under active development and not ready for stable use yet. We will update instructions and benchmarks soon.

## Install
```bash
# 1) Clone (with PI3)
git clone --recursive https://github.com/alvin0808/SVRecon.git
cd SVRecon
# If you cloned without --recursive: git submodule update --init --recursive

# 2) Conda env
conda create -n svrecon python=3.10 -y
conda activate svrecon

# 3) PyTorch (tested: 2.5.0 + CUDA 12.4)
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# 4) Optional CUDA toolkit inside conda
conda install -y -c "nvidia/label/cuda-12.4.0" cuda-toolkit

# 5) Python deps (includes PI3 deps)
pip install -r requirements.txt

# 6) Build and install sparse voxel CUDA rasterizer
pip install -e cuda/ --no-build-isolation

```
<!-- ### 1) Clone this repository (with PI3 submodule)
```bash
git clone --recursive https://github.com/alvin0808/SVRecon.git
cd SVRecon

# If you already cloned without --recursive:
git submodule update --init --recursive
```
### 2) Create conda environment
```bash
conda create -n svrecon python=3.10 -y
conda activate svrecon
```
### 3) Install PyTorch (tested: 2.5.0 + CUDA 12.4)
```bash
pip install torch==2.5.0 torchvision==0.20.0 \
  --index-url https://download.pytorch.org/whl/cu124
```
### 4) (Optional) Install CUDA toolkit inside the conda env

May need to install cuda-toolkit for your virtual environment that is aligned with the installed pytorch:
```bash
conda install -y -c "nvidia/label/cuda-12.4.0" cuda-toolkit
```
### 5) Install Python dependencies (includes PI3 deps)
```bash
pip install -r requirements.txt
```

### 6) Build and install sparse voxel CUDA rasterizer

If the build process cannot find torch due to build isolation, use --no-build-isolation.
```bash
pip install -e cuda/ --no-build-isolation
``` -->



## Reconstructing your own capture
Below go through the workflow for reconstruction from a scene capturing. 


### Data preparation
We recommend to follow [InstantNGP](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md#colmap) video or images processing steps to extract camera parameters using COLMAP. [NerfStudio](https://docs.nerf.studio/quickstart/custom_dataset.html) also works.

We now only support pinhole camera mode. Please preprocess with `--colmap_camera_model PINHOLE` of InstantNGP script or `--camera-type pinhole` of NerfStudio script.

### Pi3 initialization 

Before training, you can optionally run Pi3 to generate an initial aligned point cloud from your COLMAP reconstruction.

#### Common arguments

- `--data_root`  
  Path to the image folder used by Pi3. The folder should contain your input RGB images.

- `--model`  
  Path to Pi3 weights, e.g. `model.safetensors`.

- `--out_dir`  
  Output directory for intermediate Pi3 exports, including:
  - `camera_poses_{i}.pt` (poses per split)
  - `point_cloud_final_sor_filtered_{i}.ply` (point clouds per split)

- `--out_ply`  
  Output path for the final merged aligned point cloud (`.ply`).

- `--interval`  
  Split factor for long sequences (useful when you have many images, e.g. >50).  
  Pi3 will export results multiple times with different starting offsets:
  - `i = 0, 1, ..., interval-1`
  - each split uses frames: `i, i+interval, i+2*interval, ...`  
  Larger `interval` reduces per-run memory/time and makes the pipeline more robust on long sequences.

- `--sample_rate`  
  Random keep ratio for the final merged `.ply`.  
  Example: `--sample_rate 0.5` keeps ~50% of the aligned points (useful to reduce file size).

---
#### Run Pi3 export + COLMAP pose alignment (one command)

```bash
python scripts/pi3/run_export_and_align_colmap.py \
  --data_root /path/to/your_dataset/images \
  --model /path/to/model.safetensors \
  --out_dir output/pi3_export_colmap \
  --gt_bin /path/to/your_dataset/sparse/0/images.bin \
  --out_ply /path/to/your_dataset/sparse/0/aligned_points3D.ply \
  --interval 2 \
  --conf_th 0.05 \
  --sample_rate 0.5
```
#### COLMAP format requirements

Your dataset directory result should look like:
```text
your_dataset/
├── images/                 # input images
├── conf/                   # confidence outputs
├── normal/                 # normal outputs
└── sparse/
    └── 0/
        ├── images.bin      # COLMAP registered images (extrinsics)
        ├── cameras.bin
        └── aligned_points3D.ply   # final aligned point cloud (generated)
```

- --gt_bin must point to sparse/0/images.bin.
#### Run Pi3 export + Nerf pose alignment (one command)
````bash
python scripts/pi3/run_export_and_align_nerf.py \
  --data_root /path/to/your_nerf_dataset/image \
  --gt_json /path/to/your_nerf_dataset/transforms_train.json \
  --model /path/to/model.safetensors \
  --out_dir output/pi3_export_nerf \
  --out_ply /path/to/your_nerf_dataset/aligned_points3D.ply \
  --interval 1 \
  --sample_rate 0.5
````
#### NeRF (transforms_*.json) format requirements

Your dataset directory result should look like:
```text
your_nerf_dataset/
├── image/                 # input images
├── conf/                   # confidence outputs
├── normal/                 # normal outputs
├── transforms_train.json   # GT poses (OpenGL c2w)
├── transforms_test.json    # (optional)
└── aligned_points3D.ply    # final aligned point cloud (generated)
```
### Scene optimization
```bash
python train.py --eval --source_path $DATA_PATH --model_path $OUTPUT_PATH
```
All the results will be saved into the specified `$OUTPUT_PATH` including the following results:
- `config.yaml`: The config file for reproduction.
- `pg_view/`: Visualization of the training progress. Useful for debugging.
- `test_stat/`: Some statistic during the training.
- `test_view/`: Some visualization during the training.

The configuration is defined by the following three, the later overwrites the former.
- `src/config.py`: Define the configuable setup and their initial values.
- `--cfg_files`: Sepcify a list of config files, the later overwrites the former. Some examples are under `cfg/`.
- command line: Any field defined in `src/config.py` can be overwritten through command line. For instances: `--data_device cpu`, `--subdivide_save_gpu`.

Like InstantNGP and other NeRF variants, setting a proper main scene bounding box is important for both quality and speed. In SVRecon, it also determines where continuity loss is applied, since the continuity regularization is enforced only inside the main bound.

The main bound should cover the primary 3D region of interest. For rays and geometry outside this region, SVRecon allocates additional octree levels via --outside_level (default: 5). This is especially helpful when you do not provide foreground masks, so rays leaving the main bound can still be represented. You can control the bound behavior using the following options:
- `--bound_mode`:
    - `default`
        - Use the suggested bbox if given by dataset. Otherwise, it automatically chose from `forward` or `camera_median` modes.
    - `camera_median`
        - Set camera centroid as world origin. The bbox radius is set to the median distance between origin and cameras.
    - `camera_max`
        - Set camera centroid as world origin. The bbox radius is set to the maximum distance between origin and cameras.
    - `forward`
        - Assume [LLFF](https://github.com/Fyusion/LLFF?tab=readme-ov-file#local-light-field-fusion) forward-facing capturing. See `src/utils/bounding_utils.py` for detail heuristic.
    - `pcd`
        - Use COLMAP sparse points to compute a scene bound. See `src/utils/bounding_utils.py` for detail heuristic.
- `--bound_scale`: scaling the main scene bound (default 1).

For scenes with background masked out, use `--white_background` or `--black_background` to specify the background color.

Other hyperparameter suggestions:    

- Continuity losses for SDF geometry

    SVRecon uses the following continuity losses for SDF geometry.
    The default weights are already well tuned, so you usually do not need to change them.
    You can still adjust them if your scene requires stronger or weaker regularization.

    `--lambda_ge_density`
    Global Eikonal loss that enforces unit SDF gradient magnitude using dense sampling.

    `--lambda_vg_density`
    Local voxelwise Eikonal loss.
    This is called voxel gradient in the code and is mainly used in fine stages.

    `--lambda_ls_density`
    Laplacian smoothness loss that reduces high frequency discontinuities across voxel boundaries.

    `--lambda_normal_dmean --lambda_normal_dmed`
    Depth normal consistency regularizer that stabilizes geometry.
    If you discuss this term in your paper, you may cite 2DGS.

- Training iterations

    Use `--n_iter 8000` for smaller scenes such as DTU, and `--n_iter 10000` for larger scenes such as Tanks and Temples.


   
- `--save_quantized` to apply 8 bits quantization to the saved checkpoints. It typically reduce ~70% model size with minor quality difference.



### Meshing
After the scene optimization completed, choose one of the following:

#### TSDF fusion meshing
```bash
python extract_mesh.py $OUTPUT_PATH
```

#### Direct SDF meshing
```bash
python extract_mesh_sdf.py $OUTPUT_PATH
```

## Experiments on public dataset

**Note:** For fair comparison, keep the image preprocessing and downsampling protocol consistent across methods

### Download the 3rd-party processed datasets

- Mesh reconstruction
    - [DTU dataset](https://github.com/Totoro97/NeuS)
        - Check [scripts/dtu_preproc.py](./scripts/dtu_preproc.py) for pre-processing.
    - [Tanks&Temples dataset](https://github.com/hbb1/2d-gaussian-splatting)

### Running base setup
```bash
# Work in progress
```
<!-- exp_dir="baseline"
other_cmd_args=""

# Run training
./scripts/dtu_run.sh            output/dtu/baseline            $other_cmd_args
./scripts/tnt_run.sh            output/tnt/baseline            $other_cmd_args

# Summarize results
python scripts/dtu_stat.py            output/dtu/baseline
python scripts/tnt_stat.py            output/tnt/baseline
``` -->


## Acknowledgement
Our method is developed on the amazing open-source codebase: [SVRaster](https://github.com/NVlabs/svraster), [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization).

If you find our work useful in your research, please be so kind to give us a star and citing our paper.
```bibTeX
@article{oh2025svrecon,
  title={SVRecon: Sparse Voxel Rasterization for Surface Reconstruction},
  author={Oh, Seunghun and Choe, Jaesung and Lee, Dongjae and Lee, Daeun and Jeong, Seunghoon and Wang, Yu-Chiang Frank and Park, Jaesik},
  journal={arXiv preprint arXiv:2511.17364},
  year={2025}
}
```
