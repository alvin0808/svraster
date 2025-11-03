# SVR for ScanNet++ dataset

We now support scannet++ dataset. The [benchmark results](https://kaldir.vc.in.tum.de/scannetpp/benchmark/nvs) on 3rd-party evaluated hidden set is (at the time of 8 Mar, 2025):
<img width="700" alt="scannet++ benchmark" src="https://github.com/user-attachments/assets/3ef905e9-bc86-4d31-87bb-33b9a8bad56c" />

https://github.com/user-attachments/assets/85f55a12-b4bb-4581-924e-925a38f6a748

More results information (averaged on 50 scenes):
- Per-scene optimization time: `12 mins`.
- FPS: `197` at `1752 x 1168` image resolution. As we use `ss=1.5`, the actual rendering resolution is `2628 x 1752`.
- Voxel size distribution:
    | <3mm | 3mm-5mm | 5mm-1cm | 1cm-2cm | 2cm-3cm | >3cm | 
    | :-: | :-: | :-: | :-: | :-: | :-: |
    | 13.61% | 19.25% | 32.43% | 23.31% | 6.66% | 4.73% |
- Sparse points from COLMAP is not used in the submitted version. We later find sparse points loss helpful for geometry and slightly improve quality on the public set. Activate it by `--lambda_sparse_depth 1e-2` when running `train.py`.

### Data preparation
1. Download the source data following the procedure in [scannet++ official site](https://kaldir.vc.in.tum.de/scannetpp/).
2. Run `python scripts/scannetpp_preproc.py --indir $PATH_TO_SOURCE_DATA --outdir data/scannetpp_nvs --ids $SEQUENCE_OF_SCENE_ID`.

### Optimization configuration
The config file is provided in `cfg/scannetpp.yaml`. We detail the setting as follow.

**Scene bound heuristic.**
As this is a fully indoor dataset, we set `outside_level` to zero and assume the entire scene is inside the main scene bound. The world center is set to the centroid of training cameras and the scene radius is set to two times the maximum distance from world center to the cameras.

**SH reset trick.**
We find the view-dependent color from SH is not generalized well so we implement a trick by resetting the sh component near the end of optimization. This trick improve quality on the view "extrapolation" task like ScanNet++ dataset, while it slightly reduces quality on view "interpolation" task like mipnerf360.

**Density ascending regularizer.**
It encourages the derived normal from the density field to point toward the camera side. It improves geometry qualitatively and slightly improve quantitative result.

**Sparse point depth loss.**
It's not used in the submitted version. On the public set, it improves geometry qualitatively and novel-view results quantitatively.

<img width="512" alt="scannet++ benchmark" src="https://github.com/user-attachments/assets/33b9f955-425d-490f-8e9e-0183957522f6" />
