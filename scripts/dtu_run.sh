# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

DATA_ROOT="/home/user/data/data_DTU"
PATH_TO_OFFICIAL_DTU="/home/user/data/data_DTU/dtu_gt"

lanuch_exp() {
    local scene_id="$1"
    shift
    local output_dir="$1"
    shift
    local exp_args="$*"

    local scene_name=scan"$scene_id"

    python train.py --cfg_files cfg/dtu_mesh.yaml --source_path $DATA_ROOT/dtu_"$scene_name"/ --model_path $output_dir/$scene_name $exp_args --seunghun
    #python render.py $output_dir/$scene_name --eval_fps
    #python render.py $output_dir/$scene_name --skip_test --rgb_only --use_jpg
    #python render_fly_through.py $output_dir/$scene_name/

    
    # python extract_mesh.py $output_dir/$scene_name/ --save_gpu --use_vert_color --init_lv 8 --final_lv 10 --mesh_fname mesh_dense

    # mkdir -p $output_dir/$scene_name/mesh/latest/evaluation
    # python scripts/dtu_clean_for_eval.py $DATA_ROOT/dtu_"$scene_name"/ \
    #         $output_dir/$scene_name/mesh/latest/mesh_dense.ply
    # python scripts/dtu_eval/eval.py \
    #     --data $output_dir/$scene_name/mesh/latest/mesh_dense_cleaned_for_eval.ply \
    #     --scan $scene_id --dataset_dir $PATH_TO_OFFICIAL_DTU \
    #     --vis_out_dir $output_dir/$scene_name/mesh/latest/evaluation
    # rm -r $output_dir/$scene_name/checkpoints/
    
}


for scene in 37 40 55 63 65 69 83 # 24 97 105 106 110 114 118 122
do
    echo "============ start " $scene " ============"
    lanuch_exp $scene $1 "${@:2}"
    echo "============ end " $scene " ============"
done
