#!/bin/sh

#SBATCH --cluster=ub-hpc
###SBATCH --cluster=faculty

#SBATCH --partition=general-compute --qos=general-compute
###SBATCH --partition=scavenger --qos=scavenger

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

#SBATCH --gres=gpu:1
###SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
###SBATCH --gres=gpu:tesla_v100-pcie-16gb:2
###SBATCH --gres=gpu:nvidia_a16:12

#SBATCH --mem=40000

#SBATCH --job-name="train_vo_pvgo"

###SBATCH --output= "result_$(date +"%Y_%m_%d_%k_%M_%S").out"

###SBATCH --mail-user=taimengf@buffalo.edu
###SBATCH --mail-type=ALL

###SBATCH --requeue


export CUDA_VISIBLE_DEVICES=1

# data_dir=/user/taimengf/projects/cwx/tartanair/TartanAir/ocean/Hard/P001
data_dir=/home/data2/TartanAir/TartanAir_comb/ocean/Hard/P001
# data_dir=$1

loss_weight='(0.05,10,10,3)'
rot_w=1
trans_w=0.1
batch_size=8
lr=1e-6
epoch=13
start_epoch=1
train_portion=1

use_scale=false
if [ "$use_scale" = true ]; then
    exp_type='mono'
else
    exp_type='stereo'
fi

project_name=test_tartanair
# project_name=$2
train_name=${rot_w}Ra_${trans_w}ta_delayOptm_lr=${lr}_${loss_weight}_${exp_type}

echo "\n=============================================="
echo "project name = ${project_name}"
echo "train name = ${train_name}"
echo "data dir = ${data_dir}"
echo "=============================================="

if [ "$start_epoch" = 1 ]; then
    rm -r train_results/${project_name}/${train_name}
    rm -r train_results_models/${project_name}/${train_name}
fi
mkdir -p train_results/${project_name}/${train_name}
mkdir -p train_results_models/${project_name}/${train_name}

if [ "$use_scale" = true ]; then
    # mono: use gt scale
    python train.py \
        --result-dir train_results/${project_name}/${train_name} \
        --save-model-dir train_results_models/${project_name}/${train_name} \
        --project-name ${project_name} \
        --train-name ${train_name} \
        --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
        --batch-size ${batch_size} \
        --worker-num 2 \
        --data-root ${data_dir} \
        --start-frame 0 \
        --end-frame -1 \
        --train-epoch ${epoch} \
        --start-epoch ${start_epoch} \
        --print-interval 1 \
        --snapshot-interval 10 \
        --lr ${lr} \
        --loss-weight ${loss_weight} \
        --data-type tartanair \
        --fix-model-parts 'flow' 'stereo' \
        --rot-w ${rot_w} \
        --trans-w ${trans_w} \
        --train-portion ${train_portion} \
        --use-gt-scale
else
    # stereo: calc scale
    python train.py \
        --result-dir train_results/${project_name}/${train_name} \
        --save-model-dir train_results_models/${project_name}/${train_name} \
        --project-name ${project_name} \
        --train-name ${train_name} \
        --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
        --batch-size ${batch_size} \
        --worker-num 2 \
        --data-root ${data_dir} \
        --start-frame 0 \
        --end-frame -1 \
        --train-epoch ${epoch} \
        --start-epoch ${start_epoch} \
        --print-interval 1 \
        --snapshot-interval 10 \
        --lr ${lr} \
        --loss-weight ${loss_weight} \
        --data-type tartanair \
        --fix-model-parts 'flow' 'stereo' \
        --rot-w ${rot_w} \
        --trans-w ${trans_w} \
        --train-portion ${train_portion}
fi
