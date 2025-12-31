
# export MASTER_ADDR="0.0.0.0"
export MASTER_PORT=$(shuf -i 10000-20000 -n 1) 

export LOGDIR=./logs
export out_dir="${LOGDIR}/debug/"



EVAL_EXPNAME="SegDINO3D_ScanNetv2"  #! Baseline_ScanNet200, SegDINO3D_ScanNetv2, SegDINO3D_ScanNet200

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port=${MASTER_PORT}  \
    train_3d.py \
    --config_file configs/prototypes/${EVAL_EXPNAME}.py \
    --work_dir ${out_dir} \
    --seed 10000 \
    --eval_only \
    --load_pretrained_ckpt checkpoint/${EVAL_EXPNAME}.pth \