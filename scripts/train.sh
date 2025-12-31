
export MASTER_PORT=$(shuf -i 10000-20000 -n 1) 

export EXPNAME="SegDINO3D_ScanNet200"       # SegDINO3D_ScanNet200 or SegDINO3D_ScanNetv2 
export CONFIGNAME="SegDINO3D_ScanNet200"    # SegDINO3D_ScanNet200 or SegDINO3D_ScanNetv2 
export GPUID=0
export LOGDIR=./logs
export EXPDIR="${LOGDIR}/${EXPNAME}/"

echo "Experiment: ${EXPDIR}, Config: ${CONFIGNAME}, at GPU-${GPUID}"
mkdir $EXPDIR

if [[ "$CONFIGNAME" == *"200"* ]]; then
    PRETRAINED_CKPT="pretrained_backbone/mask3d_scannet200_aligned.pth"
else
    PRETRAINED_CKPT="pretrained_backbone/aligned_sstnet_scannet.pth"
fi
echo "Using pretrained checkpoint: ${PRETRAINED_CKPT}"


CUDA_VISIBLE_DEVICES=${GPUID} python -m torch.distributed.launch \
    --master_port=${MASTER_PORT}  \
    train_3d.py \
    --config_file configs/prototypes/${CONFIGNAME}.py \
    --work_dir ${EXPDIR} \
    --load_pretrained_ckpt ${PRETRAINED_CKPT} \
    --seed 10000 \
    >> ${EXPDIR}/training_log.log 2>&1
    