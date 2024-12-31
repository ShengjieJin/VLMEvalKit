export LMUData=~/datasets/LMUData
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nproc-per-node=1 \
    run.py \
    --data OlympiadBench \
    --mode all \
    --model Awaker2.5-VL \
    --verbose \
    --work-dir ./outputs