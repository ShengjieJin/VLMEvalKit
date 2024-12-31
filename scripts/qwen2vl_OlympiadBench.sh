export LMUData=~/datasets/LMUData
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nproc-per-node=1 \
    run.py \
    --data OlympiadBench \
    --mode all \
    --model Qwen2-VL-7B-Instruct \
    --verbose \
    --work-dir ./outputs