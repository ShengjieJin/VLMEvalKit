CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
torchrun \
    --nproc-per-node=7 \
    run.py \
    --data MathVerse_MINI_Text_Dominant \
    --mode all \
    --model Awaker2.5-VL \
    --verbose \
    --work-dir ./outputs