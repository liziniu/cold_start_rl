
set -x

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
SEED=42 
OUTPUT_DIR="./log/sft_gem-qwen2.5_3b-openr1_5k-${TIMESTAMP}-seed-${SEED}"
mkdir -p ${OUTPUT_DIR}

PORT=$(shuf -i 10000-65535 -n 1)

MAX_SEQ_LENGTH=2048
NUM_GPU=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=8
GRAD_ACC=$((BATCH_SIZE / NUM_GPU / PER_DEVICE_BATCH_SIZE))

deepspeed --master_port ${PORT} examples/train_sft.py \
    --deepspeed scripts/zero3_config.json \
    --seed $SEED \
    --model_name_or_path "Qwen/Qwen2.5-3B" \
    --tokenizer_name_or_path "./data/qwen2.5_simple_tokenizer" \
    --dataset_name ./data/openr1_5k \
    --dataset_train_split train \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_only_model True \
    --loss gem \
    --gem_beta 0.7 \
    --gem_h linear \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 2 \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --bf16 True \
    2>&1 | tee $OUTPUT_DIR/training.log

