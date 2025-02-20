set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/openr1/train.parquet \
    data.val_files=./data/openr1/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    optim.lr=2e-5 \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=openr1-sft \
    trainer.experiment_name=openr1-sft-qwen-2.5-7b-instruct \
    trainer.logger=['console'] \
    trainer.total_training_steps=1 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
