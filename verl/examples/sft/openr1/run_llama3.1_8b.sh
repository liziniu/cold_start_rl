set -x

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/openr1_5k/train.parquet \
    data.val_files=./data/openr1_5k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.max_length=16384 \
    data.truncation=error \
    optim.lr=2e-5 \
    data.train_batch_size=128 \
    data.micro_batch_size=1 \
    model.partial_pretrain=/220049033/models/meta-llama/Llama-3.1-8B \
    model.tokenizer=/220049033/models/meta-llama/Llama-3.1-8B-Instruct \
    trainer.default_local_dir=./ckpts/openr1-sft-llama-3.1-8b \
    trainer.project_name=openr1-sft \
    trainer.experiment_name=openr1-sft-llama-3.1-8b \
    trainer.logger=['console'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
