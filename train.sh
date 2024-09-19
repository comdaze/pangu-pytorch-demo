# torchrun --nproc_per_node 1 --nnodes 1 finetune/finetune_fully.py
# torchrun --nproc_per_node 1 --nnodes 1 finetune/lora_tune.py

torchrun --nproc_per_node 4 --nnodes 1 finetune/finetune_fully.py
# torchrun --nproc_per_node 4 --nnodes 1 finetune/lora_tune.py
