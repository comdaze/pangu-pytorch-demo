# torchrun --nproc_per_node 1 --nnodes 1 finetune/finetune_fully.py
# torchrun --nproc_per_node 1 --nnodes 1 finetune/lora_tune.py

torchrun --nproc_per_node 8 --nnodes 1 finetune/finetune_fully.py
# torchrun --nproc_per_node 8 --nnodes 1 finetune/lora_tune.py

# deepspeed --num_gpus=8 models/pangu_model_deepspeed.py 
# deepspeed --num_gpus=8 finetune/finetune_fully.py --use_deepspeed True

#  --num_workers 8
#  --load_pretrained True
#  --load_my_best False
#  --visualize True
#  --only_test True
#  --only_use_wind_speed_loss True
#  --use_deepspeed True