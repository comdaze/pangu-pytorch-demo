# conda create -n pangu python=3.10
# conda activate pangu
# pip install -r requirements.txt

# ./download_era5.sh
# python convert_era5.py
# python models/onnx2torch.py

# pip install s5cmd
# s5cmd cp s3://datalab/nsf-ncar-era5/aux_data/* /opt/dlami/nvme/aux_data/
# s5cmd cp s3://datalab/nsf-ncar-era5/pretrained_model/* /opt/dlami/nvme/pretrained_model/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*.pt /opt/dlami/nvme/upper/

# torchrun --nproc_per_node 1 --nnodes 1 finetune/finetune_fully.py
# torchrun --nproc_per_node 1 --nnodes 1 finetune/lora_tune.py

# torchrun --nproc_per_node 8 --nnodes 1 finetune/finetune_fully.py
# torchrun --nproc_per_node 8 --nnodes 1 finetune/lora_tune.py

# deepspeed --num_gpus=8 models/pangu_model_deepspeed.py 
# deepspeed --num_gpus=8 finetune/finetune_fully.py --use_deepspeed True --only_use_wind_speed_loss True --use_custom_mask True

#  --num_workers 4
#  --num_workers 8
#  --load_pretrained True
#  --load_my_best False
#  --visualize True
#  --only_test True
#  --only_use_wind_speed_loss True
#  --use_custom_mask True
#  --use_deepspeed True


torchrun --nproc_per_node 8 --nnodes 1 finetune/finetune_fully.py --load_pretrained True --load_my_best False --only_use_wind_speed_loss True --use_custom_mask True
