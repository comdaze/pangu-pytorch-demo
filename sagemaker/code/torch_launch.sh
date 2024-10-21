#!/bin/bash
WORKING_DIR=/opt/ml/code
SM_WORKING_DIR=/opt/ml/model

#The related information about multi-nodes cluster.
MASTER_HOST=$SM_MASTER
MASTER_ADDR=$SM_MASTER_ADDR
MASTER_PORT="23456"
NNODES="$NODE_NUMBER"
NODE_RANK="$NODE_INDEX"

GPUS_PER_NODE="$SM_NUM_GPUS"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SAVE_PATH="${SM_WORKING_DIR}/results"
LOG_FILE="${SAVE_PATH}/log.txt"
# DS_CONFIG="${WORKING_DIR}/deepspeed_config.json"

# EPOCHS=10
# BATCH_SIZE=1024
#model_id="decapoda-research/llama-7b-hf"
#model_id="pinkmanlove/llama-7b-hf"

# train_dataset_path='/opt/ml/input/data/train'
# test_dataset_path='/opt/ml/input/data/test'
# learning_rate=0.00001
# model_max_length=1536
# per_device_train_batch_size=1
# per_device_eval_batch_size=1

OPTS=""
# OPTS+=" --per_device_eval_batch_size ${per_device_eval_batch_size}"
# OPTS+=" --per_device_train_batch_size ${per_device_train_batch_size}"
# OPTS+=" --model_max_length ${model_max_length}"
# #OPTS+=" --model_name ${model_id}"
# OPTS+=" --distributed-backend nccl"
# OPTS+=" --learning_rate ${learning_rate}"
# OPTS+=" --training_dir ${train_dataset_path}"
# OPTS+=" --test_dir ${test_dataset_path}"
# OPTS+=" --deepspeed"
# OPTS+=" --deepspeed_config ${DS_CONFIG}"
# OPTS+=" --epochs ${EPOCHS}"
# OPTS+=" --batch_size ${BATCH_SIZE}"

# LINK_CMD="ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so.11.0 /opt/conda/lib/python3.10/site-packages/torch/lib/libcudart.so"
# echo ${LINK_CMD}
# ${LINK_CMD}

git clone https://github.com/whn09/pangu-pytorch ${WORKING_DIR}/pangu-pytorch
cd ${WORKING_DIR}/pangu-pytorch && pip install -r requirements.txt

CMD="torchrun ${DISTRIBUTED_ARGS} ${WORKING_DIR}/pangu-pytorch/finetune/finetune_fully.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
