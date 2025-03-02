#!/bin/bash

# Setup environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Save training script
cat > land_cover_vit_training.py << 'EOF'
# Copy the entire content of the first artifact here
EOF

# Save inference script
cat > land_cover_vit_inference.py << 'EOF'
# Copy the entire content of the second artifact here
EOF

# Create log directory
mkdir -p logs

# Run distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=2
export NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  land_cover_vit_training.py 2>&1 | tee logs/training_log.txt

# Run inference (after training completes)
# python land_cover_vit_inference.py