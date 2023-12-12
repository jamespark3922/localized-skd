
# CHECKPOINT_DIR=$LAVIS_OUTPUT_DIR/BLIP2/Unified_train/ft_boxes_480_contrastive/vcr
# CHECKPOINT_DIR=/data/jamesp/LAVIS/output/BLIP2/Unified_train/ft_boxes_480_contrastive/sherlock_comparison_subset/
CHECKPOINT_DIR=output/BLIP2/Unified_qa_chatgpt_pretrain1_v2/boxes_480_contrastive/
MASTER_PORT=10010 

# sherlock evaluation
python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} evaluate.py --cfg-path lavis/projects/blip2/eval/unified_eval/sherlock_comparison.yaml  \
    --options model.pretrained=${CHECKPOINT_DIR}/checkpoint_best.pth run.output_dir=${CHECKPOINT_DIR}/sherlock_comparison

# vcr evaluation
python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} evaluate.py --cfg-path lavis/projects/blip2/eval/unified_eval/vcr_qa.yaml  \
    --options model.pretrained=${CHECKPOINT_DIR}/checkpoint_best.pth run.output_dir=${CHECKPOINT_DIR}/vcr_qa

# vcr evaluation
python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} evaluate.py --cfg-path lavis/projects/blip2/eval/unified_eval/vcr_qar.yaml  \
    --options model.pretrained=${CHECKPOINT_DIR}/checkpoint_best.pth run.output_dir=${CHECKPOINT_DIR}/vcr_qar

# visualcomet evaluation
python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} evaluate.py --cfg-path lavis/projects/blip2/eval/unified_eval/visualcomet_ranking.yaml  \
    --options model.pretrained=${CHECKPOINT_DIR}/checkpoint_best.pth run.output_dir=${CHECKPOINT_DIR}/visualcomet_ranking
