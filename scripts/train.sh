export CUDA_VISIBLE_DEVICES=0

conda activate stdc-seg-3_6_6

python -m torch.distributed.launch \
--nproc_per_node=1 train.py \
--respath checkpoints/train_STDC1-Seg/ \
--backbone STDCNet813 \
--mode train \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet813M_73.91.tar
