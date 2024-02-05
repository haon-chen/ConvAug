export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 \
--master_port 28509 \
./train/train_aug_cdr.py \
--model_path="castorini/ance-msmarco-passage" \
--dataset_path="./data/qrecc/train_augmented.json" \
--only_last_response=false \
--per_device_train_batch_size=12 \
--learning_rate=1e-5 \
--num_train_epochs=1 \
--max_seq_length=512 \
--max_grad_norm=1.0 \
--logging_steps=1 \
--save_strategy='epoch' \
--save_total_limit=4 \
--early_stop_epoch=5 \
--report_to="none" \
--log_level="info" \
--output_dir="./checkpoints/qrecc-conv_ance_aug" \
--negative_type='bm25_hard_neg' \
--negative_ratio=3 \
--aug_weight=1.0 \
--aug_temperature=0.0012 \
--aug_strategy token_deletion,turn_deletion_depend,reorder_depend,paraphrase,extend \
--neg_strategy entity,shift \
--sample_method difficulty \
--aug_sim_file "/data/qrecc/sims_aug_pair.pkl" \
--difficulty_file "/data/qrecc/conv_difficulty.pkl" \
--neg_aug_ratio=1 \