export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 \
--master_port 28510 \
./train/train_aug_cdr.py \
--model_path="castorini/ance-msmarco-passage" \
--dataset_path="./data/topiocqa/train_augmented.json" \
--only_last_response=false \
--per_device_train_batch_size=12 \
--learning_rate=1.5e-5 \
--num_train_epochs=1 \
--max_seq_length=512 \
--max_grad_norm=1.0 \
--logging_steps=1 \
--save_strategy='epoch' \
--save_total_limit=4 \
--early_stop_epoch=5 \
--report_to="none" \
--log_level="info" \
--output_dir="./checkpoints/topiocqa-conv_ance_aug" \
--negative_type='bm25_hard_neg' \
--negative_ratio=5 \
--aug_weight=0.1 \
--aug_temperature=0.001 \
--aug_strategy token_deletion,turn_deletion_depend,reorder_depend,paraphrase,extend \
--neg_strategy entity,shift \
--sample_method difficulty \
--aug_sim_file "/data/topiocqa/sims_aug_pair.pkl" \
--difficulty_file "/data/topiocqa/conv_difficulty.pkl" \
--neg_aug_ratio=1 \