export CUDA_VISIBLE_DEVICES=0
python ./evaluate/dense_retrieval.py \
--model_path="./checkpoints/qrecc-conv_ance_aug" \
--data_path="./data/cast21/test.json" \
--qrel_path="./data/cast21/qrels.txt" \
--rel_threshold=2 \
--input_type="session" \
--only_last_response \
--max_seq_length=512 \
--dense_index_path="./data/cast21/ance" \
--output_path="./results/cast21-conv_ance_aug-concat" \
--force_emptying_dir \
