export CUDA_VISIBLE_DEVICES=0
python ./evaluate/dense_retrieval.py \
--model_path="./checkpoints/qrecc-conv_ance_aug" \
--data_path="./data/qrecc/test.json" \
--qrel_path="./data/qrecc/qrels.txt" \
--rel_threshold=1 \
--input_type="session" \
--only_last_response \
--max_seq_length=512 \
--dense_index_path="./data/qrecc/ance" \
--output_path="./results/qrecc-conv_ance_aug-concat" \
--force_emptying_dir \
