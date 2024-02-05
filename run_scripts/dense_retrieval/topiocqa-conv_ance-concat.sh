export CUDA_VISIBLE_DEVICES=0
python ./evaluate/dense_retrieval.py \
--model_path="./checkpoints/topiocqa-conv_ance_aug" \
--data_path="./data/topiocqa/test.json" \
--qrel_path="./data/topiocqa/qrels.txt" \
--rel_threshold=1 \
--input_type="session" \
--only_last_response \
--max_seq_length=512 \
--dense_index_path="./data/topiocqa/ance" \
--output_path="./results/topiocqa-conv_ance_aug-concat" \
--force_emptying_dir \
