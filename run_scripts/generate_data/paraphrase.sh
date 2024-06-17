CUDA_VISIBLE_DEVICES=0 
python ./data_augmentation/generating_data.py \
--demo_file="./data/demos/paraphrase_demos.json" \
--eval_file="./data/qrecc/train.json" \
--output_file="./data/qrecc/paraphrase.jsonl" \
--model="meta-llama/Llama-2-7b-chat-hf" \
--temperature=0.75 \
--shot=1 \
--top_p=0.9