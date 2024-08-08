
CUDA_VISIBLE_DEVICES=0 python code_ifd/put_analysis_to_data.py \
    --pt_data_path moss_data_gpt2_scores_chinese2.jsonl \
    --json_data_path data/moss003_cherry_before.jsonl \
    --json_save_path moss_data_gpt2_data_chinese2.json

