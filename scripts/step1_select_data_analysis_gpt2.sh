#step1 计算IDF
CUDA_VISIBLE_DEVICES=0 python code_ifd/data_analysis.py \
    --data_path data/math_final.jsonl \
    --save_path scores_math.jsonl \
    --model_name_or_path /lihongxia/Superfiltering/gpt2_chinese \
    --max_length 1024

#step2 IDF和数据匹配
CUDA_VISIBLE_DEVICES=0 python code_ifd/put_analysis_to_data.py \
    --pt_data_path scores_math.jsonl \
    --json_data_path data/math_final.jsonl \
    --json_save_path data_math.jsonl

#step3 基于IDF的数据筛选  单轮问答 or 多轮问答需带ID
python code_ifd/select_data_multi.py \
    --json_data_path data_math.jsonl \
    --json_save_path data_math_sort.json \
    --sample_rate 0.1 \
    --category  math

#step3 基于IDF和Facility Location筛选数据
CUDA_VISIBLE_DEVICES=0 python code_diversity_fla/do_fla.py \
    --json_data_path moss_data_gpt2_data_chinese.json \
    --json_save_path moss_data_gpt2_data_fla_10400_1040.json \
    --ifd_num 10400 \
    --fla_num 1040
