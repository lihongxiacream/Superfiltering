#pip install -r requirements.txt

cd lihongxia/Superfiltering
source super_env/bin/activate

#计算IFD分数
bash scripts/step1_select_data_analysis_gpt2.sh

#IFD分数和原始数据匹配
bash scripts/step2_put_analysis_to_data.sh

#选择前x%的数据
bash scripts/step3_select_data.sh

#Superfiltering.D
bash scripts/optional_select_data_plus_diversity.sh

#GPT4评估
#比较单个模型
bash evaluation/scripts/do_eval_generation.sh
#比较pair模型
bash evaluation/scripts/do_eval_generation_wrap.sh
#可视化
bash evaluation/scripts/do_review_eval_score.sh