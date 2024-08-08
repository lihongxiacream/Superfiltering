from code_ifd.full_flow import Superfiltering_IFD

#尝试一次性跑通三个步骤
if __name__ == '__main__':
    instance = Superfiltering_IFD(data_path='/luankexin/lihongxia/LLAMA_Factory/data/moss003_qwen.jsonl',
                                  model_name_or_path='/luankexin/lihongxia/Superfiltering/gpt2_chinese',
                                  category=['None'], #category有的话在数据集中预先定义，无的话不用设置,superfiltering会按照类别进行筛选
                                  sample_rate=0.5,
                                  json_save_path='./test_result.jsonl',
                                  multi_data='Sharegpt')
    instance.run()