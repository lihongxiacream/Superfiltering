import os
import json
import torch
import argparse
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
class Superfiltering_IFD():
    def __init__(self, data_path,model_name_or_path,category=['role','math','code'],sample_rate=0.1,json_save_path='./filter_result.jsonl',multi_data='Alpaca'):

        self.data_path = data_path
        self.model_name_or_path = model_name_or_path
        self.category = category #要筛选几组数据
        self.sample_rate = sample_rate #每组数据要多少比例
        self.json_save_path = json_save_path
        self.data_type = multi_data #Alpaca or Sharegpt 单轮还是多轮
        self.PROMPT_DICT_NONE = {
        "prompt_input": (
            "{instruction}\n{input}\n"
        ),
        "prompt_no_input": (
            "{instruction}\n"
        )}

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def parse_args(self):
        parser = argparse.ArgumentParser()
        #parser.add_argument("--data_path", type=str, default='data/alpaca_data/alpaca_data.json')
        parser.add_argument("--save_path", type=str, default='debug.jsonl')
        #parser.add_argument("--model_name_or_path", type=str, default='gpt2')
        parser.add_argument("--max_length", type=int, default=1024)
        parser.add_argument("--start_idx", type=int, default=0)
        parser.add_argument("--end_idx", type=int, default=-1)
        parser.add_argument("--prompt", type=str, default='none', help='none')
        parser.add_argument("--key_name", type=str, default='ifd_ppl', help='ifd_ppl')
        parser.add_argument("--filter_threash", type=float, default=1)
        args = parser.parse_args()
        return args

    # Used to get the ppl and emb for the whole input
    def get_perplexity_and_embedding_whole_text(self,tokenizer, model, text, max_length):

        try:
            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids.contiguous())
            loss = outputs.loss
            perplexity = torch.exp(loss)
            return perplexity.to('cpu').item(), loss.to('cpu').item()

        except:
            return 0, 0

    # Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
    def get_perplexity_and_embedding_part_text(self,tokenizer, model, text, target_span, max_length):

        try:
            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)

            start_index = text.rfind(target_span)
            start_token = len(tokenizer.encode(text[:start_index]))
            end_token = input_ids.shape[1]

            labels = input_ids.clone()
            labels[0, :start_token] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=labels)

            loss = outputs.loss
            perplexity = torch.exp(loss)

            return perplexity.to('cpu').item(), loss.to('cpu').item()

        except:
            return 0, 0

    #对于多轮问答数据进行预处理，整理成单轮问答格式计算ppl
    def preprocess_multi(self,data):

        print("------------开始将多轮问答数据转化为单轮问答数据-------------")
        # 将所有数据拆分为单轮问答
        result = []
        for i, conv in tqdm(enumerate(data)):
            for j in conv['conversations']:
                if j['from'] == 'human':
                    result.append({})
                    if conv.get('system'):
                        result[-1]['instruction'] = conv['system']
                        result[-1]['input'] = j['value']
                    else:
                        result[-1]['instruction'] = j['value']
                        result[-1]['input'] = ""
                else:
                    result[-1]['output'] = j['value']
                    result[-1]['ID'] = i
                    result[-1]['category'] = conv['category']
        print(len(data),len(result))
    def select_data_multi(self,json_data):
        args = self.parse_args()
        print("------------开始基于IFD值筛选多轮问答数据-------------")
        #计算一轮问答均分
        # 定义一下category
        result_conv = {}
        for i in self.category:
            result_conv[i] = []
        # result_conv={'roleplay':[],'code':[],'math':[]}
        conv = -1
        for i, instruct in enumerate(json_data):
            if instruct['category'] == 'roleplay':
                prompt = "You are an interactive assistant engaged in a role-playing scenario. Your role is to respond to user inquiries in a manner consistent with the character you are portraying. Always maintain a tone and style appropriate to the role, providing relevant and accurate information."

            elif instruct['category'] == 'code':
                prompt = "You are a knowledgeable programming assistant. Your role is to help users with coding tasks, including writing code snippets, explaining code, debugging, and providing best practices. Always provide clear, detailed, and accurate responses."

            elif instruct['category'] == 'math':
                prompt = "Answer the grade school math word problem below, using step-by-step problem-solving process. Print the final answer after."

            if instruct['input'] != '':
                prompt = instruct['instruction']
                input = instruct['input']
                output = instruct['output']
            else:
                input = instruct['instruction']
                output = instruct['output']

            if instruct['ID'] > conv:  # 新的一轮对话

                if result_conv[instruct['category']] != []:
                    result_conv[instruct['category']][-1]['ifd_ppl'] = sum(ppl) / len(ppl)

                ppl = []
                result_conv[instruct['category']].append({"conversations": [], "system": prompt, 'ifd_ppl': 0})
                conv = instruct['ID']

            ppl.append(instruct['ifd_ppl'])
            result_conv[instruct['category']][-1]['conversations'].append({"from": "human", "value": input})
            result_conv[instruct['category']][-1]['conversations'].append({"from": "gpt", "value": output})
        result_conv[instruct['category']][-1]['ifd_ppl'] = sum(ppl) / len(ppl)

        def sort_key(x):
            # Check if the value is nan
            if math.isnan(x[args.key_name]):
                return (0, 0)
            return (1, x[args.key_name])

        print(len(result_conv))
        # print(result_conv['math'][-1])
        # print(len(result_conv['roleplay']), len(result_conv['code']), len(result_conv['math']))

        # final_result={'roleplay':[],'code':[],'math':[]}
        final_result = []
        # for i in args.category:
        #     final_result[i] = []
        for i in result_conv:
            sample_num = int(len(result_conv[i]) * self.sample_rate)
            result = result_conv[i]
            filtered_data = [x for x in result if
                             (isinstance(x[args.key_name], (int, float)) and x[args.key_name] < args.filter_threash)]
            new_data = sorted(filtered_data, key=sort_key, reverse=True)
            new_data = new_data[:sample_num]
            final_result += new_data

        #print(len(final_result))

        with open(self.json_save_path, 'w') as file:
            json.dump(final_result, file, indent=4, ensure_ascii=False)

        print('Done: Data Selection:', self.json_save_path)

    def select_data(self,json_data):
        args = self.parse_args()
        print("----------开始IFD值筛选单轮问答数据-------------")

        def sort_key(x):
            # Check if the value is nan
            if math.isnan(x[args.key_name]):
                return (0, 0)
            return (1, x[args.key_name])

        filtered_data = [x for x in json_data if
                         (isinstance(x[args.key_name], (int, float)) and x[args.key_name] < args.filter_threash)]
        new_data = sorted(filtered_data, key=sort_key, reverse=True)

        sample_num = int(len(new_data) * self.sample_rate)
        new_data = new_data[:sample_num]
        print(len(new_data))
        with open(args.json_save_path, 'w') as file:
            for i in new_data:
                file.write(json.dumps(i, ensure_ascii=False) + '\n')

        print('Done: Data Selection:', args.json_data_path)

    def run(self):

        args = self.parse_args()
        print(args)

        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map="auto", cache_dir='../cache', output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir='../cache')

        model.eval()

        #判断一下输入是否是列表 不是列表的话按行读入
        with open(self.data_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data=[]
            with open(self.data_path, "r") as f:
                for line in f:
                    data.append(json.loads(line.strip()))

        #判断是否是多轮问答数据，是的话预处理成单轮问答
        if self.data_type=="sharegpt":
            data=self.preprocess_multi(data)

        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx != -1 else len(data)
        sampled_data = data[start_idx:end_idx]

        # if not os.path.exists(args.save_path):
        #     with open(args.save_path, "w") as file:
        #         pass  # Creates an empty file
        #
        # with open(args.save_path, "r") as file:
        #     exsisting_num =  sum(1 for _ in file)
        # sampled_data = sampled_data[exsisting_num:]


        if args.prompt == 'none':
            prompt_no_input = self.PROMPT_DICT_NONE["prompt_no_input"]
            prompt_input = self.PROMPT_DICT_NONE["prompt_input"]

        print("-------开始计算指令的IFD值----------")
        pt_data = []
        for i in tqdm(range(len(sampled_data))):

            data_i = sampled_data[i]
            instruct_i = data_i['instruction']
            output_i = data_i['output']

            input_i = data_i['input'] if 'input' in data_i.keys() else ''
            if input_i == '':
                temp_dict = {'instruction':instruct_i}
                promt_to_use = prompt_no_input.format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use

            else:
                temp_dict = {'instruction':instruct_i,'input':input_i}
                promt_to_use = prompt_input.format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use


            instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            instruct_i_len = instruct_i_input_ids.shape[1]

            if output_i == '':
                temp_data_i = {}
            else:
                ppl_out_alone, loss_out_alone = self.get_perplexity_and_embedding_whole_text(tokenizer, model, output_i, args.max_length-instruct_i_len+1)
                ppl_out_condition, loss_out_condition = self.get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, args.max_length)

                temp_data_i = {}
                temp_data_i['ppl'] = [0,ppl_out_alone,0,ppl_out_condition]
                temp_data_i['loss'] = [0,loss_out_alone,0,loss_out_condition]

            pt_data.append(temp_data_i)

            assert len(data) == len(pt_data)
            #put analysis to data
            new_data = []
            for i in tqdm(range(len(pt_data))):

                json_data_i = data[i]

                pt_data_i = pt_data[i]
                if pt_data_i == {}:
                    ppl_Q_direct, ppl_A_direct, ppl_Q_condition, ppl_A_condition = np.nan, np.nan, np.nan, np.nan
                    loss_Q_direct, loss_A_direct, loss_Q_condition, loss_A_condition = np.nan, np.nan, np.nan, np.nan
                else:
                    ppl_Q_direct, ppl_A_direct, ppl_Q_condition, ppl_A_condition = \
                        pt_data_i['ppl'][0], pt_data_i['ppl'][1], pt_data_i['ppl'][2], pt_data_i['ppl'][3]
                    loss_Q_direct, loss_A_direct, loss_Q_condition, loss_A_condition = \
                        pt_data_i['loss'][0], pt_data_i['loss'][1], pt_data_i['loss'][2], pt_data_i['loss'][3]

                json_data_i['ppl_A_direct'] = ppl_A_direct
                json_data_i['ppl_A_condition'] = ppl_A_condition
                try:
                    json_data_i['ifd_ppl'] = ppl_A_condition / ppl_A_direct
                except ZeroDivisionError:
                    json_data_i['ifd_ppl'] = 0

                new_data.append(json_data_i)
            # with open(args.save_path, "a") as file:
            #     file.write(json.dumps(temp_data_i) + '\n')

        if self.data_type=='Sharegpt':
            self.select_data_multi(new_data)
        else:
            self.select_data(new_data)
        print('Done: Data Analysis:',args.data_path)