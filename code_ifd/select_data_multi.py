import json
import numpy as np
import argparse
from tqdm import tqdm
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='alpaca_data_gpt2_data.json')
    parser.add_argument("--json_save_path", type=str, default='alpaca_data_gpt2_data_10per.json')
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--filter_threash", type=float, default=1)
    parser.add_argument("--key_name", type=str, default='ifd_ppl',help='ifd_ppl')
    parser.add_argument("--category", type=str, nargs='*',default='math')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

    with open(args.json_data_path, "r") as f:
        json_data = json.load(f)
    sample_num = int(len(json_data)*args.sample_rate)
    #sample_num = int(52002*args.sample_rate)

    #计算一轮问答均分
    #定义一下category
    result_conv={}
    for i in args.category:
        result_conv[i]=[]
    #result_conv={'roleplay':[],'code':[],'math':[]}
    conv=-1
    for i,instruct in enumerate(json_data):
        if instruct['category']=='roleplay':
            prompt="You are an interactive assistant engaged in a role-playing scenario. Your role is to respond to user inquiries in a manner consistent with the character you are portraying. Always maintain a tone and style appropriate to the role, providing relevant and accurate information."
            input=instruct['instruction']
            output=instruct['output']
        elif instruct['category']=='code':
            prompt="You are a knowledgeable programming assistant. Your role is to help users with coding tasks, including writing code snippets, explaining code, debugging, and providing best practices. Always provide clear, detailed, and accurate responses."
            input=instruct['instruction']
            output=instruct['output']
        else:
            prompt="Answer the grade school math word problem below, using step-by-step problem-solving process. Print the final answer after."
            input=instruct['instruction']
            output=instruct['output']

        if instruct['ID']>conv:#新的一轮对话

            if result_conv[instruct['category']]!=[]:
                result_conv[instruct['category']][-1]['ifd_ppl']=sum(ppl)/len(ppl)

            ppl=[]
            result_conv[instruct['category']].append({"conversations":[],"system":prompt,'ifd_ppl':0})
            conv=instruct['ID']

        ppl.append(instruct['ifd_ppl'])
        result_conv[instruct['category']][-1]['conversations'].append({"from": "human", "value": input})
        result_conv[instruct['category']][-1]['conversations'].append({"from": "gpt", "value": output})
    result_conv[instruct['category']][-1]['ifd_ppl'] = sum(ppl) / len(ppl)
    
    def sort_key(x):
        # Check if the value is nan
        if math.isnan(x[args.key_name]):
            return (0, 0) 
        return (1, x[args.key_name])
    
    print(result_conv['math'][-1])
    print(len(result_conv['roleplay']),len(result_conv['code']),len(result_conv['math']))

    #final_result={'roleplay':[],'code':[],'math':[]}
    final_result={}
    for i in args.category:
        final_result[i]=[]
    for i in result_conv:
        sample_num=int(len(result_conv[i])*args.sample_rate)
        result=result_conv[i]
        filtered_data = [x for x in result if (isinstance(x[args.key_name], (int, float)) and x[args.key_name] < args.filter_threash)]
        new_data = sorted(filtered_data, key=sort_key, reverse=True)
        new_data = new_data[:sample_num]
        final_result[i]=new_data

    print(len(final_result))

    with open(args.json_save_path, 'w') as file:
        json.dump(final_result, file, indent=4, ensure_ascii=False)
    
    print('Done: Data Selection:',args.json_data_path)


if __name__ == '__main__':
    main()