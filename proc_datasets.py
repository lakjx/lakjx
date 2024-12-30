import json
import random
import re
# def create_training_sample(state, action):
#         # 构造输入文本
#     input_text = f"""System state:
#     - UEs: {state['num_UEs']}
#     - Time step: {state['time_step']}
#     - UE buffer: {describe_buffer_state(state['ue_obs_history'][-1])}
#     - Last UE PHY actions: {describe_actions(state['ue_actions_history'][-1])}
#     - Last UE UCM: {describe_ucm_actions(state['ue_msg_history'][-1])}
#     - Last BS DCM: {describe_dcm_actions(state['bs_msg_history'][-1])}
#     - Data channel: {describe_bs_state(state['bs_obs_history'][-1],state['num_UEs'])}

#     Depending on the system state, make the following decisions:
#     1. Each UE's PHY actions (transmit/delete/none)
#     2. Each UE's UCM (UCM0/UCM1)
#     3. BS DCM to each UE (DCM0/DCM1/DCM2)
#     """

#     # 替换 UCM 和 DCM 的值
#     ucm_mapping = {'SR': 'UCM0', None: 'UCM1'}
#     dcm_mapping = {'SG': 'DCM0', 'ACK': 'DCM1', None: 'DCM2'}
#     ue_ucm = [ucm_mapping[ucm] for ucm in action['ucm']]
#     ue_dcm = [dcm_mapping[dcm] for dcm in action['dcm']]
#     # 构造输出文本
#     decisions = ", ".join([f"UE{i}: PHY={phy}, UCM={ucm}, DCM={dcm}" for i, (phy, ucm, dcm) in enumerate(zip(action['ue_actions'], ue_ucm, ue_dcm))])
#     output_text = f"""Decisions:
#     {decisions}
#     """
#     return {
#         "instruction": "As a 5G/6G scheduler, make scheduling decisions for UEs and BS based on the system state.",
#         "input": input_text,
#         "output": output_text
#     }
# def create_training_sample(state, action):
#     del state['time_step']
#     input_text = f"""{state} """
#     input_text = input_text.replace("{", "").replace("}", "").replace("'", "").replace(":", "=").strip()
#     output_text = f"""Next time step, {describe_actions(action)}"""
#     return {
#         "instruction": input_text,
#         "input": "Try to make next time step decisions based on the current system state.",
#         "output": output_text
#     }
def create_training_sample(state, action):
    del state['time_step']
    input_text = describe_state(state)
    output_text = f"""Next time step, {describe_actions(action)}"""
    return {
        # "instruction": "Make decisions for BS and " + str(state['num_UEs']) + " UEs based on the current system state.",
        "instruction":"Decide for BS and "+str(state['num_UEs']) + " UEs based on their state history.",
        "input": re.sub(r"'(?!s\b)", "", input_text),
        "output": re.sub(r"'(?!s\b)", "", output_text)
    }

def create_training_sample_singleUE(state, action):
    del state['time_step']
    input_text_list = describe_state_single(state)
    output_text_list = describe_actions_single(action)
    # output_text_list = f"""Next time step, {describe_actions_single(action)}"""

    rr = []
    for i in range(len(input_text_list)):
        r = {
            "instruction": "Decide for UE based on its state history.",
            "input": re.sub(r"'(?!s\b)", "", input_text_list[i]),
            "output": "Next time step, " + re.sub(r"'(?!s\b)", "", output_text_list[i])
        }
        rr.append(r)
    return rr
def create_training_sample_singleBS(action):
    
    ipt_txt,out_txt = describe_actions_single(action,bs=True)
    r = {
        "instruction": "Rule: BS sends SG to a UE with SR and ACK to UE occupying the channel.",
        "input": "Decide for BS based on current state: " + re.sub(r"'(?!s\b)", "", ipt_txt),
        "output": re.sub(r"'(?!s\b)", "", out_txt)
    }
    return r
# 辅助函数
def describe_state_single(state,bs=False):
    ue_obs_his = state['ue_obs_history']
    ue_act_his = state['ue_actions_history']
    ue_msg_his = state['ue_msg_history']
    des=[]
    if not bs:
        for id in range(state['num_UEs']):
            action = [act[id] for act in ue_act_his]
            obs = [obs[id] for obs in ue_obs_his]
            ucm = [msg[id] for msg in ue_msg_his]
            dcm = [msg[id] for msg in state['bs_msg_history']]
            ue_line = f"UE's observations are {obs}, actions are {action}, UCM are {ucm} and DCM are {dcm}."
            des.append(ue_line)
        return des  
    # else:
    #     bs_obs = state['bs_obs_history'][-1]
    #     bs_msg = state['bs_msg_history']
    #     for id in range(state['num_UEs']):
    #         dcm = [msg[id] for msg in bs_msg]
    #         ue_line = f"UE{id}'s DCM are {dcm}."
    #         des.append(ue_line)
    #     if bs_obs[0] == -1:
    #         bs_line = f"BS's observation is {bs_obs}, which means channel is idle in the last time step."
    #     elif bs_obs[0] == -2:
    #         bs_line = f"BS's observation is {bs_obs}, which means channel is collision in the last time step."
    #     else:
    #         bs_line = f"BS's observation is {bs_obs}, which means UE{bs_obs[0]} was performing Transmit in the last time step."
    #     des.append(bs_line)
    #     return "\n".join(des)

def describe_actions_single(actions,bs=False):
    ue_actions = actions['ue_actions']
    ue_ucm = actions['ucm']
    ue_dcm = actions['dcm']
    if not bs:
        des=[]
        for id in range(len(ue_actions)):
            text_ue = f"UE should perform {ue_actions[id]} and send {ue_ucm[id]} to BS"
            des.append(text_ue)
        return des
    else:
        ipt_des,out_des = [],[]
        sr_queue = []
        ack_ue = None
        sg_ue = None
        for id,val in enumerate(ue_ucm):
            if val == 'SR':
                sr_queue.append(f"UE{id}")
        if len(sr_queue) == 0:
            ipt_des.append("No SR request")
        else:
            ipt_des.append(f"SR request from [{','.join(sr_queue)}]")
        
        transmit_id = next((id for id, val in enumerate(ue_actions) if val == 'Transmit'), None)
        if transmit_id is not None:
            channel_des = f"Chanel is occupied by UE{transmit_id}"
        else:
            channel_des = "Channel is idle"
        ipt_des.append(channel_des)
        
        for id,val in enumerate(ue_dcm):
            if val == 'SG':
                sg_ue = id
            elif val == 'ACK':
                ack_ue = id
            if sg_ue is not None and ack_ue is not None:
                break
        if sg_ue is not None:
            out_des.append(f"BS should send SG to UE{sg_ue}")
        else:
            out_des.append("No SG to send")
        if ack_ue is not None:
            out_des.append(f"BS should send ACK to UE{ack_ue}")
        else:
            out_des.append("No ACK to send")
        return ", ".join(ipt_des),", ".join(out_des)

def describe_state(state):
    ue_obs_his = state['ue_obs_history']
    ue_act_his = state['ue_actions_history']
    bs_obs_his = state['bs_obs_history']
    ue_msg_his = state['ue_msg_history']
    bs_msg_his = state['bs_msg_history']
    des=[]
    for id in range(state['num_UEs']):
        ue_name = f"UE{id}"
        action = [act[id] for act in ue_act_his]
        obs = [obs[id] for obs in ue_obs_his]
        ucm = [msg[id] for msg in ue_msg_his]
        dcm = [msg[id] for msg in bs_msg_his]
        ue_line = f"{ue_name}'s observations are {obs}, actions are {action}, UCM are {ucm} and DCM are {dcm}."
        des.append(ue_line)
    bs_obs = bs_obs_his[-1]
    # bs_msg = bs_msg_his[-1]
    #BS观测表示上一时刻哪个UE在发送数据
    if bs_obs[0] == -1:
        bs_line = f"BS's observation is {bs_obs}, which means channel is idle in the last time step."
    elif bs_obs[0] == -2:
        bs_line = f"BS's observation is {bs_obs}, which means channel is collision in the last time step."
    else:
        bs_line = f"BS's observation is {bs_obs}, which means UE{bs_obs[0]} was performing Transmit in the last time step."
    des.append(bs_line)
    return "\n".join(des)


def describe_actions(actions):
    # return ", ".join([f"UE{i}: {act}" for i, act in enumerate(actions)])
    ue_actions = actions['ue_actions']
    ue_ucm = actions['ucm']
    ue_dcm = actions['dcm']
    des=[]
    for id in range(len(ue_actions)):
        # text_ue = f"UE{id} should perform {ue_actions[id]} and send {ue_ucm[id]} to BS"
        text_ue = f"UE{id} should perform {ue_actions[id]}, send {ue_ucm[id]} to BS, and BS should send {ue_dcm[id]} to UE{id}."
        des.append(text_ue)
    # text_bs = f"BS should send {ue_dcm} to the UEs respectively."
    # des.append(text_bs)
    return "\n".join(des)

def expand_ue(input_text, ue_count, target_count):
    """
    将较少数量的 UE 样本扩展为更多的 UE 样本，扩展方式为复制现有的 UE 信息。
    :param input_text: 原始输入文本 (少量 UE 信息)
    :param ue_count: 原始 UE 的数量
    :param target_count: 目标扩展后的 UE 数量
    :return: 扩展后的输入文本
    """
    if ue_count >= target_count:
        return input_text  # 如果已有 UE 数量足够，无需扩展
    
    ue_sections = input_text.split("For UE")  # 根据 "For UE" 分割每个 UE 的段落
    expanded_text = ue_sections[0]  # 保留开头部分（BS部分）
    
    # 计算需要扩展的数量
    required_additional_ue = target_count - ue_count
    
    # 使用现有的 UE 信息进行扩展
    for i in range(ue_count):
        expanded_text += f"For UE{i}, {ue_sections[i + 1].strip()}"
    
    for i in range(required_additional_ue):
        # 选择一个已有的 UE 信息进行复制扩展
        ue_to_duplicate = ue_sections[(i % ue_count) + 1]  # 分割后的第一个是空的，所以 +1
        # 为扩展出来的 UE 重新编号
        expanded_text += f"For UE{ue_count + i}, {ue_to_duplicate.strip()}"
    
    return expanded_text

def random_crop_ue(input_text, ue_count, target_count):
    """
    从原始 UE 输入文本中随机裁剪掉若干 UE,生成目标数量的 UE 样本。
    :param input_text: 原始输入文本 (多 UE 信息组成的长文本)
    :param ue_count: 原始 UE 的数量
    :param target_count: 目标 UE 的数量
    :return: 裁剪后的输入文本
    """
    ue_sections = input_text.split("For UE")  # 根据 "For UE" 分割每个 UE 的段落
    assert ue_count >= target_count, "目标数量不能超过原始 UE 数量"

    # 随机选择 target_count 个 UE
    selected_ues = random.sample(range(ue_count), target_count)
    
    # 重新组合裁剪后的文本
    cropped_text = ue_sections[0]  # 保留开头部分（BS部分）
    for i in sorted(selected_ues):
        cropped_text += "For UE" + ue_sections[i + 1]  # i+1 因为 split 后第一个元素是空
    return cropped_text

def remove_duplicates(data):
    # 创建一个集合来存储已见过的(input, output)对
    seen = set()
    # 创建新列表存储不重复的数据
    unique_data = []
    
    for item in data:
        # 将input和output组成元组作为判断标准
        key = (item["input"], item["output"])
        
        # 如果这个(input, output)对之前没见过，就保留这条数据
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    
    return unique_data


def proc_rawdata():
    file_path = ['./data/raw_datasets4.json']
    # 读取 JSON 文件
    data = []
    for file in file_path:
        with open(file, 'r') as f:
            data.extend(json.load(f))
    
    proc_datasets = []
    for sample in data:
        proc_datasets.append(create_training_sample(sample['state'], sample['action']))
    
    print(f"原始数据条数: {len(proc_datasets)}")
    proc_datasets_unique = remove_duplicates(proc_datasets)
    print(f"剔除重复后的数据条数: {len(proc_datasets_unique)}")

    with open('./data/processed_datasets3.json', 'w') as f:
        json.dump(proc_datasets_unique, f, indent=4)

def proc_rawdata_singleUE():
    file_path =['./data/raw_datasets2.json']
    tol_data = []
    # 读取 JSON 文件
    for file in file_path:
        with open(file, 'r') as f:
            data = json.load(f)
            tol_data.extend(data)
    
    proc_datasets = []
    for sample in tol_data:
        proc_datasets.extend(create_training_sample_singleUE(sample['state'], sample['action']))

    #剔除重复的数据
    print(f"原始数据条数: {len(proc_datasets)}")
    proc_datasets_unique = remove_duplicates(proc_datasets)
    print(f"剔除重复后的数据条数: {len(proc_datasets_unique)}")
    
    with open('./data/processed_UE_single.json', 'w') as f:
        json.dump(proc_datasets_unique, f, indent=4)

def proc_rawdata_singleBS():
    file_path =['./data/raw_datasets2.json','./data/raw_datasets4.json']
    data = []
    for file in file_path:
        with open(file, 'r') as f:
            data.extend(json.load(f))
    
    proc_datasets = []
    for sample in data:
        proc_datasets.append(create_training_sample_singleBS(sample['action']))
    #剔除重复的数据
    print(f"原始数据条数: {len(proc_datasets)}")
    proc_datasets_unique = remove_duplicates(proc_datasets)
    print(f"剔除重复后的数据条数: {len(proc_datasets_unique)}")
    
    with open('./data/processed_BS_single.json', 'w') as f:
        json.dump(proc_datasets_unique, f, indent=4)


if __name__ == "__main__":
    proc_rawdata()
    # proc_rawdata_singleUE()
    # proc_rawdata_singleBS()

    '''
    proc_datasets1 是有time_step的数据
    proc_datasets2 去除了time_step,UE的数量是2,3,4,5,6


    proc_UE_single 只针对UE的数据集,且unique,是从raw_datasets2&4中提取的
    
    proc_datasets3 UE数量是2,5,8 且unique 已经修正UEx is tx的问题
    '''