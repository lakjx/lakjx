import torch
import numpy as np
import argparse
import itertools
import gym
import copy
import re
from gym import spaces
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
class DotDic(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __deepcopy__(self, memo=None):
		return DotDic(copy.deepcopy(dict(self), memo=memo))
class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        #random_array = prng.np_random.rand(self.num_discrete_space)
        random_array = np.random.rand(self.num_discrete_space)

        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)

class MacProtocolEnv():
    def __init__(self, args, discrete_action=True):
        self.args = args
        self.is_training = True
        self.rho = self.args.rho
        self.UE_num = self.args.UE_num
        self.p_SDU_arrival = self.args.p_SDU_arrival
        self.tbl_error_rate = self.args.tbl_error_rate
        self.TTLs = self.args.TTLs  # Max. duration of episode
        self.recent_k = self.args.recent_k
        self.collision_count = 0
        self.gen_data_count = 0
        self.UE_act_space = DotDic({
            'Do Nothing': 0,
            'Transmit': 1,
            'Delete': 2
        })
        # UE_obs \in [0,|B|]
        self.UE_obs_space = spaces.Discrete(self.args.UE_txbuff_len + 1)
        # BS_obs \in [0,|U|+1]
        self.BS_obs_space = spaces.Discrete(self.UE_num + 2)

        self.BS_msg_space = DotDic({
            'Null': 0,
            'SG': 1,
            'ACK': 2
        })
        self.BS_msg_total_space = list(itertools.product(range(len(self.BS_msg_space)), repeat=self.UE_num))
        self.UE_msg_space = DotDic({
            'Null': 0,
            'SR': 1
        })

        self.agents = ['UE_' + str(i) for i in range(self.UE_num)] + ['BS']
        self.num_agents = len(self.agents)
        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            #physical action
            u_action_space = spaces.Discrete(len(self.UE_act_space))
            if agent != 'BS':
                total_action_space.append(u_action_space)
            # elif agent == 'BS' and self.args.need_comm == False:
            #     total_action_space.append(spaces.Discrete(2))
            #communication action
            if self.args.need_comm:
                if agent != 'BS':
                    c_action_space = spaces.Discrete(len(self.UE_msg_space))
                    total_action_space.append(c_action_space)
                else:
                    c_action_space = spaces.Discrete(len(self.BS_msg_space)**self.UE_num)
                    total_action_space.append(c_action_space)
            
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    action_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    raise NotImplementedError
                self.action_space.append(action_space)
            elif len(total_action_space) == 1:
                self.action_space.append(total_action_space[0])
            else:
                self.action_space.append([])
            # observation space
            if agent != 'BS':
                obs_dim = 4*(self.recent_k+1) if self.args.need_comm else 2*(self.recent_k+1)
                self.observation_space.append(spaces.Discrete(obs_dim))
            else:
                obs_dim = self.recent_k+1 + self.UE_num*2*(self.recent_k+1) if self.args.need_comm else self.recent_k+1
                self.observation_space.append(spaces.Discrete(obs_dim))
            share_obs_dim += obs_dim
        self.share_observation_space = [spaces.Discrete(share_obs_dim)] * self.num_agents

        self.reset()

    def reset(self):
        self.step_count = 0
        self.collision_count = 0
        self.gen_data_count = 0

        self.UEs = [UE(i,self.args) for i in range(self.UE_num)]

        # # self.UE_SDU_Generate()
        self.UE_obs = np.zeros((self.UE_num,), dtype=np.int32)
        self.UE_actions = np.zeros((self.UE_num,), dtype=np.int32)
        self.BS_obs = np.zeros((1,), dtype=np.int32)
        self.BS_msg = np.zeros((self.UE_num,), dtype=np.int32)
        self.UE_msg = np.zeros((self.UE_num,), dtype=np.int32)

        self.trajact_UE_obs = [copy.deepcopy(self.UE_obs) for _ in range(self.recent_k + 1)]
        self.trajact_UE_actions = [copy.deepcopy(self.UE_actions) for _ in range(self.recent_k + 1)]
        self.trajact_BS_obs = [copy.deepcopy(self.BS_obs) for _ in range(self.recent_k + 1)]
        self.trajact_BS_msg = [copy.deepcopy(self.BS_msg) for _ in range(self.recent_k + 1)]
        self.trajact_UE_msg = [copy.deepcopy(self.UE_msg) for _ in range(self.recent_k + 1)]


        
        self.sdus_received = []
        self.data_channel = []
        self.rewards = 0
        self.done = False
        # record observations for each agent
        obs_n = []
        for agent in self.agents:
            if agent == 'BS':
                obs_n.append(self.get_bs_internal_stat())
            else:
                tar_ue_idx = int(agent.split('_')[1])
                obs_n.append(self.get_ue_internal_stat(tar_ue_idx))
        return obs_n

    def step(self, action_n,UCM=None,DCM=None):
        # UE_actions = [act[0] for (i, act) in enumerate(action_n) if self.agents[i] != 'BS']
        # UCM = [act[1] for (i, act) in enumerate(action_n) if self.agents[i] != 'BS'] if self.args.need_comm else None
        # DCM = self.BS_msg_total_space[action_n[-1][0]] if self.args.need_comm else None
        UE_actions = action_n
        #测试状态下，打印每个UE的buffer状态
        if not self.is_training:
            for UE in self.UEs:
                print(UE.name,UE.buff)         

        if isinstance(UE_actions, list):
            UE_actions = np.array(UE_actions)
        elif isinstance(UE_actions, torch.Tensor):
            UE_actions = UE_actions.cpu().numpy()
        
        #随机生成UE的SDU
        new_data_list = self.UE_SDU_Generate()
        print('new_data_list:',new_data_list) if not self.is_training else None

        self.UE_actions = UE_actions
        self.UE_Signaling_policy(np.array(UCM)) if UCM is not None else self.UE_Signaling_policy()
        self.BS_Signaling_policy(np.array(DCM)) if DCM is not None else self.BS_Signaling_policy()
        error_del = 0
        self.data_channel = []
        for UE in self.UEs:
            if len(UE.buff) > 0:
                if UE_actions[UE.name_id] == self.UE_act_space.Transmit and UE.buff[0] != new_data_list[UE.name_id]:
                    data = UE.transmit_SDU()
                    self.data_channel.append(data)
                
                elif UE_actions[UE.name_id] == self.UE_act_space.Delete and UE.buff[0] != new_data_list[UE.name_id]:
                    del_data = UE.delete_SDU()
                    if del_data not in self.sdus_received:
                        error_del = error_del + 1
                    else:
                        error_del = error_del - 1
            else:
                pass
        self.check_channel(error_del)                 
    
        self.trajact_UE_obs.append(copy.deepcopy(np.array([UE.get_obs() for UE in self.UEs])))
        self.trajact_UE_actions.append(copy.deepcopy(self.UE_actions))
        self.trajact_BS_obs.append(copy.deepcopy(self.BS_obs) if isinstance(self.BS_obs, np.ndarray) else np.array([self.BS_obs]))
        self.trajact_BS_msg.append(copy.deepcopy(self.BS_msg))
        self.trajact_UE_msg.append(copy.deepcopy(self.UE_msg))
        if len(self.trajact_UE_obs) > self.recent_k+1:
            self.trajact_UE_obs.pop(0)
            self.trajact_UE_actions.pop(0)
            self.trajact_BS_obs.pop(0)
            self.trajact_BS_msg.pop(0)
            self.trajact_UE_msg.pop(0)

        self.done = self.step_count >= self.TTLs
        self.step_count += 1
        if not self.is_training:
            print('step:',self.step_count,'UE_act:',UE_actions,'datachannel:',self.data_channel,'rewards:',self.rewards)
            print('BS_recieved:',self.sdus_received)
        
        obs_n, reward_n, done_n, info_n = [], [], [], []
        for agent in self.agents:
            if agent == 'BS':
                obs_n.append(self.get_bs_internal_stat())
            else:
                tar_ue_idx = int(agent.split('_')[1])
                obs_n.append(self.get_ue_internal_stat(tar_ue_idx))
            reward_n.append(self.get_rwd())
            done_n.append(self.done)
            info_n.append({})

        return obs_n, reward_n, done_n, info_n

            
    def get_ue_internal_stat(self,tar_ue_idx):
        # x =(o,a,n,m) o: UE_obs, a: UE_actions, n: UE_msg, m: BS_msg
        # o = np.transpose([UE.get_obs() for UE in self.UEs])
        # o = UE.get_obs()
        # a = self.UE_actions[UE.name_id]
        # n = self.UE_msg[UE.name_id]
        # m = self.BS_msg[UE.name_id]
        # return np.array([o,a,n,m])
        if len(self.trajact_UE_obs) < self.recent_k + 1:
            # 填充缺失的轨迹数据，使用最近的观测值
            gap = self.recent_k + 1 - len(self.trajact_UE_obs)
            o = [self.trajact_UE_obs[0][tar_ue_idx]] * gap + [self.trajact_UE_obs[i][tar_ue_idx] for i in range(len(self.trajact_UE_obs))]
            a = [self.trajact_UE_actions[0][tar_ue_idx]] * gap + [self.trajact_UE_actions[i][tar_ue_idx] for i in range(len(self.trajact_UE_actions))]
            n = [self.trajact_UE_msg[0][tar_ue_idx]] * gap + [self.trajact_UE_msg[i][tar_ue_idx] for i in range(len(self.trajact_UE_msg))]
            m = [self.trajact_BS_msg[0][tar_ue_idx]] * gap + [self.trajact_BS_msg[i][tar_ue_idx] for i in range(len(self.trajact_BS_msg))]
        else:
            o = [self.trajact_UE_obs[i][tar_ue_idx] for i in range(self.recent_k + 1)]
            a = [self.trajact_UE_actions[i][tar_ue_idx] for i in range(self.recent_k + 1)]
            n = [self.trajact_UE_msg[i][tar_ue_idx] for i in range(self.recent_k + 1)]
            m = [self.trajact_BS_msg[i][tar_ue_idx] for i in range(self.recent_k + 1)]

        if self.args.need_comm:
            return np.concatenate((o, a, n, m), axis=0).flatten()
        else:
            return np.concatenate((o, a), axis=0).flatten()
            
            

    def get_bs_internal_stat(self):
        #x=(o_b,n_all,m_all)
        #检查BS_obs的数据类型
        assert self.BS_obs_space.contains(self.BS_obs[0] if isinstance(self.BS_obs,np.ndarray) else self.BS_obs)
        
        if len(self.trajact_BS_obs) < self.recent_k + 1:
            # 填充缺失的轨迹数据，使用最近的观测值
            gap = self.recent_k + 1 - len(self.trajact_BS_obs)
            self.trajact_BS_obs = [self.trajact_BS_obs[0]] * gap + self.trajact_BS_obs
            self.trajact_BS_msg = [self.trajact_BS_msg[0]] * gap + self.trajact_BS_msg
            self.trajact_UE_msg = [self.trajact_UE_msg[0]] * gap + self.trajact_UE_msg

        if self.args.need_comm:
            return np.concatenate((self.trajact_BS_obs, self.trajact_UE_msg, self.trajact_BS_msg), axis=1).flatten()
        else:
            return np.array(self.trajact_BS_obs).flatten()
    
    def get_rwd(self):
        return self.rewards
    
    def check_channel(self,error_del):
        # rho = 3
        #查看data通道上是否有冲突  e.g. data_channel = ['UE0_1', 'UE1_0', 'UE2_1']
        if len(self.data_channel) == 1: # 正常数传
            data = self.data_channel[0]
            # self.BS_obs = int(data.split('_')[0][2:])+1
            self.BS_obs = int(data.split('_')[0][2:])
            if np.random.rand() > self.tbl_error_rate: #正确接收
                if data not in self.sdus_received:
                    self.sdus_received.append(data)
                    self.rewards = 2*self.rho
                else:
                    self.rewards = -self.rho
        elif self.data_channel == []: # 空闲
            self.BS_obs = 0
            self.rewards = -self.rho
        else:
            self.collision_count += 1
            self.BS_obs = self.UE_num + 1
            self.rewards = -1

        self.rewards = self.rewards - error_del*self.rho
        #判断BS_obs是否合法
        assert self.BS_obs_space.contains(self.BS_obs)
                                   
    def UE_SDU_Generate(self):
        gen_data_list = []
        for UE in self.UEs:
            if np.random.rand() < self.p_SDU_arrival:
                cur_gen_data = UE.generate_SDU()
                self.gen_data_count += 1
            else:
                cur_gen_data = None
            gen_data_list.append(cur_gen_data)
        return gen_data_list

    def BS_Signaling_policy(self,DCM=None):
        # BS can send one control message to each UE
        if DCM is None:
            DCM = np.random.randint(0, len(self.BS_msg_space), self.UE_num)
            self.BS_msg = DCM
        else:
            self.BS_msg = DCM
    def UE_Signaling_policy(self,UCM=None):
        # each UE can send one control message to BS
        if UCM is None:
            UCM = np.random.randint(0, len(self.UE_msg_space), self.UE_num)
            self.UE_msg = UCM
        else:
            self.UE_msg = UCM
    
    def get_Goodput(self):
        if self.step_count == 0:
            raise ValueError('step_count is 0!')
        return len(self.sdus_received)/self.step_count
    def get_collision_rate(self):
        return self.collision_count / self.step_count
    def get_buffer_occupancy(self):
        return [UE.get_obs()/UE.buff_size for UE in self.UEs]
    def get_packet_arrival_rate(self):
        return len(self.sdus_received)/self.gen_data_count
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

class UE():
    def __init__(self, name_id , args):
        self.name_id = name_id
        self.name = 'UE' + str(name_id)
        self.buff_size = args.UE_txbuff_len
        self.buff = []
        self.datacount = 0
        self.SG = False
        self.ACK = False

    def generate_SDU(self):
        gen_data = None
        if len(self.buff) < self.buff_size:
            gen_data = self.name + '_' + str(self.datacount)
            self.buff.append(gen_data)
            self.datacount += 1
        return gen_data
    
    def delete_SDU(self):
        if len(self.buff) > 0:
            del_data = self.buff.pop(0)
            return del_data
        else:
            print('Delete_SDU error!'+ self.name + ' buffer is empty!')
            return None      
    
    def transmit_SDU(self):
        if len(self.buff) > 0:
            return self.buff[0]
        else:
            print('Transmit_SDU error!'+ self.name + ' buffer is empty!')
            return None
    
    def get_obs(self):
        return len(self.buff)

def load_model(base_model_path,adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        device_map = "auto",
        torch_dtype = torch.half,
        use_cache=False
    )
    if type(adapter_path) == list:
        adapter_path_merge = adapter_path[0]
        adapter_path_BS = adapter_path[1]
    else:
        adapter_path_merge = adapter_path
        adapter_path_BS = None
    model = PeftModel.from_pretrained(base_model, model_id=adapter_path_merge, torch_dtype=torch.half,adapter_name='adapter_merge')
    if adapter_path_BS is not None:
        model.load_adapter(adapter_path_BS,adapter_name='adapter_BS')
    model.eval()
    return model,tokenizer

def generate_response(model,tokenizer,instruction,input_text,max_length=1024):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if type(input_text) == str:
        model.set_adapter('adapter_merge')
        prompt = f"Human:{instruction}\n{input_text}\n\n Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
                # do_sample=False
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        return response
    elif type(input_text) == list:
        responses = []
        length = len(input_text)
        for i,text in enumerate(input_text):
            if i != length-1:
                model.set_adapter('adapter_merge')
                prompt = f"Human:{instruction[0]}\n{text}\n\n Assistant:"  
            else:
                model.set_adapter('adapter_BS')
                prompt = f"Human:{instruction[1]}\n{text}\n\n Assistant:"
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                    # do_sample=False
                )
            # 解码输出
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            responses.append(response)
        return responses

def test_model(model, tokenizer, test_cases):
    """
    测试模型
    """
    for i, case in enumerate(test_cases):
        instruction = case["instruction"]
        input_text = case.get("input", "")
        print(f"\nTest case {i+1}:")
        print(f"Instruction: {instruction}")
        print(f"Input: {input_text}")
        
        response = generate_response(model, tokenizer, instruction, input_text)
        print(f"Response: {response}")
        print("-" * 50)

def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', type=int, default=3)
    parser.add_argument('--recent_k', type=int, default=1)
    parser.add_argument('--UE_num', type=int, default=8)
    parser.add_argument('--UE_txbuff_len', type=int, default=20)
    # parser.add_argument('--UE_max_generate_SDUs', type=int, default=2)
    parser.add_argument('--p_SDU_arrival', type=float, default=0.5)
    parser.add_argument('--tbl_error_rate', type=float, default=1e-3)
    parser.add_argument('--TTLs', type=int, default=24)
    parser.add_argument('--UCM', type=int, default=None)
    parser.add_argument('--DCM', type=int, default=None)
    parser.add_argument('--need_comm', type=bool, default=True)
    args = parser.parse_args()
    return args
def proc_llama_output(out,ue_num):
    if type(out) == str:
        out = out.split('\n')
        UE_actions,UCM_msg,DCM_msg = [],[],[]
        if len(out) != ue_num:
            raise ValueError('The number of UE actions is not equal to the number of UEs!')
        for i in range(len(out)):
            line = out[i]
            act = line.split('perform ')[-1].split(',')[0]
            act = {'Transmit': 1, 'Delete': 2}.get(act, 0)
            UE_actions.append(act)

            ucm = line.split('send ')[1].split(' to BS')[0]
            ucm = {'SR': 1}.get(ucm, 0)
            UCM_msg.append(ucm)

            dcm = line.split('BS should send ')[1].split(' to')[0].replace(' ','')
            dcm = {'ACK':2,'SG':1}.get(dcm,0)
            DCM_msg.append(dcm)
        return UE_actions,UCM_msg,DCM_msg
    elif type(out) == list:
        bs_out = out.pop()
        UE_actions,UCM_msg,DCM_msg = [],[],[]
        for line_ue in out:
            act = line_ue.split('perform ')[-1].split(' and')[0]
            act = {'Transmit': 1, 'Delete': 2}.get(act, 0)
            UE_actions.append(act)

            ucm = line_ue.split('send ')[1].split(' to BS')[0]
            ucm = {'SR': 1}.get(ucm, 0)
            UCM_msg.append(ucm)
        line_bs = bs_out
        DCM_msg = [0]*ue_num
        if 'send ACK to' in line_bs:
            sgn = line_bs.split('send ACK to UE')[1].split(', ')[0]
            if len(sgn) == 1:
                DCM_msg[int(sgn)] = 2
        if 'send SG to' in line_bs:
            sgn = line_bs.split('send SG to UE')[1].split(', ')[0]
            if len(sgn) == 1:
                DCM_msg[int(sgn)] = 1 
            
        return UE_actions,UCM_msg,DCM_msg

def instruct_UEAndBS(instructions):
    sen1 = f"Decide for BS and " + str(instructions['num_UEs']) + " UEs based on their 2-step state history."
    sen2=[]
    for id in range(instructions['num_UEs']):
        ue_name = f"UE{id}"
        action = [act[id] for act in instructions["ue_actions_history"]]
        obs = [obs[id] for obs in instructions["ue_obs_history"]]
        ucm = [msg[id] for msg in instructions["ue_msg_history"]]
        dcm = [msg[id] for msg in instructions["bs_msg_history"]]
        ue_line = f"{ue_name}'s observations are {obs}, actions are {action}, UCM are {ucm} and DCM are {dcm}."
        sen2.append(ue_line)
    bs_obs = instructions["bs_obs_history"][-1]
    bs_line = f"BS's observation is {bs_obs}, which means UE{bs_obs[0]} was performing Transmit in the last time step."
    sen2.append(bs_line)
    sen2 = re.sub(r"'(?!s\b)", "", '\n'.join(sen2))
    return sen1,sen2
def instruct_singleUEBS(instructions):
    sen1,sen2=[],[]
    sen1.append(f"Decide for UE based on its 2-step state history.")
    for id in range(instructions['num_UEs']):
        ue_name = f"UE{id}"
        action = [act[id] for act in instructions["ue_actions_history"]]
        obs = [obs[id] for obs in instructions["ue_obs_history"]]
        ucm = [msg[id] for msg in instructions["ue_msg_history"]]
        dcm = [msg[id] for msg in instructions["bs_msg_history"]]
        ue_line = f"UE's observations are {obs}, actions are {action}, UCM are {ucm} and DCM are {dcm}."
        ue_line = re.sub(r"'(?!s\b)", "", ue_line)
        sen2.append(ue_line)
    
    sen1.append(f"Rule: BS sends SG to a UE with SR and ACK to UE occupying the channel.")
    sr_queue = []
    ipt_des = []
    for id,val in enumerate(instructions["ue_msg_history"][-1]):
        if val == 'SR':
            sr_queue.append(f"UE{id}")
    if len(sr_queue) == 0:
        ipt_des.append("Decide for BS based on current state: "+"No SR request")
    else:
        ipt_des.append("Decide for BS based on current state:"+f"SR request from [{','.join(sr_queue)}]")
    transmit_id = next((id for id, val in enumerate(instructions["ue_actions_history"][-1]) if val == 'Transmit'), None)
    if transmit_id is not None:
        channel_des = f"Chanel is occupied by UE{transmit_id}"
    else:
        channel_des = "Channel is idle"
    ipt_des.append(channel_des)
    ipt_des = ', '.join(ipt_des)

    sen2.append(ipt_des)

    return sen1,sen2

def test_env(env,p_model,tokenizer):

    env.is_training = False
    print("init obs:",env.reset())
    # env.step(np.random.randint(0, 3, env.UE_num))
    t=1
    act_map = {0:'None',1:'Transmit',2:'Delete'}
    dcm_map = {0:'None',1:'SG',2:'ACK'}
    ucm_map = {0:'None',1:'SR'}
    
    while env.done == False:
        instructions = {"num_UEs":env.UE_num}
        instructions["ue_obs_history"] = [list(a) for a in env.trajact_UE_obs]
        instructions["ue_actions_history"] = [[act_map[x] for x in sublist.tolist()] for sublist in env.trajact_UE_actions]
        instructions["bs_obs_history"] = [list(a) for a in env.trajact_BS_obs]
        instructions["bs_msg_history"] = [[dcm_map[x] for x in sublist.tolist()] for sublist in env.trajact_BS_msg]
        instructions["ue_msg_history"] = [[ucm_map[x] for x in sublist.tolist()] for sublist in env.trajact_UE_msg]
        
        # sen1,sen2 = instruct_UEAndBS(instructions)
        sen1,sen2 = instruct_singleUEBS(instructions)
        
        out = generate_response(p_model,tokenizer,sen1,sen2)
        print(out)
        #获取动作脚本
        UE_actions,ue_ucm,ue_dcm = proc_llama_output(out,env.UE_num)
        
        # UE_actions = np.random.randint(0, 3, env.UE_num)
        o,r,_,_ =env.step(UE_actions,ue_ucm,ue_dcm)

        print("observation:{}".format(o))
        print("reward:{}".format(r))
        t+=1
    print('Goodput:',env.get_Goodput())
    print('collision rate:',env.get_collision_rate())
    print('buffer occupancy:',np.average(env.get_buffer_occupancy()))
    print('packet arrival rate:',env.get_packet_arrival_rate())


if __name__ == "__main__":
    # 模型路径
    BASE_MODEL_PATH = "./data/pretrained_models/shakechen/Llama-2-7b-hf"
    ADAPTER_PATH_MERGE = "./data/finetuned_models/llama-2-7b-hf-merge/checkpoint-3000"
    # ADAPTER_PATH_MERGE = "./data/finetuned_models/llama7b-global/checkpoint-3300"
    ADAPTER_PATH_BS = "./data/finetuned_models/llama7b-singleBS/checkpoint-200"
    
    # 加载模型
    model, tokenizer = load_model(BASE_MODEL_PATH, [ADAPTER_PATH_MERGE,ADAPTER_PATH_BS])
    
    # 测试用例
    test_cases = [
        {
        "instruction": "num_UEs= 2, ue_obs_history= [[1, 1], [2, 2]], ue_actions_history= [[None, None], [None, None]], bs_obs_history= [[0], [0]], bs_msg_history= [[None, None], [None, None]], ue_msg_history= [[None, None], [None, None]]",
        "input": "Try to make next time step decisions based on the current system state.",
        "output": "Next time step, UE0 should perform None and send SR to BS\nUE1 should perform None and send SR to BS\nBS should send [None, 'SG'] to the UEs respectively."
        }
    ]
    
    # 运行测试
    # test_model(model, tokenizer, test_cases)

    args = arg_config()
    env = MacProtocolEnv(args)
    test_env(env=env,p_model=model,tokenizer=tokenizer)