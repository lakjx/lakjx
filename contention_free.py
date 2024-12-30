import numpy as np
import copy
from collections import deque
import json
class UE:
    def __init__(self, name_id, buffer_size):
        self.name_id = name_id
        self.name = 'UE' + str(name_id)
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.gen_sdu_conut = 0
        
        self.state_machine = 'Idle'

        # 相关变量
        self.packets_dropped = []
        self.phy_act,self.dcm_act,self.ucm_act = [None],[None],[None]
    
    def fresh_stat_machine(self):
        if len(self.buffer) == 0 and not self.phy_act[-1] and not self.dcm_act[-1] and not self.ucm_act[-1]:
            self.state_machine = 'Idle'
        elif len(self.buffer) > 0 and not self.phy_act[-1] and not self.dcm_act[-1] and not self.ucm_act[-1]:
            self.state_machine = 'Ready'
        elif self.ucm_act[-1] == 'SR' and not self.phy_act[-1] and not self.dcm_act[-1]:
            self.state_machine = 'W_for_SG'
        elif self.dcm_act[-1] == 'SG' and not self.phy_act[-1] and not self.ucm_act[-1]:
            self.state_machine = 'SG_Get'
        elif self.phy_act[-1] == 'Tx' and not self.dcm_act[-1] and not self.ucm_act[-1]:
            self.state_machine = 'W_for_ACK'
        elif self.dcm_act[-1] == 'ACK' and not self.phy_act[-1] and not self.ucm_act[-1]:
            self.state_machine = 'ACK_Get'
        elif self.phy_act[-1] == 'Del' and self.ucm_act[-1] == 'SR' and not self.dcm_act[-1]  and len(self.buffer) > 0:
            self.state_machine = 'W_for_SG'
        elif self.phy_act[-1] == 'Del' and not self.dcm_act[-1] and not self.ucm_act[-1] and len(self.buffer) == 0:
            self.state_machine = 'Idle'        
        
    def add_sdu(self, new_sdu_list):
        for new_sdu in new_sdu_list:
            self.gen_sdu_conut += 1
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(new_sdu)
            else:
                self.packets_dropped.append(new_sdu)
        self.fresh_stat_machine()

    def execute(self):
        ucm,phy_act = None,None
        if self.state_machine == 'Idle':
            pass
        elif self.state_machine == 'Ready':
            ucm = 'SR'
        elif self.state_machine == 'W_for_SG':
            pass
        elif self.state_machine == 'SG_Get':
            phy_act = 'Tx'
        elif self.state_machine == 'W_for_ACK':
            pass
        elif self.state_machine == 'ACK_Get':
            phy_act = 'Del'
            ucm = 'SR' if len(self.buffer) > 1 else None         
        return ucm,phy_act
    
    def buffer_manager(self):
        dat = None
        if self.phy_act[-1] == 'Tx':
            dat = self.buffer[0]
        elif self.phy_act[-1] == 'Del':
            self.buffer.popleft()
        return dat
    
class Base_Station:
    def __init__(self,num_ue):                
        self.state_machine = ['Idle' for _ in range(num_ue)]
        
        self.data_received = []
        self.sr_lst = []
        self.dcm_act = []
        self.chosen_ue = None 

    def fresh_stat_machine(self,data_new = False):
        self.chosen_ue = None  
        if len(self.sr_lst) >0:
            self.chosen_ue = np.random.choice(self.sr_lst)
        
        for i in range(len(self.state_machine)):
            if i == self.chosen_ue:
                self.state_machine[i] = 'SR_Choose'
            elif self.dcm_act[-1][i] and self.dcm_act[-1][i].split('_')[0] == 'SG':
                self.state_machine[i] = 'W_for_Data'
            elif data_new and i == self.data_received[-1]['ue']:
                self.state_machine[i] = 'Data_Received'
            else:
                self.state_machine[i] = 'Idle'      
        
    def execute(self):
        dcm_tmp = [None for _ in range(len(self.state_machine))]
        for i in range(len(self.state_machine)):
            if self.state_machine[i] == 'Idle':
                pass
            elif self.state_machine[i] == 'SR_Choose':
                dcm_tmp[i] = 'SG_to_' + str(self.chosen_ue)
            elif self.state_machine[i] == 'W_for_Data':
                pass
            elif self.state_machine[i] == 'Data_Received':
                tx_ue_id = int(self.data_received[-1]['ue'])
                dcm_tmp[i] = 'ACK_to_' + str(tx_ue_id)
        return dcm_tmp
        

def simulate_poisson_arrivals(rate, duration):
  """
  Args:
    rate: 泊松过程的到达率lambda,表示单位时间内的平均到达次数。
    duration: 模拟的总时长。
  Returns:
    一个 NumPy 数组，包含所有事件的到达时间。
  """
  num_events = np.random.poisson(rate * duration)
  interarrival_times = np.random.exponential(scale=1/rate, size=num_events)
  arrival_times = np.cumsum(interarrival_times)
  arrival_times = arrival_times[arrival_times <= duration]

  return arrival_times

if __name__ == "__main__":
    #固定所有随机数
    np.random.seed(11)
    # 设置参数
    L = 2      # UE数量
    B = 20      # 缓冲区大小
    T = 24     # 模拟步数
    pa = 0.3    # SDU到达概率
    
    bler_list = np.round(np.random.uniform(0.9, 0.99, L), 3)
    
    arrival_times = [simulate_poisson_arrivals(pa, T).astype(int).tolist() for _ in range(L)]
    
    # 初始化UE/BS
    bs = Base_Station(num_ue=L)
    ue_list = []
    for i in range(L):
        ue_list.append(UE(i,B))
        
    # 开始模拟
    for t in range(24):
        #每个用户产生SDU,根据泊松到达
        for i in range(L):
            sdu = []
            for j in arrival_times[i]:
                if j == t:
                    sdu.append({'ue':i,'gen_t':t})
            ue_list[i].add_sdu(sdu)
        
        # UE/BS执行,释放出信令及动作
        ue_ucm,ue_phy_act,DATACHANNEL= [],[],[]
        for i in range(L):
            ucm,phy_act = ue_list[i].execute()
            ue_ucm.append(ucm)
            ue_phy_act.append(phy_act)
        bs_dcm = bs.execute()
        print("time:",t,"ue_ucm:",ue_ucm,"ue_phy_act:",ue_phy_act,"bs_dcm:",bs_dcm)
        
        #更新UE的物理动作、上/下行消息动作
        for i in range(L):
            if bs_dcm[i]:
                signal = bs_dcm[i].split('_')[0]
                signal_to_id = int(bs_dcm[i].split('_')[-1])
                if signal == 'SG':
                    ue_list[i].ucm_act.append(ue_ucm[i])
                    ue_list[i].phy_act.append(ue_phy_act[i])
                    ue_list[i].dcm_act.append('SG' if i == signal_to_id else None) 
                elif signal == 'ACK':
                    ue_list[i].ucm_act.append(ue_ucm[i])
                    ue_list[i].phy_act.append(ue_phy_act[i])
                    ue_list[i].dcm_act.append('ACK' if i == signal_to_id else None)
            else:
                ue_list[i].ucm_act.append(ue_ucm[i])
                ue_list[i].phy_act.append(ue_phy_act[i])
                ue_list[i].dcm_act.append(None)
        
        #更新BS的下行消息动作、SR请求列表、数据接收
        bs.dcm_act.append(bs_dcm)
        bs.sr_lst = copy.deepcopy([i for i in range(L) if ue_ucm[i] == 'SR'])
        for i in range(L):
            dat =ue_list[i].buffer_manager()
            DATACHANNEL.append(dat) if dat else None
        is_data_new = False
        if len(DATACHANNEL) == 1:
            data = DATACHANNEL.pop()            
            if bler_list[data['ue']] > np.random.uniform():
                data['recv_t'] = t+1
                bs.data_received.append(data)
                is_data_new = True
        
        #更新UE/BS状态机
        for i in range(L):
            ue_list[i].fresh_stat_machine()
        bs.fresh_stat_machine(data_new=is_data_new)
        
        #如果所有UE buffer为空，结束模拟
        if all([ue.gen_sdu_conut == len(arrival_times[ue.name_id]) for ue in ue_list]):
            break
    
    #输出模拟结果
    print("Arrival_times:",arrival_times)
    print("BLER:",bler_list)
    packt_total = sum([ue.gen_sdu_conut for ue in ue_list])
    packt_drop = sum([len(ue.packets_dropped) for ue in ue_list])
    packt_recv = len(bs.data_received)
    print(f"Dropped: {packt_drop}, Received: {packt_recv}, Total: {packt_total}")
    print(f"Arriv Ratio: {packt_recv / packt_total :.3f}")
    
    print(f"Goodputs:{len(bs.data_received)/t:.3f}") 
    Delay = {}
    for data in bs.data_received:
        if data['ue'] not in Delay:
            Delay[data['ue']] = []
        Delay[data['ue']].append(data['recv_t'] - data['gen_t'])
    for ue in Delay:
        print(f"UE{ue}'s Avg delay:{np.mean(Delay[ue]):.3f}")
    print(f"Avg delay:{np.mean([np.mean(Delay[ue]) for ue in Delay]):.3f}")        
