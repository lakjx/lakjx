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
        self.phy_act = [None]
        self.dcm_act = [None]
        self.ucm_act = [None]
    
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
        elif self.phy_act[-1] == 'Del' and not self.dcm_act[-1] and not self.ucm_act[-1] and len(self.buffer) > 0:
            self.state_machine = 'Ready'
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
        self.dcm_act = [None]
        
    # def fresh_stat_machine(self,data_new = False):
    #     if len(self.sr_lst) >0:
    #         self.state_machine = 'SR_Choose'
    #     else:
    #         self.state_machine = 'Idle'
    #     if self.dcm_act[-1] :
    #         if self.dcm_act[-1].split('_')[0] == 'SG':
    #             self.state_machine = 'W_for_Data'
        
    #     if data_new:
    #         self.state_machine = 'Data_Received'
    def fresh_stat_machine(self,data_new = False):
        if len(self.sr_lst) >0:
            self.state_machine = 'SR_Choose'
        else:
            self.state_machine = 'Idle'
        if self.dcm_act[-1] :
            if self.dcm_act[-1].split('_')[0] == 'SG':
                self.state_machine = 'W_for_Data'
        
        if data_new:
            self.state_machine = 'Data_Received'  
        
    def execute(self):
        dcm = None
        if self.state_machine == 'Idle':
           pass
        elif self.state_machine == 'SR_Choose':
            choose_ue = np.random.choice(self.sr_lst)
            dcm = 'SG_to_' + str(choose_ue)
            self.sr_lst = []
        elif self.state_machine == 'W_for_Data':
            pass
        elif self.state_machine == 'Data_Received':
            tx_ue_id = int(self.data_received[-1]['ue'])
            dcm = 'ACK_to_' + str(tx_ue_id)
        return dcm
        

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
    np.random.seed(1)
    # 设置参数
    L = 3      # UE数量
    B = 5      # 缓冲区大小
    T = 24     # 模拟步数
    pa = 0.5    # SDU到达概率
    
    bler_list = np.random.uniform(0.6,0.9,L) # 信道误码率
    
    arrival_times = [simulate_poisson_arrivals(pa, T).astype(int).tolist() for i in range(L)]
    print("arrival_times:",arrival_times)
    
    # 初始化UE/BS
    bs = Base_Station(num_ue=L)
    ue_list = []
    for i in range(L):
        ue_list.append(UE(i,B))
        
    # 开始模拟
    for t in range(100):
        
        for i in range(L):
            sdu = []
            for j in arrival_times[i]:
                if j == t:
                    sdu.append({'ue':i,'gen_t':t})
            ue_list[i].add_sdu(sdu)
        
        ue_ucm,ue_phy_act,data_channel= [],[],[]
        for i in range(L):
            ucm,phy_act = ue_list[i].execute()
            ue_ucm.append(ucm)
            ue_phy_act.append(phy_act)
        bs_dcm = bs.execute()
        print("time:",t,"ue_ucm:",ue_ucm,"ue_phy_act:",ue_phy_act,"bs_dcm:",bs_dcm)
        
        if bs_dcm:
            signal = bs_dcm.split('_')[0]
            signal_to_id = int(bs_dcm.split('_')[-1])
            if signal == 'SG':
                for i in range(L):
                    ue_list[i].ucm_act.append(ue_ucm[i])
                    ue_list[i].phy_act.append(ue_phy_act[i])
                    ue_list[i].dcm_act.append('SG' if i == signal_to_id else None) 
            elif signal == 'ACK':
                for i in range(L):
                    ue_list[i].ucm_act.append(ue_ucm[i])
                    ue_list[i].phy_act.append(ue_phy_act[i])
                    ue_list[i].dcm_act.append('ACK' if i == signal_to_id else None)
        else:
            for i in range(L):
                ue_list[i].ucm_act.append(ue_ucm[i])
                ue_list[i].phy_act.append(ue_phy_act[i])
                ue_list[i].dcm_act.append(None)
        
        for i in range(L):
            dat =ue_list[i].buffer_manager()
            data_channel.append(dat) if dat else None
        bs.dcm_act.append(bs_dcm)
        bs.sr_lst = copy.deepcopy([i for i in range(L) if ue_ucm[i] == 'SR'])
        if len(data_channel) ==1:
            data = data_channel[0]
            data['recv_t'] = t+1
            bs.data_received.append(data)
            is_data_new = True
        else:
            is_data_new = False
        
        for i in range(L):
            ue_list[i].fresh_stat_machine()
        bs.fresh_stat_machine(data_new=is_data_new)
        
        #如果所有UE buffer为空，结束模拟
        if all([len(ue.buffer) == 0 for ue in ue_list]):
            break
    
    print("dropped packets number:",sum([len(ue.packets_dropped) for ue in ue_list]))
    print("received packets number:",len(bs.data_received))
      
    print("goodputs",len(bs.data_received)/t) 
    # print("received packets:",bs.data_received)
    Delay = [data['recv_t'] - data['gen_t'] for data in bs.data_received]
    print("average delay:",np.mean(Delay))