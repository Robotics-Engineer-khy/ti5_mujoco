import numpy as np
from envs.jvrc.send2robot import Controller
class JVRC:
    def __init__(self, close_jid, pdgains, dt, active, client):
        self.client = client
        self.control_dt = dt
        self.actuators = active
        self.kp = pdgains[0]
        self.kd = pdgains[1]
        assert self.kp.shape==self.kd.shape==(self.client.nu(),)
        self.client.set_pd_gains(self.kp, self.kd)
        self.prev_action = None
        self.prev_torque = None
        self.iteration_count = np.inf#在ppo.py sample()s赋值,在楼梯任务中默认生成跨步高为0.1的台阶
        self.frame_skip = int(self.control_dt/self.client.sim_dt())
        base_position = [0, 0, 0.9]
        base_orientation = [1, 0, 0, 0]
        self.motor_offset = np.array([0, 0, 0.45, 0.7, 0.25, 0, 0, 0, -0.45, -0.7, -0.25, 0])
        # self.motor_offset = np.array([0, 0.45, 0, 0.7, 0.2, 0,
        #                               0, -0.45, 0, -0.7, -0.2, 0])
        # self.motor_offset = np.zeros(12)
        self.init_qpos = base_position + base_orientation + [0]*(self.client.nq()-7)#25=7+18
        motor_qposadr = self.client.get_motor_qposadr()
        for i in range(self.client.nu()):
            self.init_qpos[motor_qposadr[i]]=self.motor_offset[i]
        for chain in close_jid:
            self.init_qpos[chain[1]] =-self.init_qpos[chain[0]]
            self.init_qpos[chain[2]] =self.init_qpos[chain[0]]
        self.init_qvel = [0] * self.client.nv()  # 24=6+18
        self.gear = self.client.get_gear_ratios()  # 减速比
        self.tau_pd = self.client.step_pd(self.motor_offset, np.zeros(self.client.nu()))  # 电机的扭矩
        # self.con = Controller(self.client.nu(), self.gear, np.array(self.motor_offset), self.kp, self.kd)#********************************************self.client.nu()

    def step(self, action):
        filtered_action = np.zeros(len(self.motor_offset))
        for idx, act_id in enumerate(self.actuators):
            filtered_action[act_id] = action[idx]
        # add fixed motor offset
        filtered_action += self.motor_offset
        if self.prev_action is None:
            self.prev_action = filtered_action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.client.get_act_joint_torques())
        self.prev_action = filtered_action
        self.prev_torque = np.asarray(self.client.get_act_joint_torques())
        self.client.set_pd_gains(self.kp, self.kd)
        self.do_simulation(filtered_action, self.frame_skip)
        # print("filtered_action\n", filtered_action)
        # self.con.set_pd_target(filtered_action)  # 时间控制放测试代码外层********************************************

    def do_simulation(self, target, n_frames):
        for _ in range(n_frames):
            self.tau_pd = self.client.step_pd(target, np.zeros(self.client.nu()))# 电机的扭矩
            #希望到达目标时速度为0。循环中由于实时位置速度会变，算出的扭矩也不同
            tau = [(i/j) for i,j in zip(self.tau_pd, self.gear)]#转换成控制信号ctrl
            self.client.set_motor_torque(tau)
            self.client.step()

