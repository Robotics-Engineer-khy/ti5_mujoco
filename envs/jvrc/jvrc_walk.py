import os
from logging import info
from envs.jvrc.send2imu import IMU
import numpy as np
import transforms3d as tf3
import collections
from tasks import walking_task
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.jvrc import robot
from .gen_xml import builder

class JvrcWalkEnv(mujoco_env.MujocoEnv):
    def __init__(self):
        sim_dt = 0.001#pd控制周期,mujoco步进器周期
        control_dt = 0.02#决策/render周期,必须是sim_dt的整数倍；在一个self.control_dt内只执行一次policy，即输出的电机目标位置不变。
        path_to_xml_out = '/tmp/mjcf-export/jvrc_walk/Tihu.xml'
        if not os.path.exists(path_to_xml_out):
            builder(path_to_xml_out)
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)
        pdgains = np.zeros((12, 2))
        pdgains.T[0] = np.array([56, 56, 64, 96, 24, 8,
                                 56, 56, 64, 96, 24, 8])
        pdgains.T[1] = 0.1 * pdgains.T[0]
        # pdgains.T[0] = np.array([32, 25, 43, 57, 11, 7,
        #                          31, 25, 42, 57, 11, 7])
        # pdgains.T[1] = 0.1 * pdgains.T[0]
        self.actuators = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.interface = robot_interface.RobotInterface(self.model, self.data, 'R_ANKLE_R_S', 'L_ANKLE_R_S')
        self.close_jid = [[self.interface.get_jnt_qposadr_by_name('L_ANKLE_P')[0], self.interface.get_jnt_qposadr_by_name('L_ANKLE_P2')[0],self.interface.get_jnt_qposadr_by_name('L_ANKLE_P4')[0]],
                          [self.interface.get_jnt_qposadr_by_name('R_ANKLE_P')[0], self.interface.get_jnt_qposadr_by_name('R_ANKLE_P2')[0],self.interface.get_jnt_qposadr_by_name('R_ANKLE_P4')[0]],
                          [self.interface.get_jnt_qposadr_by_name('L_KNEE')[0], self.interface.get_jnt_qposadr_by_name('L_KNEE_P1')[0],self.interface.get_jnt_qposadr_by_name('L_KNEE_P')[0]],
                          [self.interface.get_jnt_qposadr_by_name('R_KNEE')[0], self.interface.get_jnt_qposadr_by_name('R_KNEE_P1')[0],self.interface.get_jnt_qposadr_by_name('R_KNEE_P')[0]]]

        self.task = walking_task.WalkingTask(client=self.interface,
                                             dt=control_dt,
                                             neutral_foot_orient=np.array([1, 0, 0, 0]),
                                             root_body='PELVIS_S',
                                             lfoot_body='L_ANKLE_R_S',
                                             rfoot_body='R_ANKLE_R_S',)

        self.task._goal_height_ref = 0.865
        self.task._total_duration = 0.5
        self.task._swing_duration = 0.3
        self.task._stance_duration = 0.2

        self.robot = robot.JVRC(self.close_jid, pdgains.T, control_dt, self.actuators, self.interface)

        self.target = []

        # define indices for action and obs mirror fns
        base_mir_obs = [0.1, -1, 2, -3,                 # root orient
                        -4, 5, -6,                      # root ang vel
                        -13, -14, -15, -16, -17, -18,   # motor pos [1]
                         -7,  -8,  -9, -10, -11, -12,   # motor pos [2]
                        -25, -26, -27, -28, -29, -30,   # motor vel [1]
                        -19, -20, -21, -22, -23, -24,   # motor vel [2]
                        -37, -38, -39, -40, -41, -42,
                        -31, -32, -33, -34, -35, -36]

        append_obs = [(len(base_mir_obs)+i) for i in range(6)]
        self.robot.clock_inds = append_obs[4:]
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()
        self.robot.mirrored_acts = [-6, -7, -8, -9, -10, -11, -0.1, -1, -2, -3, -4, -5,]

        # set action space
        action_space_size = len(self.robot.actuators)
        self.action_space = np.zeros(action_space_size)
        # set observation space
        self.base_obs_len = 49
        self.observation_space = np.zeros(self.base_obs_len)
        # self.imu = IMU()#********************************************************************

    def get_obs(self):
        # external state
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        ext_state = np.concatenate((self.task.mode, [self.task._goal_speed_ref],clock))
        #---------------------------------------训练用internal state--------------------------------------
        #'''
        #kbemf = np.random.normal(20, 5, 12)
        #print(kbemf)
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_orient = tf3.euler.euler2quat(root_r, root_p, 0)
        root_ang_vel = qvel[3:6]
        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()
        motor_tau = self.robot.tau_pd #- kbemf * motor_vel#实际应用的扭矩
        motor_pos = [motor_pos[i] for i in self.actuators]
        motor_vel = [motor_vel[i] for i in self.actuators]
        # height = self.interface.get_root_body_pos()
        # print(height)
        #print(f"仿真位置：\n{motor_pos} rad")
        #print(f"仿真速度：\n{motor_vel} rad/s")
        #print(f"仿真扭矩：\n{motor_tau} NM")
        #'''
        # internal state
        # ---------------------------------------测试用internal state--------------------------------------
        '''
        root_state = np.array(self.imu.get_root_state())*np.pi/180#root_r, root_p, root_y,root_ang_vel#********************************************************************8
        root_orient = tf3.euler.euler2quat(root_state[0], root_state[1], root_state[2])
        root_ang_vel = root_state[3:]
        motor_pos, motor_vel, motor_tau=self.robot.con.get_motor_state()
        
        #print(f"实际位置：\n{motor_pos} rad")
        #print(f"实际速度：\n{motor_vel} rad/s")
        #print(f"施加的扭矩：\n{motor_tau} NM")
        '''
        # --------------------------------------------------------------------------------------------------
        robot_state = np.concatenate([
            root_orient,
            root_ang_vel,
            motor_pos,
            motor_vel,
            motor_tau])

        state = np.concatenate([robot_state, ext_state])
        assert state.shape == (self.base_obs_len,)
        return state.flatten()

    def step(self, a):
        self.target.append(a)
        # make one control step
        self.robot.step(a)#输入关节的相对位置，根据初始姿态算出绝对位置，再根据绝对位置frame_skip次“pd控制算出扭矩并步进”
        #推进时间计算奖励

        self.task.step()#phase+1
        info={}
        #决策后，frame_skip次pd控制前，算出的关节扭矩和的关节绝对位置
        total_reward = self.task.calc_reward(self.robot.init_qpos[7:])
        # check if terminate
        done = self.task.done()#自碰撞或跑飞或摔倒则done
        obs = self.get_obs()
        return obs, total_reward, done, info

    def reset_model(self):
        # 初始半蹲姿态的基础上添加扰动
        self.init_qpos = self.robot.init_qpos[:]
        self.init_qvel = self.robot.init_qvel[:]

        # '''
        # dynamics randomization
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.interface.get_actuated_joint_names()]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(2, 8)    # actuated joint frictionloss
            self.model.dof_damping[jnt] = np.random.uniform(0.2, 3)        # actuated joint damping
            # self.model.dof_armature[jnt] *= np.random.uniform(0.90, 1.10)  # actuated joint armature
        # '''
        friction_loss = [0, 0, 0, 0, 0, 0, 2, 2, 2, 6, 0, 0, 2, 0, 0, 2,
                                           2, 2, 2, 6, 0, 0, 2, 0, 0, 2]
        self.model.dof_frictionloss = [i for i in friction_loss]

        dof_damping = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 0, 0, 1,
                                         1, 1, 1, 2, 0, 0, 1, 0, 0, 1]
        self.model.dof_damping = [i for i in dof_damping]
        # '''
        #print(self.model.dof_frictionloss)
            # 质量和质心随机化，5%范围
            #body_id = self.model.dof_bodyid[jnt]
            #self.model.body_mass[body_id] *= np.random.uniform(0.999, 1.001)
            #self.model.body_ipos[body_id] += np.random.uniform(-0.01, 0.01, size=3)
        # '''

        # 添加噪声扰动
        c = 0.02
        c = 0
        self.init_qpos = self.init_qpos + np.random.uniform(low=-c, high=c, size=self.model.nq)
        self.init_qvel = self.init_qvel + np.random.uniform(low=-c, high=c, size=self.model.nv)
        for chain in self.close_jid:
            self.init_qpos[chain[1]] = -self.init_qpos[chain[0]]
            self.init_qpos[chain[2]] = self.init_qpos[chain[0]]
        # modify init state acc to task
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]#把列表的[]去除（[0]->0）
        self.init_qpos[root_adr+2] = 0.882
        # '''
        # 父类的函数，mj_forward过
        self.set_state(np.asarray(self.init_qpos), np.asarray(self.init_qvel))

        self.task.reset()

        obs = self.get_obs()

        return obs

