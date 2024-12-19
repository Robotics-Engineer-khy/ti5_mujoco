import numpy as np
import transforms3d as tf3
from tasks import rewards

class WalkingTask(object):
    """Dynamically stable walking on biped."""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 waist_r_joint='waist_r',
                 waist_p_joint='waist_p',
    ):
        self._client = client
        self._control_dt = dt
        self._neutral_foot_orient=neutral_foot_orient
        self._mass = self._client.get_robot_mass()
        # These depend on the robot, hardcoded for now
        # Ideally, they should be arguments to __init__
        self._goal_speed_ref = []
        self._goal_height_ref = []
        self._swing_duration = []
        self._stance_duration = []
        self._total_duration = []

        self._root_body_name = root_body
        self._lfoot_body_name = lfoot_body
        self._rfoot_body_name = rfoot_body

    def calc_reward(self, init_qpos):
        #只取角速度
        self.l_foot_vel = self._client.get_lfoot_body_vel()
        self.r_foot_vel = self._client.get_rfoot_body_vel()
        self.l_foot_vel = self.l_foot_vel[3:]
        self.r_foot_vel = self.r_foot_vel[3:]
        #左右脚获得的地面反作用力Ground Reaction Force，GRF>0
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]
        foot_frc_score = rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc)#0.15*（0～1）
        foot_vel_score = rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel)#0.15*（0～1）
        fwd_vel_score = rewards._calc_fwd_vel_reward(self)#（0～1）
        yaw_vel_score = rewards._calc_yaw_vel_reward(self)#0.05*（0～1）
        height_score = rewards._calc_height_reward(self) # 0.05*（0～1）
        qpos_score = rewards._calc_qpos_reward(self,init_qpos)
        qvel_score = rewards._calc_qvel_reward(self)
        body_orient_score=(rewards._calc_body_orient_reward(self,self._root_body_name)+rewards._calc_body_orient_reward(self,self._lfoot_body_name)+rewards._calc_body_orient_reward(self,self._rfoot_body_name))/3
        foot_height_score = rewards._calc_foot_pos_clock_reward(self)
        foot_distance_score = rewards._calc_feet_separation_reward(self)

        reward=np.array([foot_frc_score,foot_vel_score,fwd_vel_score,yaw_vel_score,height_score,
                         qpos_score,qvel_score,body_orient_score,foot_height_score,foot_distance_score])
        weight = np.array([0.15,0.15,0.1,0.2,0.1,0.1,0.1,0.1,0.1,0.1])
        if self.mode[1]==1:
            weight[2] = 0
            weight[0] += weight[-1]/2
            weight[1] += weight[-1]/2
            weight[-1] = 0
        else:
            weight[3] = 0
        total_reward=sum(reward*weight)
        return total_reward

    def step(self):
        if self._phase>self._period:
            self._phase=0
        self._phase+=1
        return

    def done(self):
        contact_flag = self._client.check_self_collisions()
        qpos = self._client.get_qpos()
        terminate_conditions = {"qpos[2]_ll":(qpos[2] < 0.75),
                                "qpos[2]_ul":(qpos[2] > 1),
                                "contact_flag":contact_flag,
        }
        done = True in terminate_conditions.values()
        return done

    def reset(self):
        #mode_id = np.random.randint(1)
        mode_id = 0
        self.mode = np.zeros(3)
        self.mode[mode_id] = 1
        if mode_id == 0:
            self._goal_speed_ref = np.random.choice([0, np.random.uniform(0.3, 0.4)])#前进速度
            self._goal_speed_ref = 0.3 #前进速度
        elif mode_id == 1:
            self._goal_speed_ref = np.random.uniform(-0.5, 0.5)#旋转角速度
        else:
            self._goal_speed_ref = 0#站立
        # 配置双足触觉传感器
        #print(self._goal_speed_ref)
        self.right_clock, self.left_clock = rewards.create_phase_reward(self._swing_duration,
                                                                        self._stance_duration,
                                                                        0.1,
                                                                        "grounded",
                                                                        1/self._control_dt)
        # number of control steps in one full cycle
        # (one full cycle includes left swing + right swing)
        self._period = np.floor(2*self._total_duration*(1/self._control_dt))
        # 初始化成一个行走周期中的任意阶段（站摆情况）
        self._phase = np.random.randint(0, self._period)

