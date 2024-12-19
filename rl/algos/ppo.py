"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import ray
from rl.envs import WrapEnv


class PPOBuffer:
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        #学习一次前的所有探索数据
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []#只和state有关（状态-动作的期望）
        self.returns = []#还和动作有关，从各条轨迹中提取（state,action）对应的价值

        self.ep_returns = [] # for logging
        self.ep_lens    = []

        self.gamma, self.lam = gamma, lam

        self.ptr = 0
        self.traj_idx = [0]#每条轨迹的起点在buffer中的索引

    def __len__(self):
        return len(self.states)

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        self.states  += [state.squeeze(0)]
        self.actions += [action.squeeze(0)]
        self.rewards += [reward.squeeze(0)]
        self.values  += [value.squeeze(0)]

        self.ptr += 1

    def finish_path(self, last_val=None):
        self.traj_idx += [self.ptr]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]#提取此条轨迹对应的rewards
        returns = []

        R = last_val.squeeze(0).copy()  # Avoid copy?
        for reward in reversed(rewards):
            # 计算轨迹每个状态采取某动作的对应的价值（后续的将得奖励和），越早的奖励权重越大
            R = self.gamma * R + reward
            returns.insert(0, R)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

    def get(self):
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )

class PPO:
    def __init__(self, args, save_path):
        self.gamma          = args['gamma']
        self.lam            = args['lam']
        self.lr             = args['lr']
        self.eps            = args['eps']
        self.ent_coeff      = args['entropy_coeff']
        self.clip           = args['clip']
        self.minibatch_size = args['minibatch_size']
        self.epochs         = args['epochs']
        self.max_traj_len   = args['max_traj_len']
        self.use_gae        = args['use_gae']
        self.n_proc         = args['num_procs']
        self.grad_clip      = args['max_grad_norm']
        self.mirror_coeff   = args['mirror_coeff']
        self.eval_freq      = args['eval_freq']
        self.recurrent      = False
        # batch_size depends on number of parallel envs
        self.batch_size = self.n_proc * self.max_traj_len
        #一个策略一次学习一个batch，batch_size=n_proc * maxstep(maxstep>max_traj_len)
        #maxstep为max_traj_len的k倍时，一个cpu的一次收集能容纳k条成功的/未提前终止的轨迹
        self.vf_coeff = 0.5
        self.target_kl = 0.04# By default, there is no limit on the kl div

        #self.total_steps = 0
        self.highest_reward = 50
        self.limit_cores = 0

        # counter for training iterations
        self.iteration_count = 0

        self.save_path = save_path
        self.eval_fn = os.path.join(self.save_path, 'eval.txt')
        with open(self.eval_fn, 'w') as out:
            out.write("test_ep_returns,test_ep_lens\n")

        self.train_fn = os.path.join(self.save_path, 'train.txt')
        with open(self.train_fn, 'w') as out:
            out.write("ep_returns,ep_lens\n")

        # os.environ['OMP_NUM_THREA DS'] = '1'
        # if args['redis_address'] is not None:
        #     ray.init(num_cpos=self.n_proc, redis_address=args['redis_address'])
        # else:
        #     ray.init(num_cpus=self.n_proc)

    def save(self, policy, critic, suffix=""):

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch model
        torch.save(policy, os.path.join(self.save_path, "actor" + suffix + filetype))
        torch.save(critic, os.path.join(self.save_path, "critic" + suffix + filetype))

    @ray.remote
    @torch.no_grad()
    def sample(self, env_fn, policy, critic, max_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):
        """
        Sample max_steps number of total timesteps, truncating
        trajectories if they exceed max_traj_len number of timesteps.
        """
        
        torch.set_num_threads(1)#Ray用的每个核上的PyTorch只用1个核，以免冲突
        env = WrapEnv(env_fn)  # TODO
        env.robot.iteration_count = self.iteration_count
        # memory最长max_steps，一个memory可能存放多条轨迹，每条轨迹最短到done或者memory_full截止，最长max_traj_len被截断
        memory = PPOBuffer(self.gamma, self.lam)
        memory_full = False

        while not memory_full:
            state = torch.Tensor(env.reset())
            done = False
            traj_len = 0
            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            if hasattr(critic, 'init_hidden_state'):
                critic.init_hidden_state()
            #一条轨迹
            while not done and traj_len < max_traj_len:
                action = policy(state, deterministic=deterministic, anneal=anneal)
                value = critic(state)
                next_state, reward, done, _ = env.step(action.numpy())

                memory.store(state.numpy(), action.numpy(), reward, value.numpy())
                memory_full = (len(memory) == max_steps)

                state = torch.Tensor(next_state)
                traj_len += 1

                if memory_full:
                    break

            value = critic(state)
            #唯有终止状态，状态价值等于奖励
            memory.finish_path(last_val=(not done) * value.numpy())#如果是训练成功就将value算作终止状态的价值，否则终止状态的价值视作0（对应失败时的done）
        return memory#学习一次前所有的探索数据

    #train时deterministic=False（非决定的，即加扰动），test时deterministic=True
    def sample_parallel(self, env_fn, policy, critic, min_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):

        worker = self.sample
        args = (self, env_fn, policy, critic, min_steps // self.n_proc, max_traj_len, deterministic, anneal, term_thresh)

        # Create pool of workers, each getting data for min_steps
        workers = [worker.remote(*args) for _ in range(self.n_proc)]#执行n_proc个self.sample(*args)
        result = ray.get(workers)

        # O(n)
        def merge(buffers):
            merged = PPOBuffer(self.gamma, self.lam)
            for buf in buffers:
                offset = len(merged)
                merged.states  += buf.states
                merged.actions += buf.actions
                merged.rewards += buf.rewards
                merged.values  += buf.values
                merged.returns += buf.returns

                merged.ep_returns += buf.ep_returns
                merged.ep_lens += buf.ep_lens

                merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]#每条轨迹起点随之多个buf合并发生偏移
                merged.ptr += buf.ptr

            return merged

        total_buf = merge(result)

        return total_buf

    def update_policy(self, obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=None, mirror_action=None):
        policy = self.policy
        critic = self.critic
        old_policy = self.old_policy

        values = critic(obs_batch)
        pdf = policy.distribution(obs_batch)#网络输出action的正态分布函数
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)

        old_pdf = old_policy.distribution(obs_batch)
        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)#policy网络作出action_batch决策的对数概率

        # ratio between old and new policy, should be one at the first iteration
        ratio = (log_probs - old_log_probs).exp()#exp^log(p(x)/q(x))即pdf(action)/old_pdf(action)，重要性采样

        # clipped surrogate loss
        #ratio刺激策略改变（探索），但如果策略改变过大（ratio过大），clip_loss会终止ratio的作用，将关注点转移到优势函数
        cpi_loss = ratio * advantage_batch * mask#越大越好
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask#越大越好
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        # only used for logging
        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip).float()).item()#被裁减的均值

        # Value loss using the TD(gae_lambda) target
        critic_loss = self.vf_coeff * F.mse_loss(return_batch, values)

        # 鼓励探索，熵越大随机性越强
        entropy_penalty = -(pdf.entropy() * mask).mean()

        # Mirror Symmetry Loss
        if mirror_observation is not None and mirror_action is not None:
            deterministic_actions = policy(obs_batch)
            mir_obs = mirror_observation(obs_batch)
            mirror_actions = policy(mir_obs)
            mirror_actions = mirror_action(mirror_actions)
            mirror_loss = (deterministic_actions - mirror_actions).pow(2).mean()
        else:
            mirror_loss = torch.Tensor([0])

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            #最小化KL散度来使得模型预测的分布接近真实的数据分布，仅用来判断每次学习是否提前终止
            approx_kl_div = torch.mean((ratio - 1) - log_ratio)

        return (
            actor_loss,
            entropy_penalty,
            critic_loss,
            approx_kl_div,
            mirror_loss,
            clip_fraction,
        )

    def train(self,
              env_fn,
              policy,
              critic,
              n_itr,#20000次学习
              anneal_rate=1.0):

        self.old_policy = deepcopy(policy)#每次学习policy更新多次，old policy才更新一次
        self.policy = policy
        self.critic = critic

        self.actor_optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr, eps=self.eps)

        #train_start_time = time.time()

        obs_mirr, act_mirr = None, None
        if hasattr(env_fn(), 'mirror_observation'):
            obs_mirr = env_fn().mirror_clock_observation

        if hasattr(env_fn(), 'mirror_action'):
            act_mirr = env_fn().mirror_action

        curr_anneal = 1.0
        curr_thresh = 0#0～0.35
        start_itr = 0
        ep_counter = 0
        do_term = False

        test_ep_lens = []
        test_ep_returns = []

        for itr in range(n_itr):
            self.iteration_count = itr
            #sample_start_time = time.time()
            # 平均奖励比较高时（最高3/4*max_traj_len）令动作的噪声标准差（勘探指数）减小/减小扰动，但最小不小于0.5*args.std_dev
            if self.highest_reward > (2/3)*self.max_traj_len and curr_anneal > 0.5:
                curr_anneal *= anneal_rate
            if do_term and curr_thresh < 0.35:
                curr_thresh = .1 * 1.0006**(itr-start_itr)
            batch = self.sample_parallel(env_fn, self.policy, self.critic, self.batch_size, self.max_traj_len, anneal=curr_anneal, term_thresh=curr_thresh)
            observations, actions, returns, values = map(torch.Tensor, batch.get())#获取n_proc个cpu收集的总四元组数据包
            num_samples = batch.storage_size()#num_samples每次都一样
            #elapsed = time.time() - sample_start_time
            #print("Sampling took {:.2f}s for {} steps.".format(elapsed, num_samples))

            # Normalize advantage
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or num_samples
            #self.total_steps += num_samples#多次学习的总采样次数

            self.old_policy.load_state_dict(policy.state_dict())#仅在每次学习前更新一次（获取上一轮最后的policy的更新）

            # Is false when 1.5*self.target_kl is breached
            continue_training = True
            #开始学习优化，相同的数据用self.epochs次，在kl达标的情况下尽可能学习
            #optimizer_start_time = time.time()
            for epoch in range(self.epochs):
                actor_losses = []
                entropies = []#熵
                critic_losses = []
                kls = []
                mirror_losses = []
                clip_fractions = []
                if self.recurrent:
                    #把所有采样的轨迹分批
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx)-1))#边数=点数-1
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=False)
                else:
                    # 把所有采样的四元组分批
                    random_indices = SubsetRandomSampler(range(num_samples))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)#按random_indices原序分批，每次返回一个minibatch
                #更新batch次
                for indices in sampler:
                    if self.recurrent:
                        obs_batch       = [observations[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        action_batch    = [actions[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        return_batch    = [returns[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        advantage_batch = [advantages[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        mask            = [torch.ones_like(r) for r in return_batch]

                        obs_batch       = pad_sequence(obs_batch, batch_first=False)
                        action_batch    = pad_sequence(action_batch, batch_first=False)
                        return_batch    = pad_sequence(return_batch, batch_first=False)
                        advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                        mask            = pad_sequence(mask, batch_first=False)
                    else:
                        obs_batch       = observations[indices]
                        action_batch    = actions[indices]
                        return_batch    = returns[indices]
                        advantage_batch = advantages[indices]
                        mask            = 1

                    scalars = self.update_policy(obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=obs_mirr, mirror_action=act_mirr)
                    actor_loss, entropy_penalty, critic_loss, approx_kl_div, mirror_loss, clip_fraction = scalars

                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy_penalty.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(approx_kl_div.item())
                    mirror_losses.append(mirror_loss.item())
                    clip_fractions.append(clip_fraction)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        #print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break
                    # clip_grad_norm_将参数的梯度的L2范数限制在一个指定的范围内。如果梯度的L2范数超过了这个范围，就会被缩放到这个范围以内，而保持梯度方向不变
                    self.actor_optimizer.zero_grad()
                    (actor_loss + self.mirror_coeff*mirror_loss + self.ent_coeff*entropy_penalty).backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
                    self.critic_optimizer.step()
                if not continue_training:
                    break
            #一次学习完成后
            #elapsed = time.time() - optimizer_start_time
            #print("Optimizer took: {:.2f}s".format(elapsed))

            if np.mean(batch.ep_lens) >= self.max_traj_len * 0.75:
                ep_counter += 1
            if do_term == False and ep_counter > 50:
                do_term = True
                start_itr = itr
            #elapsed = time.time() - train_start_time
            #print("Total time elapsed: {:.2f}s. Total steps: {} (fps={:.2f})".format(elapsed, self.total_steps, self.total_steps/elapsed))

            # save metrics
            with open(self.train_fn, 'a') as out:
                out.write("{},{}\n".format(np.mean(batch.ep_returns), np.mean(batch.ep_lens)))
            # #每eval_freq轮测试一次，每次测试都会保存有_itr过程的模型和统计图，仅当奖励破新高时才保存没有后缀的模型

            if itr % self.eval_freq == 0 :
                print("********** Iteration {} ***********".format(itr))
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (batch)', "%8.5g" % np.mean(batch.ep_returns)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', "%8.5g" % np.mean(batch.ep_lens)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Actor loss', "%8.3g" % np.mean(actor_losses)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Critic loss', "%8.3g" % np.mean(critic_losses)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mirror loss', "%8.3g" % np.mean(mirror_losses)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % np.mean(kls)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % np.mean(entropies)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Clip Fraction', "%8.3g" % np.mean(clip_fractions)) + "\n")
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()
                #evaluate_start = time.time()
                test = self.sample_parallel(env_fn, self.policy, self.critic, self.batch_size, self.max_traj_len, deterministic=True)
                #eval_time = time.time() - evaluate_start
                #print("evaluate time elapsed: {:.2f} s".format(eval_time))

                avg_eval_reward = np.mean(test.ep_returns)
                #print("====EVALUATE EPISODE====  (Return = {})".format(avg_eval_reward))

                # save metrics
                with open(self.eval_fn, 'a') as out:
                    out.write("{},{}\n".format(np.mean(test.ep_returns), np.mean(test.ep_lens)))
                test_ep_lens.append(np.mean(test.ep_lens))
                test_ep_returns.append(np.mean(test.ep_returns))
                plt.clf()
                xlabel = [i*self.eval_freq for i in range(len(test_ep_lens))]
                plt.plot(xlabel, test_ep_lens, color='blue', marker='o', label='Ep lens')
                plt.plot(xlabel, test_ep_returns, color='green', marker='o', label='Returns')
                plt.xticks(np.arange(0, itr+1, step=self.eval_freq))
                plt.xlabel('Iterations')
                plt.ylabel('Returns/Episode lengths')
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(self.save_path, 'eval.svg'), bbox_inches='tight')

                # save policy
                #self.save(policy, critic, "_" + repr(itr))

                # save as actor.pt, if it is best
                if self.highest_reward < avg_eval_reward:
                    self.highest_reward = avg_eval_reward
                    self.save(policy, critic)
                    print("save best reward: ",self.highest_reward)
