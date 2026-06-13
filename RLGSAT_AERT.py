import pickle
from collections import defaultdict

import gym
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from src.models.modules.Attention import SelfAttention, MultiHeadAttention

from src.utils.metrics import Precision, NormalizedDCG
from src.utils.utils import random_walk


class RecEnv(gym.Env):
    def __init__(self, api_num, embedding_dim, api_embeds, device, batch_size=3):
        super(RecEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(api_num)
        self.observation_space = embedding_dim
        self.api_embeds = api_embeds.to(device)
        self.batch_size = batch_size
        self.device = torch.device(device)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return options['mashup'], {}

    def step(self, action):
        reward1 = []
        for i in range(action['topk_idx'].shape[0]):
            preds = self.api_embeds[action['topk_idx'][i]]
            ground_truth_label = torch.nonzero(action['ground-truth'][i], as_tuple=True)
            ground_truth = self.api_embeds[ground_truth_label]

            accumulate_reward = 0
            action_list = action['topk_idx'][i].to('cpu').tolist()
            ground_truth_label = torch.nonzero(
                action['ground-truth'][i], as_tuple=True
            )[0].to('cpu').tolist()

            for j in ground_truth_label:
                if j in action_list:
                    accumulate_reward += 1

            score = torch.matmul(preds, ground_truth.T)
            rows, cols = linear_sum_assignment(-score.detach().cpu().numpy())
            total_score = score[rows, cols]

            argmax_sum = (
                torch.sum(total_score * torch.abs(total_score)) / total_score.shape[0]
                + accumulate_reward
            )
            reward1.append(argmax_sum)

        reward1 = torch.tensor(reward1, device=self.device, dtype=torch.float32)
        return reward1


class GSAT(pl.LightningModule):
    @property
    def device(self):
        return self._device

    def __init__(
        self,
        data_dir,
        n,
        gamma,
        beta,
        weight_decay,
        critic_lr,
        actor_lr,
        device,
        embedding_model='openai',
        fine_tuning=False,
        topk=None,
        dataset='Mashup',
        RSGAP=False,
        warm_turn=8,
        exp_mode="legacy",   # 单参数控制实验模式
    ):
        super().__init__()

        self.device = torch.device(device)
        self.dataset = dataset
        self.fine_tuning = fine_tuning
        self.weight_decay = weight_decay
        self.topk = topk
        self.mashup_topk = 100
        self.RSGAP = RSGAP
        self.max_step = 4
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.n = n
        self.automatic_optimization = False
        self.gamma = gamma
        self.beta = beta

        # ===== 实验模式 =====
        self.exp_mode = exp_mode
        self.warmup_turn = warm_turn
        self._stage2_initialized = False
        self.warmup = True

        if dataset == 'Youshu':
            edge_path = '/standard/train_edges.emb'
        else:
            edge_path = '/train_edges.emb'

        if dataset == 'Youshu':
            self.input_channel = 96
            enhenced_edges = torch.load(data_dir + edge_path).numpy().T
            with open(data_dir + '/raw/train_item', 'rb') as f:
                user_item = pickle.load(f)
            self.mashup_embeds = nn.Embedding(user_item.shape[0], self.input_channel)
            self.api_embeds = nn.Embedding(user_item.shape[1], self.input_channel)
            self.hidden_channel = int(self.input_channel / 2)
            self.num_api = self.api_embeds.num_embeddings
            self.num_mashup = self.mashup_embeds.num_embeddings
        else:
            if embedding_model == 'openai':
                pre_data_dir = data_dir + '/preprocessed_data/openai_emb/'
                if dataset in ['Mashup', 'Huggingface']:
                    self.api_embeds = torch.stack(
                        torch.load(pre_data_dir + 'api_openai_text_embedding.pt'), dim=0
                    )
                    self.mashup_embeds = torch.stack(
                        torch.load(pre_data_dir + 'mashup_openai_text_embedding.pt'), dim=0
                    )
                else:
                    self.api_embeds = torch.load(pre_data_dir + 'api_openai_text_embedding.pt')
                    self.mashup_embeds = torch.load(pre_data_dir + 'mashup_openai_text_embedding.pt')
            elif embedding_model == 'bert' and fine_tuning is True:
                pre_data_dir = data_dir + '/preprocessed_data/description/'
                self.mashup_embeds = torch.load(pre_data_dir + 'pre_bert_mashup_embedding.pt')
                self.api_embeds = torch.load(pre_data_dir + 'pre_bert_api_embedding.pt')
            elif embedding_model == 'bert' and fine_tuning is False:
                pre_data_dir = data_dir + '/preprocessed_data/bert/'
                self.api_embeds = torch.load(pre_data_dir + 'bert_apis_embeddings.emb')
                self.mashup_embeds = torch.load(pre_data_dir + 'bert_mashup_embeddings.emb')[1]
            else:
                pre_data_dir = data_dir + '/preprocessed_data/word2vec/'
                self.api_embeds = torch.stack(
                    torch.load(pre_data_dir + 'api_word2vec_text_embedding.pt'), dim=0
                )
                self.mashup_embeds = torch.stack(
                    torch.load(pre_data_dir + 'mashup_word2vec_text_embedding.pt'), dim=0
                )

            enhenced_edges = torch.load(data_dir + edge_path).numpy().T
            self.input_channel = self.api_embeds.shape[1]
            self.hidden_channel = int(self.api_embeds.shape[1] / 2)
            self.num_api = self.api_embeds.size(0)
            self.num_mashup = self.mashup_embeds.size(0)

        edge_index = []
        for edge in enhenced_edges.T:
            edge_index.append([edge[0], edge[1]])
            edge_index.append([edge[1], edge[0]])
        self.edge_index = torch.tensor(edge_index).T.to(self.device)

        self.api_embeds = self.api_embeds.to(self.device)
        self.mashup_embeds = self.mashup_embeds.to(self.device)

        self.GRU = nn.GRU(self.input_channel, self.input_channel)
        self.mashup_api = {}
        self.api_mashup = {}
        for edge in enhenced_edges.T:
            if int(edge[0]) not in self.mashup_api:
                self.mashup_api[int(edge[0])] = [int(edge[1])]
            else:
                self.mashup_api[int(edge[0])].append(int(edge[1]))

            if int(edge[1]) not in self.api_mashup:
                self.api_mashup[int(edge[1])] = [int(edge[0])]
            else:
                self.api_mashup[int(edge[1])].append(int(edge[0]))

        self.mashup_api.update(self.api_mashup)

        node_sequence_list = []
        for node in range(self.num_mashup + self.num_api):
            seq = random_walk(self.mashup_api, node, steps=20)
            node_sequence_list.append(torch.tensor(seq, dtype=torch.long))
        self.node_sequence_list = pad_sequence(node_sequence_list, batch_first=True).to(self.device)

        self.node_attention_mlp = nn.Linear(self.input_channel, self.input_channel)
        self.motivation = nn.LeakyReLU()
        self.node_attention = SelfAttention(self.input_channel)

        self.mashup_attention_1 = MultiHeadAttention(self.input_channel, self.input_channel)
        self.mashup_attention_2 = MultiHeadAttention(self.input_channel, self.input_channel)

        self.mashups_MLP = nn.Sequential(
            nn.Linear(self.input_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.input_channel)
        )
        self.api_MLP = nn.Sequential(
            nn.Linear(self.input_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.input_channel)
        )

        self.step_MLP = nn.Sequential(
            nn.Linear(self.num_api, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.num_api),
        )

        self.critic = Critic(self.input_channel, self.api_embeds, device=self.device)

        self.deduction_proj = nn.Linear(self.input_channel, self.input_channel)
        self.deduction_gate = nn.Linear(2 * self.input_channel, self.input_channel)

        self.reward_list = []

        self.P = {}
        self.P_val = {}
        self.DCG = {}
        self.DCG_val = {}
        for k in self.topk:
            self.P[k] = Precision(k).to(self.device)
            self.P_val[k] = Precision(k).to(self.device)
            self.DCG[k] = NormalizedDCG(k).to(self.device)
            self.DCG_val[k] = NormalizedDCG(k).to(self.device)

        self.criterion = torch.nn.MultiLabelSoftMarginLoss()
        self.env = RecEnv(
            self.api_embeds.shape[0],
            self.api_embeds.shape[1],
            self.api_embeds,
            device=self.device
        )

        # ===== 根据 exp_mode 建立实验配置 =====
        self._build_exp_mode()

    # ------------------------------------------------------------------
    # 实验模式配置
    # ------------------------------------------------------------------
    def _build_exp_mode(self):
        aliases = {
            "s1best_s2_1": "s2_1",
            "s1best_s2_3": "s2_3",
            "s1best_s2_10": "s2_10",
            "s1last_s2_3": "s2_3",
        }

        mode = aliases.get(self.exp_mode, self.exp_mode)

        presets = {
            # 保持你当前旧逻辑
            "legacy": {
                "enable_stage2": True,
                "stage2_epoch_limit": None,
                "stage2_frozen": False,
                "stage2_reset_optim": False,
                "stage2_reset_critic": False,
                "stage2_freeze_backbone": False,
                "stage2_actor_loss_scale": 0.01,
                "stage2_beta_scale": 1.0,
                "stage2_actor_lr_scale": 1.0,
                "stage2_critic_lr_scale": 1.0,
                "stage2_adv_norm": False,
                "stage2_adv_clip": None,
            },

            # 仅一阶段，不启用二阶段
            "s1_only": {
                "enable_stage2": False,
                "stage2_epoch_limit": 0,
                "stage2_frozen": False,
                "stage2_reset_optim": False,
                "stage2_reset_critic": False,
                "stage2_freeze_backbone": False,
                "stage2_actor_loss_scale": 0.01,
                "stage2_beta_scale": 1.0,
                "stage2_actor_lr_scale": 1.0,
                "stage2_critic_lr_scale": 1.0,
                "stage2_adv_norm": False,
                "stage2_adv_clip": None,
            },

            # 稳定化二阶段：1 / 3 / 10 epochs
            "s2_1": {
                "enable_stage2": True,
                "stage2_epoch_limit": 1,
                "stage2_frozen": False,
                "stage2_reset_optim": True,
                "stage2_reset_critic": True,
                "stage2_freeze_backbone": False,
                "stage2_actor_loss_scale": 0.01,
                "stage2_beta_scale": 0.1,
                "stage2_actor_lr_scale": 0.1,
                "stage2_critic_lr_scale": 0.5,
                "stage2_adv_norm": True,
                "stage2_adv_clip": 5.0,
            },
            "s2_3": {
                "enable_stage2": True,
                "stage2_epoch_limit": 3,
                "stage2_frozen": False,
                "stage2_reset_optim": True,
                "stage2_reset_critic": True,
                "stage2_freeze_backbone": False,
                "stage2_actor_loss_scale": 0.01,
                "stage2_beta_scale": 0.1,
                "stage2_actor_lr_scale": 0.1,
                "stage2_critic_lr_scale": 0.5,
                "stage2_adv_norm": True,
                "stage2_adv_clip": 5.0,
            },
            "s2_10": {
                "enable_stage2": True,
                "stage2_epoch_limit": 10,
                "stage2_frozen": False,
                "stage2_reset_optim": True,
                "stage2_reset_critic": True,
                "stage2_freeze_backbone": False,
                "stage2_actor_loss_scale": 0.01,
                "stage2_beta_scale": 0.1,
                "stage2_actor_lr_scale": 0.1,
                "stage2_critic_lr_scale": 0.5,
                "stage2_adv_norm": True,
                "stage2_adv_clip": 5.0,
            },

            # 二阶段流程存在，但冻结不更新
            "frozen_s2_3": {
                "enable_stage2": True,
                "stage2_epoch_limit": 3,
                "stage2_frozen": True,
                "stage2_reset_optim": False,
                "stage2_reset_critic": False,
                "stage2_freeze_backbone": False,
                "stage2_actor_loss_scale": 0.0,
                "stage2_beta_scale": 0.1,
                "stage2_actor_lr_scale": 1.0,
                "stage2_critic_lr_scale": 1.0,
                "stage2_adv_norm": False,
                "stage2_adv_clip": None,
            },
        }

        if mode not in presets:
            raise ValueError(f"Unknown exp_mode: {self.exp_mode}. Available: {list(presets.keys())}")

        cfg = presets[mode]
        for k, v in cfg.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # 常规方法
    # ------------------------------------------------------------------
    def forward(self, users, state):
        embeddings = torch.cat([self.mashup_embeds, self.api_embeds], dim=0)

        seq_nodes = self.node_sequence_list[users]
        seq_emb = embeddings[seq_nodes]
        enhanced_embedding = self.GRU(seq_emb)[0][:, 0, :]

        input1 = self.node_attention_mlp(embeddings)
        input1 = self.motivation(input1)
        input1 = self.node_attention(input1, self.edge_index)

        embeddings = embeddings.clone()
        embeddings.index_add_(0, users, enhanced_embedding)
        embeddings += input1

        mashup_embeddings = embeddings[:self.num_mashup]
        user_emb = mashup_embeddings[users]
        sim = user_emb @ mashup_embeddings.T
        _, idx = torch.topk(sim, k=self.mashup_topk, dim=-1)
        neighbor_emb = mashup_embeddings[idx]
        mashup_embeddings = torch.cat([user_emb.unsqueeze(1), neighbor_emb], dim=1)

        mashup_embeddings = self.mashup_attention_1(
            mashup_embeddings, mashup_embeddings, mashup_embeddings
        )[0]
        mashup_embeddings = self.mashup_attention_2(
            mashup_embeddings, mashup_embeddings, mashup_embeddings
        )[0]
        mashup_embeddings = mashup_embeddings[:, 0, :].squeeze(1)

        embeddings = embeddings.clone()
        embeddings[users] = mashup_embeddings

        mashup_embeddings = self.mashups_MLP(mashup_embeddings)
        api_embeddings = self.api_MLP(embeddings[self.num_mashup:])

        state_vec = state.squeeze(0).float()  # [B, N]
        api_expand = api_embeddings.unsqueeze(0).expand(state_vec.shape[0], -1, -1)  # [B,N,D]
        attn_score = torch.sum(api_expand * mashup_embeddings.unsqueeze(1), dim=-1)  # [B,N]
        attn_score = attn_score.masked_fill(~state_vec.bool(), float('-inf'))

        empty_mask = (state_vec.sum(dim=1) == 0)
        if empty_mask.any():
            attn_score[empty_mask] = 0.0

        attn_weight = torch.softmax(attn_score, dim=1)
        implemented = torch.sum(attn_weight.unsqueeze(-1) * api_expand, dim=1)
        deduction = self.deduction_proj(implemented)
        gate = torch.sigmoid(
            self.deduction_gate(torch.cat([mashup_embeddings, implemented], dim=-1))
        )
        remaining_mashup_embeddings = mashup_embeddings - gate * deduction

        return remaining_mashup_embeddings, api_embeddings

    def _get_actor_parameters(self):
        critic_param_ids = {id(p) for p in self.critic.parameters()}
        return [p for p in self.parameters() if id(p) not in critic_param_ids]

    def _select_actions_sequentially(self, score_raw, base_invalid_mask, k=5, forced_actions=None):
        B, N = score_raw.shape
        device = score_raw.device

        running_mask = base_invalid_mask.clone()
        selected_actions = []
        selected_log_probs = []
        step_entropies = []
        fallback_steps = 0
        sampling_steps = 0

        temperature = float(getattr(self, "train_sampling_temperature", 1.0))
        temperature = max(temperature, 1e-6)

        for t in range(k):
            masked_score = score_raw.masked_fill(running_mask, -1e9)

            all_invalid = running_mask.all(dim=1)
            if all_invalid.any():
                fallback_steps += int(all_invalid.any().item())
                fallback_score = score_raw.masked_fill(base_invalid_mask, -1e9)
                base_all_invalid = base_invalid_mask.all(dim=1)
                if base_all_invalid.any():
                    fallback_score[base_all_invalid] = score_raw[base_all_invalid]
                masked_score[all_invalid] = fallback_score[all_invalid]

            score_for_policy = masked_score
            if self.training and forced_actions is None and temperature != 1.0:
                score_for_policy = masked_score / temperature

            log_probs = F.log_softmax(score_for_policy, dim=1)
            probs = log_probs.exp()
            step_entropy = -(probs * log_probs).sum(dim=1)

            if forced_actions is not None:
                action_t = forced_actions[:, t]
            elif self.training:
                action_t = torch.multinomial(probs, num_samples=1).squeeze(1)
                sampling_steps += 1
            else:
                action_t = probs.argmax(dim=1)

            log_prob_t = log_probs.gather(1, action_t.unsqueeze(1)).squeeze(1)

            selected_actions.append(action_t)
            selected_log_probs.append(log_prob_t)
            step_entropies.append(step_entropy)

            running_mask = running_mask.clone()
            running_mask[torch.arange(B, device=device), action_t] = True

        self._last_policy_sampling_steps = sampling_steps

        selected_actions = torch.stack(selected_actions, dim=1)
        selected_log_probs = torch.stack(selected_log_probs, dim=1)
        step_entropies = torch.stack(step_entropies, dim=1)
        next_state = base_invalid_mask.float().clone()
        next_state.scatter_(1, selected_actions, 1.0)

        return selected_actions, selected_log_probs, step_entropies, next_state, fallback_steps

    def _decode_k_per_step(self, total_k=20):
        return max(1, int(total_k / self.max_step))

    def _rollout_policy(self, users, init_state, rollout_steps=None, k_per_step=None):
        if rollout_steps is None:
            rollout_steps = self.max_step
        if k_per_step is None:
            k_per_step = self._decode_k_per_step(total_k=20)

        next_state = init_state.clone().float()
        final_score = torch.full_like(next_state, -1e9)
        all_actions = []
        all_log_probs = []
        all_entropies = []
        total_fallback_steps = 0

        for _ in range(rollout_steps):
            mashup_embeddings, api_embeddings = self.forward(users, next_state.unsqueeze(0))
            score_raw = torch.matmul(mashup_embeddings, api_embeddings.T)

            selected_action, selected_log_probs, step_entropies, next_state, fallback_steps = \
                self._select_actions_sequentially(
                    score_raw=score_raw,
                    base_invalid_mask=next_state.bool(),
                    k=k_per_step,
                    forced_actions=None
                )

            selected_rank_score = selected_log_probs.exp()
            final_score.scatter_(1, selected_action, selected_rank_score)

            all_actions.append(selected_action)
            all_log_probs.append(selected_log_probs)
            all_entropies.append(step_entropies.mean(dim=1))
            total_fallback_steps += fallback_steps

        all_actions = torch.cat(all_actions, dim=1)
        all_log_probs = torch.cat(all_log_probs, dim=1)
        all_entropies = torch.stack(all_entropies, dim=1)

        return {
            "next_state": next_state,
            "final_score": final_score,
            "selected_action": all_actions,
            "joint_log_prob": all_log_probs.sum(dim=1),
            "entropy": all_entropies.mean(dim=1),
            "fallback_steps": total_fallback_steps,
        }

    # ------------------------------------------------------------------
    # Stage-2 辅助函数
    # ------------------------------------------------------------------
    def _stage2_elapsed_epochs(self):
        return max(0, self.current_epoch - self.warmup_turn)

    def _stage2_update_enabled(self):
        if self.warmup:
            return False
        if not self.enable_stage2:
            return False
        if self.stage2_frozen:
            return False
        if self.stage2_epoch_limit is None:
            return True
        return self._stage2_elapsed_epochs() < self.stage2_epoch_limit

    def _unwrap_optimizer(self, optimizer):
        return optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer

    def _reset_optimizer_state(self, optimizer, lr=None):
        opt = self._unwrap_optimizer(optimizer)
        opt.state = defaultdict(dict)
        if lr is not None:
            for pg in opt.param_groups:
                pg["lr"] = lr

    def reset_critic_parameters(self):
        def _reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        self.critic.apply(_reset)

    def freeze_backbone_for_stage2(self):
        for p in self.parameters():
            p.requires_grad = False

        for module in [self.deduction_proj, self.deduction_gate]:
            for p in module.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_train_start(self) -> None:
        pass

    def on_train_epoch_start(self) -> None:
        prev_warmup = getattr(self, "warmup", True)

        # s1_only: 永远保持在一阶段
        if not self.enable_stage2:
            self.warmup = True
        else:
            self.warmup = self.current_epoch < self.warmup_turn

        enter_stage2 = prev_warmup and (not self.warmup) and self.enable_stage2 and (not self._stage2_initialized)

        if enter_stage2:
            print(f"[Stage2] Entering stage2 at epoch {self.current_epoch}, exp_mode={self.exp_mode}")

            if self.stage2_reset_critic:
                print("[Stage2] Reset critic parameters")
                self.reset_critic_parameters()

            if self.stage2_freeze_backbone:
                print("[Stage2] Freeze backbone")
                self.freeze_backbone_for_stage2()
                self.print_trainable_parameters()

            if self.stage2_reset_optim:
                print("[Stage2] Reset optimizer states and apply stage2 learning rates")
                optimizer_actor, optimizer_critic = self.optimizers()
                self._reset_optimizer_state(
                    optimizer_actor,
                    self.actor_lr * self.stage2_actor_lr_scale
                )
                self._reset_optimizer_state(
                    optimizer_critic,
                    self.critic_lr * self.stage2_critic_lr_scale
                )

            self._stage2_initialized = True

    def print_trainable_parameters(self):
        print("Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        optimizer_actor, optimizer_critic = self.optimizers()

        state, action, reward = batch['state'], batch['action'], batch['reward']

        action_sum = state['action_sum'].squeeze(0).to(self.device).float()  # [B, N]
        users = state['users'].squeeze(0).to(self.device).long()  # [B]
        reward = reward.squeeze(0).to(self.device).float()  # [B]
        B = users.shape[0]

        # --------------------------------------------------
        # 1. 当前策略前向：得到 mashup / api 表示
        # --------------------------------------------------
        mashup_embeddings, api_embeddings = self.forward(
            users, state['action_sum'].to(self.device)
        )

        score_raw = torch.matmul(mashup_embeddings, api_embeddings.T)  # [B, N]
        base_invalid_mask = action_sum.bool()  # [B, N]

        current_selected_action, current_selected_log_probs, current_step_entropies, current_next_state, fallback_steps = \
            self._select_actions_sequentially(
                score_raw=score_raw,
                base_invalid_mask=base_invalid_mask,
                k=5,
                forced_actions=None
            )

        current_joint_log_prob = current_selected_log_probs.sum(dim=1)
        entropy = current_step_entropies.mean(dim=1)

        masked_score_for_log = score_raw.masked_fill(base_invalid_mask, -1e9)
        all_invalid = base_invalid_mask.all(dim=1)
        if all_invalid.any():
            masked_score_for_log[all_invalid] = score_raw[all_invalid]

        value = self.critic(action_sum, mashup_embeddings.detach()).squeeze(-1)

        # --------------------------------------------------
        # 2. warm-up 还是 online
        # --------------------------------------------------
        if self.warmup:
            selected_action = action['topk_idx'].squeeze(0).to(self.device).long()
            next_state_for_target = action['api_index'].squeeze(0).to(self.device).float()
            reward_used = reward

            _, selected_log_probs, step_entropies, _, forced_fallback_steps = \
                self._select_actions_sequentially(
                    score_raw=score_raw,
                    base_invalid_mask=base_invalid_mask,
                    k=selected_action.shape[1],
                    forced_actions=selected_action
                )

            joint_log_prob = selected_log_probs.sum(dim=1)
            entropy = step_entropies.mean(dim=1)
            fallback_steps += forced_fallback_steps

            with torch.no_grad():
                next_mashup_embeddings, _ = self.forward(users, next_state_for_target.unsqueeze(0))
                next_value = self.critic(
                    next_state_for_target, next_mashup_embeddings.detach()
                ).squeeze(-1)

            episode_reward = reward_used.sum().item()

        else:
            action_new = {
                'api': current_next_state,
                'ground-truth': action['ground-truth'].squeeze(0).to(self.device),
                'api_index': current_next_state,
                'topk_idx': current_selected_action
            }

            reward_new = self.env.step(action=action_new).to(self.device).float()
            if reward_new.dim() > 1:
                reward_new = reward_new.squeeze(0)

            reward_used = reward_new
            next_state_for_target = current_next_state
            joint_log_prob = current_joint_log_prob

            with torch.no_grad():
                next_mashup_embeddings, _ = self.forward(users, next_state_for_target.unsqueeze(0))
                next_value = self.critic(
                    next_state_for_target, next_mashup_embeddings.detach()
                ).squeeze(-1)

            episode_reward = reward_used.sum().item()

        self.reward_list.append(episode_reward)

        # --------------------------------------------------
        # 3. TD target / advantage
        # --------------------------------------------------
        target_value = reward_used + self.gamma * next_value
        target_value = target_value.detach()

        advantage = (target_value - value).detach()
        if not self.warmup:
            if self.stage2_adv_norm:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            if self.stage2_adv_clip is not None:
                advantage = torch.clamp(advantage, -self.stage2_adv_clip, self.stage2_adv_clip)

        # --------------------------------------------------
        # 4. Actor loss
        # --------------------------------------------------
        beta_used = self.beta if self.warmup else (self.beta * self.stage2_beta_scale)
        actor_loss = (-(joint_log_prob) * advantage).mean() - beta_used * entropy.mean()

        if not self.warmup:
            actor_loss = actor_loss * self.stage2_actor_loss_scale

        # --------------------------------------------------
        # 5. Critic loss
        # --------------------------------------------------
        critic_loss = F.mse_loss(value, target_value)

        # --------------------------------------------------
        # 6. Optimize
        # --------------------------------------------------
        update_enabled = True if self.warmup else self._stage2_update_enabled()

        if update_enabled:
            actor_params = self._get_actor_parameters()
            optimizer_actor.zero_grad()
            self.manual_backward(actor_loss)
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            self.manual_backward(critic_loss)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            optimizer_critic.step()

        # --------------------------------------------------
        # 7. Logging
        # --------------------------------------------------
        self.log("actor_loss", actor_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        self.log("critic_loss", critic_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        self.log("train_reward_step", episode_reward, prog_bar=True, on_step=True, on_epoch=False, batch_size=B)
        self.log("entropy", entropy.mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("policy_sampling_steps", float(getattr(self, "_last_policy_sampling_steps", 0)),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("advantage_mean", advantage.mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("advantage_abs_mean", advantage.abs().mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("policy_fallback_steps", float(fallback_steps), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)

        top5_values = torch.topk(masked_score_for_log, k=5, dim=1).values
        self.log("reward_mean", reward_used.mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("reward_std", reward_used.std(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("value_mean", value.mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("value_std", value.std(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("target_value_mean", target_value.mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("target_value_std", target_value.std(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("td_abs_mean", (target_value - value).abs().mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("advantage_std", advantage.std(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("top1_score_mean", top5_values[:, 0].mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("top5_score_mean", top5_values[:, 4].mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("top1_top5_gap_mean", (top5_values[:, 0] - top5_values[:, 4]).mean(),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=B)

        self.log("stage2_active", 0.0 if self.warmup else 1.0,
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("stage2_elapsed_epochs",
                 float(self._stage2_elapsed_epochs()) if not self.warmup else 0.0,
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("stage2_update_enabled", float(update_enabled and (not self.warmup)),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("stage2_frozen", float((not self.warmup) and self.stage2_frozen),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("beta_used", float(beta_used),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=B)

        return actor_loss + critic_loss

    # ------------------------------------------------------------------
    # 验证 / 测试
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        user = batch['users'].to(self.device)
        pos_items = batch['pos_items'].to(self.device)

        init_state = torch.zeros(user.shape[0], self.api_embeds.shape[0], device=self.device)
        rollout = self._rollout_policy(
            users=user,
            init_state=init_state,
            rollout_steps=self.max_step,
            k_per_step=self._decode_k_per_step(total_k=20)
        )

        final_score = rollout["final_score"]

        if not self.trainer.sanity_checking:
            for k in self.topk:
                self.P_val[k].update(final_score, pos_items)
                self.DCG_val[k].update(final_score, pos_items)
                self.log("val/P@" + str(k), self.P_val[k].compute(), on_step=False, on_epoch=True, prog_bar=True)
                self.log("val/DCG@" + str(k), self.DCG_val[k].compute(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        user = batch['users'].to(self.device)
        pos_items = batch['pos_items'].to(self.device)

        init_state = torch.zeros(user.shape[0], self.api_embeds.shape[0], device=self.device)
        rollout = self._rollout_policy(
            users=user,
            init_state=init_state,
            rollout_steps=self.max_step,
            k_per_step=self._decode_k_per_step(total_k=20)
        )

        final_score = rollout["final_score"]

        if not self.trainer.sanity_checking:
            for k in self.topk:
                self.P[k].update(final_score, pos_items)
                self.DCG[k].update(final_score, pos_items)
                self.log("test/P@" + str(k), self.P[k].compute(), on_step=False, on_epoch=True)
                self.log("test/DCG@" + str(k), self.DCG[k].compute(), on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        actor_params = self._get_actor_parameters()
        optimizer_actor = torch.optim.Adam(
            actor_params, lr=self.actor_lr, weight_decay=self.weight_decay
        )
        optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay
        )
        return [optimizer_actor, optimizer_critic]

    @device.setter
    def device(self, value):
        self._device = value


class Critic(nn.Module):
    def __init__(self, state_dim, api_embeds, device):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(2 * api_embeds.shape[0], 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)
        self.api_embeds = api_embeds.T.to(device)

    def forward(self, state1, mashup_embeds):
        state = torch.matmul(mashup_embeds, self.api_embeds)
        state = torch.cat([state, state1], dim=1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)