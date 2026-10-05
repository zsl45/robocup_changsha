"""P668 参赛策略入口：多目标持续覆盖的精细速度匹配控制器。

官方评测从本模块调用 build_policy(context)，返回对象需实现 reset / act / close。
方案定位：method_type = rule，不做学习，直接利用协议公开的物理与观测做控制。

物理常量一律在 reset() 时从 `context.task`（公开的 PublicTaskParams）读取，
不硬编码任何 dt / damping / drive_force / robot_mass / target_radius /
target_max_speed / sense_radius / map_half_extent / horizon。

速度上界由物理参数推导（这是本版修正的一处关键错误）：

    动作盒是 [-1,1]^2，力向量 F = drive_force * a 的模上界为 drive_force * sqrt(2)，
    故稳态速度上界 v_max = dt * drive_force * sqrt(2) / (mass * damping)，
    在 task-v1.yaml 下为 0.1 * sqrt(2) / 0.25 = 0.5657（低于 robot_max_speed=1.0）。
    旧版把它写成单轴稳态 0.4，等于人为把可用速度砍掉 29%，实测损失约 2.8~3.4 分。

控制律：

    desired_v = k_t * v_target + k_p * (p_target - p_self)      # 再按 v_max 限速
    action    = (desired_v - decay * v_self) / (dt * force / mass)

核心认识：官方积分 p += dt*v 之后再 v = (1-damping)*v + (dt*force/mass)*a，
且场景把机器人初速清零，故 v_k = v_max*(1-(1-damping)^k)。
单轴 10 步累计可达约 0.249010、对角约 0.352，而覆盖半径 0.15；第 1 个计分步
机器人完全无法移动。覆盖率因此主要由起手位置与这约 0.25 的机动量用得准不准决定，
长距离路径规划既不可达也无收益。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # 官方运行时会先把本人目录加入 sys.path，故绝对导入可用
    from policy_core import (
        PhysicsParams,
        PlanParams,
        clamp_action,
    )
except ImportError:  # 以包形式导入时回退到相对导入
    from .policy_core import (  # type: ignore[no-redef]
        PhysicsParams,
        PlanParams,
        clamp_action,
    )

_CONFIG_NAME = "policy.json"
_DEFAULT_PARAMS = PlanParams()


class CoverageRulePolicy:
    """单个机器人的规则策略实例；每个 agent 独立持有，不共享可变状态。"""

    def __init__(self, params: PlanParams | None = None):
        self.p = params if params is not None else PlanParams()
        self.ph = PhysicsParams()

        self.half_extent = 1.0
        self.num_targets = 0
        self.num_agents = 0
        self.horizon = 1
        self.agent_index = 0

        self._self_pos = np.zeros(2, dtype=np.float64)
        self._self_vel = np.zeros(2, dtype=np.float64)
        self._prev_aim: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------ 生命周期

    def reset(self, context) -> None:
        """回合开始：从公开任务参数读取物理量，并清空全部回合内记忆。"""
        self.num_agents = int(context.num_agents)
        self.num_targets = int(context.num_targets)
        self.horizon = max(1, int(context.horizon))
        self.agent_index = int(context.agent_index)

        # 物理常量一律来自协议公开参数，不再硬编码
        self.ph = PhysicsParams.from_task(
            context.task, horizon=self.horizon, cap_scale=self.p.cap_scale
        )
        # 官方另有 robot_max_speed 上限；本配置下 v_max=0.5657 < 1.0，此处仍显式取 min
        robot_max_speed = float(getattr(context.task, "robot_max_speed", 0.0) or 0.0)
        if robot_max_speed > 0.0:
            self.ph.velocity_cap = min(self.ph.velocity_cap, robot_max_speed)

        self.half_extent = self.ph.map_half_extent
        self._self_pos = np.zeros(2, dtype=np.float64)
        self._self_vel = np.zeros(2, dtype=np.float64)
        self._prev_aim = {}

    def close(self) -> None:
        """释放引用；本策略不持有外部资源。"""
        self._prev_aim = {}

    # ------------------------------------------------------------ 决策

    def act(self, observation) -> np.ndarray:
        """依据本机器人当前局部观测输出一步动作，形状 (2,)，分量落在 [-1, 1]。"""
        self_state = np.asarray(observation["self_state"], dtype=np.float64)
        self._self_pos = self_state[:2] * self.half_extent
        self._self_vel = self_state[2:4]

        targets_rel = np.asarray(observation["targets"], dtype=np.float64)
        target_visible = np.asarray(observation["target_visible"], dtype=bool)

        # 只对"本步真正可见"的目标做控制：机动量有限，扑向旧估计没有收益，
        # 反而会把已经贴住的目标丢掉。
        if not bool(np.any(target_visible)):
            self._prev_aim = {}
            return clamp_action(-self._self_vel / self.ph.accel_per_action)

        visible = target_visible
        # 距离用归一化相对位置直接算（sqrt(dx^2+dy^2)*L，等价于乘以 map_half_extent）
        dists = np.linalg.norm(targets_rel[:, :2], axis=1) * self.half_extent
        dists = np.where(visible, dists, np.inf)
        j = int(np.argmin(dists))

        aims = self._self_pos[None, :] + targets_rel[:, :2] * self.half_extent

        # 一阶差分估计目标速度（限幅到公开的目标速度上界）
        tv = np.zeros(2, dtype=np.float64)
        prev = self._prev_aim.get(j)
        if prev is not None:
            cap = self.ph.target_max_speed / self.ph.dt
            tv = np.clip((aims[j] - prev) / self.ph.dt, -cap, cap)
        self._prev_aim = {j: aims[j].copy()}

        aim = aims[j] + tv * self.p.lead_s
        desired_vel = self.p.tvel_kp * tv + self.p.kp * (aim - self._self_pos)
        speed = float(np.linalg.norm(desired_vel))
        if speed > self.ph.velocity_cap:
            desired_vel = desired_vel * (self.ph.velocity_cap / speed)

        action = (desired_vel - self.ph.vel_decay * self._self_vel) / self.ph.accel_per_action
        return self._apply_avoidance(action, observation)

    def _apply_avoidance(self, action: np.ndarray, observation) -> np.ndarray:
        """叠加对可见队友的短程斥力；默认增益为 0，实测对分数无正向贡献。"""
        if self.p.avoid_gain <= 0.0:
            return clamp_action(action)

        peers = np.asarray(observation["peers"], dtype=np.float64)
        peer_visible = np.asarray(observation["peer_visible"], dtype=bool)
        push = np.zeros(2, dtype=np.float64)
        for i in range(min(peers.shape[0], peer_visible.shape[0])):
            if not bool(peer_visible[i]):
                continue
            rel = peers[i, :2] * self.half_extent
            dist = float(np.linalg.norm(rel))
            if 1e-9 < dist < self.p.avoid_radius:
                strength = (self.p.avoid_radius - dist) / self.p.avoid_radius
                push = push + (-rel / dist) * strength
        if float(np.linalg.norm(push)) > 1e-9:
            action = np.asarray(action, dtype=np.float64) + self.p.avoid_gain * push
        return clamp_action(action)


def _load_params(artifact_dir: Path) -> PlanParams:
    """从 artifacts/policy.json 读取可调参数；缺失或字段不全时使用内置默认值。"""
    cfg_path = Path(artifact_dir) / _CONFIG_NAME
    if not cfg_path.is_file():
        return _DEFAULT_PARAMS
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return _DEFAULT_PARAMS
    fields = set(PlanParams.__dataclass_fields__)
    kwargs = {k: float(v) for k, v in raw.items() if k in fields}
    return PlanParams(**kwargs)


def build_policy(context):
    """官方唯一入口：按 artifacts 目录中的参数构建规则策略。"""
    return CoverageRulePolicy(params=_load_params(context.artifact_dir))
