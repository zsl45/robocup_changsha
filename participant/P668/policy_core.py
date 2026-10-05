"""P668 控制器核心：可调参数，以及从公开任务参数推导的物理量。

本模块不假定任何物理常量，全部由 `entry.CoverageRulePolicy.reset()` 从
`context.task`（协议公开的 `PublicTaskParams`）读出后填入 `PhysicsParams`。

推导依据（官方 coverage_bench.envs.physics 与 mpe2 World.integrate_state）：

    p_{k+1} = p_k + dt * v_k
    v_{k+1} = (1 - damping) * v_k + (drive_force / mass) * dt * a_k

动作盒是 [-1,1]^2，故力向量 F = drive_force * a 的模上界为 drive_force * sqrt(2)，
稳态速度上界

    v_max = dt * drive_force * sqrt(2) / (mass * damping)

在 configs/task-v1.yaml 下等于 0.1 * sqrt(2) / 0.25 = 0.5657（低于 robot_max_speed=1.0）。
注意单轴稳态只有 0.4、对角可达 0.5657；把上界写成 0.4 会人为砍掉约 29% 的速度能力。

由 v_0 = 0（场景把机器人初速清零）出发，单轴逐步累计可达半边长
（a=(1,0)，t=1..10）为

    [0, 0.01, 0.0275, 0.050625, 0.077969, 0.108477,
     0.141357, 0.176018, 0.212014, 0.249010]

即第 1 个计分步机器人完全无法移动，10 步单轴可达 0.249010、对角约 0.352。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SQRT2 = 1.4142135623730951


def clamp_action(action: np.ndarray) -> np.ndarray:
    """把动作限制到 [-1, 1] 并转为 float32。

    协议要求动作是 float32[2]；dtype 不符会被判 ACTION_INVALID，而不是被静默转换。
    """
    clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return np.ascontiguousarray(clipped, dtype=np.float32)


@dataclass
class PlanParams:
    """可调参数（由 artifacts/policy.json 覆盖）。"""

    kp: float = 3.0             # 位置回中增益；过大会让速度指令饱和从而冲过目标
    tvel_kp: float = 1.0        # 目标速度前馈权重
    lead_s: float = 0.0         # 预测提前量（秒）；实测 0 最好
    cap_scale: float = 1.0      # 速度上界缩放：1.0 = 物理上界；仅用于配对对照实验
    avoid_gain: float = 0.0     # 队友斥力增益；实测对分数无正向贡献，默认关闭
    avoid_radius: float = 0.12  # 斥力生效半径


@dataclass
class PhysicsParams:
    """公开物理参数及其派生量；派生量由 `derive()` 计算。"""

    dt: float = 0.1
    damping: float = 0.25
    drive_force: float = 1.0
    robot_mass: float = 1.0
    robot_radius: float = 0.05
    map_half_extent: float = 1.0
    target_radius: float = 0.15
    target_max_speed: float = 0.2
    horizon: int = 10

    vel_decay: float = field(init=False, default=0.75)
    accel_per_action: float = field(init=False, default=0.1)
    velocity_cap: float = field(init=False, default=0.5656854249492381)

    def derive(self, cap_scale: float = 1.0) -> None:
        """由公开物理参数推导积分系数与速度上界。"""
        self.vel_decay = 1.0 - self.damping
        self.accel_per_action = (self.drive_force / self.robot_mass) * self.dt
        self.velocity_cap = (
            self.dt * self.drive_force * SQRT2 / (self.robot_mass * self.damping)
        ) * cap_scale

    @classmethod
    def from_task(cls, task, horizon: int, cap_scale: float = 1.0) -> "PhysicsParams":
        """按协议公开的 PublicTaskParams 构造，并立即推导派生量。"""
        obj = cls(
            dt=float(task.dt),
            damping=float(task.damping),
            drive_force=float(task.drive_force),
            robot_mass=float(task.robot_mass),
            robot_radius=float(task.robot_radius),
            map_half_extent=float(task.map_half_extent),
            target_radius=float(task.target_radius),
            target_max_speed=float(task.target_max_speed),
            horizon=int(horizon),
        )
        obj.derive(cap_scale=cap_scale)
        return obj
