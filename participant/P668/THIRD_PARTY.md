# 第三方依赖与许可声明

本提交为纯规则（rule）方法，不含任何第三方代码、预训练模型或外部数据。

- 运行时依赖：仅使用评测镜像自带的 numpy，以及官方包 `coverage_bench` 的公开接口。
  未引入任何额外推理依赖，故 `requirements-infer.lock` 中无依赖条目。
- 未使用 PyTorch、Stable-Baselines3、SuperSuit 或任何学习框架的代码与权重。
- 策略中不含任何 `pickle` 系反序列化、外部下载或动态载入调用。
- 入口 `entry.py` 与 `policy_core.py` 为本人原创，可按仓库根目录 LICENSE（MIT）授权公开。
