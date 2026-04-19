# PIL Prototype

This repository now contains a runnable prototype of **Private Incentive Learning with Adaptive Privacy Scheduling (PIL-APS)** built from scratch around the notes in `PIL.md`.

## Layout

- `core/models.py`: sender, posterior, and actor networks.
- `core/trainer.py`: PIL config, adaptive privacy scheduler, and PIL trainer.
- `core/dpmac_trainer.py`: fixed-privacy baseline inspired by DPMAC.
- `core/i2c_trainer.py`: deterministic communication baseline inspired by I2C.
- `core/maddpg_trainer.py`: no-communication decentralized baseline inspired by MADDPG.
- `benchmarks/mpe_suite.py`: lightweight PettingZoo MPE runner for `CN`, `CCN`, and `PP`.
- `benchmarks/matrix_games.py`: matrix-game runner for `binary_sum` and `multi_round_sum`.
- `benchmarks/comm_critic.py`: shared centralized-critic, directed sender/receiver, and posterior utilities for the benchmark runners.
- `metrics/privacy.py`: RDP accounting and KL distortion helpers.
- `metrics/constraints.py`: welfare regret and approximate IC/IR summaries.
- `experiments/run_pil_aps.py`: run PIL-APS and save metrics JSON.
- `experiments/run_dpmac.py`: run the fixed-privacy baseline and save metrics JSON.
- `experiments/run_i2c.py`: run the deterministic communication baseline.
- `experiments/run_maddpg.py`: run the no-communication baseline.
- `experiments/run_synthetic_benchmark.py`: run one synthetic baseline from a single CLI entrypoint.
- `experiments/compare_pil_vs_dpmac.py`: run both methods, save summaries, and generate plots when `matplotlib` is available.
- `experiments/compare_baselines.py`: compare PIL against `DPMAC`, `I2C`, and `MADDPG` in one pass.
- `experiments/run_mpe_benchmark.py`: run one MPE scenario with `PIL`, `DPMAC`, `I2C`, `TarMAC`, or `MADDPG`.
- `experiments/compare_mpe_suite.py`: compare multiple baselines across the MPE suite.
- `experiments/run_matrix_game.py`: run one matrix-game experiment.
- `experiments/compare_matrix_games.py`: compare multiple baselines across the matrix-game suite.

## Quick Start

Run PIL:

```bash
python experiments/run_pil_aps.py --num_blocks 20 --inner_updates 20
```

Run the fixed-privacy baseline:

```bash
python experiments/run_dpmac.py --num_blocks 20 --inner_updates 20
```

Run the I2C-style baseline:

```bash
python experiments/run_i2c.py --num_blocks 20 --inner_updates 20
```

Run the MADDPG-style baseline:

```bash
python experiments/run_maddpg.py --num_blocks 20 --inner_updates 20
```

Run the original 2-way comparison:

```bash
python experiments/compare_pil_vs_dpmac.py --num_blocks 20 --inner_updates 20 --seeds 7
```

Run the multi-baseline comparison:

```bash
python experiments/compare_baselines.py --num_blocks 20 --inner_updates 20 --seeds 7 --baselines pil,dpmac,i2c,maddpg
```

Run one synthetic experiment from a single entrypoint:

```bash
python experiments/run_synthetic_benchmark.py --algorithm pil --num_blocks 20 --inner_updates 20
```

Run the MPE suite on one scenario:

```bash
python experiments/run_mpe_benchmark.py --scenario cn --algorithm tarmac --episodes 100 --eval_interval 10 --eval_episodes 8
```

Compare multiple MPE baselines:

```bash
python experiments/compare_mpe_suite.py --scenarios cn,ccn,pp --algorithms pil,dpmac,i2c,tarmac,maddpg --seeds 7,11,23
```

Run a matrix game:

```bash
python experiments/run_matrix_game.py --game binary_sum --algorithm pil --num_blocks 25 --inner_updates 20
```

Compare matrix-game baselines:

```bash
python experiments/compare_matrix_games.py --games binary_sum,multi_round_sum --algorithms pil,dpmac,i2c,tarmac,maddpg --seeds 7,11,23
```

Plot per-model convergence for Exp 1:

```bash
python experiments/plot_exp1_convergence.py
```

Outputs are written into `experiments/*.json`, and comparison plots are written into `plots/`.

## Notes

- The `MPE` and `matrix-game` benchmarks now share a centralized-critic backbone rather than actor-only REINFORCE, so the `MADDPG` baseline is no longer just "local policy with zero messages".
- `DPMAC` is no longer treated as `I2C + Gaussian noise`: it uses directed sender-to-receiver messages, learned receiver-side aggregation, stochastic signaling, norm clipping, and fixed privacy noise.
- `I2C` keeps deterministic communication with simple message averaging and no privacy noise.
- `TarMAC` keeps attention-weighted message aggregation and no privacy noise.
- `PIL` now builds on the directed private communication stack, adds a privacy-aware posterior/planner auxiliary signal, and remains the only adaptive-privacy method in this repository.
- The synthetic `PIL` trainer now uses posterior-aware action mixing: when posterior uncertainty or privacy noise is high, the policy leans more on the planner contract instead of fully trusting the decentralized actor.
- `PIL` result files now expose both `last` and `best` checkpoints; for adaptive-privacy runs, `final` is the selected best checkpoint rather than blindly using the last block.
- The adaptive scheduler is stabilized to avoid overspending privacy budget early and collapsing into very noisy final blocks.
- The MPE experiments currently use `pettingzoo.mpe`; newer PettingZoo releases deprecate these tasks in favor of `mpe2`, but this repo stays with the locally installed package for quick reproduction.
- The MPE benchmark now logs privacy-oriented metrics too: `privacy.epsilon`, `privacy.total_rho_spent`, `empirical_leakage`, and `kl_distortion`.
- MPE comparison runs generate reward and privacy plots such as `plots/mpe_cn_reward_compare.png`, `plots/mpe_cn_epsilon_compare.png`, `plots/mpe_cn_leakage_compare.png`, and `plots/mpe_cn_kl_compare.png`.
- The matrix-game suite is continuous-output and communication-centric; it is meant for fast ablations of privacy scheduling, communication structure, and message quality.
- `QMIX` is not included in this prototype because the current environment is continuous and communication-centric; a faithful QMIX baseline would require a discrete action/value-mixing setup rather than a thin wrapper here.
