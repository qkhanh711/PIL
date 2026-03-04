# Private Incentive Learning with Adaptive Privacy Scheduling


## Abstract

Privacy fundamentally reshapes incentive design in decentralized multi-agent systems, as noise introduced for differential privacy corrupts type estimation, yielding distorted contracts and reduced social welfare.

We study strategic multi-agent systems in which agents possess private types and interact under differential privacy constraints, creating a fundamental welfare–privacy tradeoff.

We propose **Private Incentive Learning with Adaptive Privacy Scheduling (PIL-APS)**, a framework that jointly optimizes privacy allocation, incentive mechanisms, and individual rationality. Our scheme allocates privacy as a finite composable resource characterized via Rényi Differential Privacy (RDP), while quantifying privacy-induced information distortion using the KL divergence between privatized and optimal signaling distributions.

The resulting problem is formulated as a constrained dynamic welfare maximization program that simultaneously enforces:

* Global differential privacy
* Incentive compatibility
* Individual rationality

---

# 1. Introduction

Multi-Agent Reinforcement Learning (MARL) has achieved remarkable success in cooperative and decentralized decision-making problems, including robotic coordination and distributed control. Modern deep MARL frameworks such as actor–critic and value factorization methods enable scalable learning in complex environments.

In many real-world platforms, however, agents are **strategic** and possess **private types** that influence both individual utilities and collective welfare. Effective coordination therefore requires information sharing, yet revealing private information exposes agents to manipulation and privacy risks.

Differential Privacy (DP) provides a principled mechanism for limiting information leakage via calibrated noise injection. Recent work incorporates DP into MARL and distributed learning. However, existing approaches typically treat privacy as a **fixed constraint**, decoupled from mechanism design and welfare optimization.

Injecting privacy noise fundamentally reshapes the incentive landscape:

* Stronger privacy → more noise → distorted type inference
* Distorted inference → suboptimal contracts
* Dynamic composition → cumulative distortion over time

Thus, privacy allocation directly affects the feasibility of:

* Incentive Compatibility (IC)
* Individual Rationality (IR)

We treat privacy as a **finite, dynamically allocatable resource** and jointly optimize learning, incentives, and privacy scheduling.

---

# 2. Related Work

### Multi-Agent Learning

* Cooperative MARL
* Actor–critic methods
* Value factorization (e.g., QMIX)
* Communication-aware MARL

### Differential Privacy

* Differential privacy in MARL
* Federated learning with DP
* Adaptive clipping
* Rényi Differential Privacy

### Incentive Mechanisms

* Contract theory in federated learning
* Incentive-aware privacy design
* Strategic learning under information constraints

---

# 3. System Model

## 3.1 Bayesian Decentralized Markov Game

We consider a Bayesian decentralized Markov game with $N$ strategic agents.

Each agent $i$ possesses a private type:

$\theta_i \in \Theta_i$
drawn from prior $\mathcal{P}(\theta)$ and fixed throughout the episode.

The MDP is defined as:

$\langle \mathcal{S}, {\mathcal{A}_i}_{i=1}^N, P, r, \gamma \rangle$


At time $t$:

* State: $s_t \in \mathcal{S}$
* Observation: $o_{i,t} \sim \mathcal{O}_i(s_t, \theta_i)$
* Action: $a_{i,t} \sim \pi_i(\cdot \mid o_{i,t})$
* Transition: $s_{t+1} \sim P(\cdot \mid s_t, a_t)$
* Base reward: $r_t = r(s_t, a_t)$

---

## 3.2 Differentially Private Signaling with Adaptive Scheduling

Each agent encodes private information:

$
x_{i,t} = f_\phi(\theta_i, o_{i,t})
$

with bounded $\ell_2$-sensitivity $\Delta_t$.

### Adaptive Privacy Scheduler

Noise variance is parameterized as:

$
\sigma_t^2 = \text{softplus}(g_\eta(s_t, \hat{\theta}*{t-1})) + \sigma*{\min}^2
$

Privatized message:

$
m_{i,t} = x_{i,t} + u_{i,t}, \quad u_{i,t} \sim \mathcal{N}(0, \sigma_t^2 I)
$

### Global RDP Composition

Per-step RDP:

$
\rho_t = \frac{\alpha \Delta_t^2}{2\sigma_t^2}
$

Total privacy cost:

$
\rho_{\text{total}} = \sum_{t=1}^{T} \rho_t
$

Converted to ((\epsilon, \delta))-DP:

$
\epsilon_{\text{total}} =
\rho_{\text{total}} +
\frac{\log(1/\delta)}{\alpha - 1}
$

---

## 3.3 Type Estimation and Contract Mechanism

Type estimator:

$
\hat{\theta}*t = h*\psi(m_{1,t}, \dots, m_{N,t})
$

Transfer rule:

$
c_t = C_\omega(\hat{\theta}_t, s_t)
$

Modified reward:

$
\tilde{r}*{i,t} = r(s_t, a_t) + c*{i,t}
$

---

## 3.4 Incentive Compatibility (IC)

Truthful strategy maximizes expected utility:

$
\mathbb{E}_{u_t}
\left[
\sum_t \gamma^t \tilde{r}*i(\theta_i)
\right]
\ge
\mathbb{E}*{u_t}
\left[
\sum_t \gamma^t \tilde{r}_i(\theta'_i)
\right]
$

for all $\theta'_i \neq \theta_i$.

---

## 3.5 Individual Rationality (IR)

$
\mathbb{E}
\left[
\sum_t \gamma^t \tilde{r}_{i,t}
\right]
\ge 0
$

---

## 3.6 Information-Theoretic Privacy Distortion

Let:

* Optimal non-private distribution:
  $p^*(x) = \mathcal{N}(\mu^*, \Sigma^*)$

* Privatized distribution:
  $p_\sigma(x) = \mathcal{N}(\mu, \Sigma + \sigma_t^2 I)$

Distortion:

$
D_{\text{KL}} = D_{\text{KL}}(p_\sigma | p^*)
$

Closed-form expression exists for Gaussian distributions.

---

# 4. Dynamic Mechanism Design under Privacy Constraints

## 4.1 Planner’s Problem

$
\max_{\phi,\psi,\omega,\eta}
\mathbb{E}
\left[
\sum_{t=1}^T
\gamma^t
\sum_{i=1}^N
\tilde{r}_{i,t}
\right]
\lambda_{KL}
\sum_{t=1}^T
D_{KL,t}
$

Subject to:

$
\sum_{t=1}^T \rho_t \le \rho_{\text{budget}}
$

$
\text{IC constraints}
$

$
\text{IR constraints}
$

---

## 4.2 Lagrangian Formulation

Dual variables:

* $\lambda_\rho$ (privacy shadow price)
* $\lambda_{IC}$
* $\lambda_{IR}$

Lagrangian:

$
\mathcal{L} =- \mathbb{E}\left[\sum_t \gamma^t \sum_i \tilde{r}_{i,t}\right]-\lambda_{KL} \sum_t D_{KL,t}+\lambda_\rho\left(\sum_t \rho_t - \rho_{\text{budget}}\right)+\lambda_{IC} \sum_i \Delta_i^{IC}+\lambda_{IR} \sum_i \Delta_i^{IR}$

---

## 4.3 Primal–Dual Learning

Saddle-point optimization:

$
\min_{\phi,\psi,\omega,\eta}
\max_{\lambda_\rho,\lambda_{IC},\lambda_{IR} \ge 0}
\mathcal{L}
$

Dual updates:

$
\lambda_\rho \leftarrow
\left[
\lambda_\rho + \beta(\rho_{\text{total}} - \rho_{\text{budget}})
\right]^+
$

$
\lambda_{IC} \leftarrow
\left[
\lambda_{IC} + \beta \Delta_{IC}
\right]^+
$

$
\lambda_{IR} \leftarrow
\left[
\lambda_{IR} + \beta \Delta_{IR}
\right]^+
$

### Economic Interpretation

* $\lambda_\rho$: shadow price of privacy
* $\lambda_{IC}$: incentive violation penalty
* $\lambda_{IR}$: participation penalty

---

# 5. Algorithm

## Algorithm 1: PIL-APS

**Input:** Initialize $\phi, \psi, \omega, \eta$, dual variables

For each iteration:

1. Sample trajectory
2. Encode private signals
3. Schedule noise
4. Generate privatized messages
5. Estimate types
6. Compute transfers
7. Update rewards
8. Compute RDP
9. Update primal parameters
10. Update dual variables

---

# 6. Experiments

(To be completed)

* Welfare vs Privacy Budget
* Privacy Scheduler Behavior
* IC/IR Violation Trends
* Comparison with Fixed-DP baseline (e.g., DPMAC)

---

# 7. Conclusion

We propose a unified framework that:

* Treats privacy as a finite dynamic resource
* Quantifies privacy distortion via KL divergence
* Jointly enforces IC and IR
* Solves via primal–dual learning

The framework reveals a fundamental welfare–privacy–incentive tradeoff and provides adaptive privacy allocation across time.

---
# PIL
# PIL
