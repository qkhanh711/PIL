# Differentially Private Multi-Agent Communication (DPMAC)

## System Setting and Privacy Guarantee

---

# 1. Problem Setting

We consider a cooperative **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** with communication.

The environment is defined as:

[
\mathcal{G} =
\langle
\mathcal{S},
{\mathcal{A}*i}*{i=1}^N,
P,
r,
\gamma
\rangle
]

where:

* ( \mathcal{S} ) is the global state space.
* ( \mathcal{A}_i ) is the action space of agent (i).
* ( P(s' \mid s, a_1, \dots, a_N) ) is the transition kernel.
* ( r(s, a_1, \dots, a_N) ) is the shared reward.
* ( \gamma \in (0,1) ) is the discount factor.

At time step (t):

* The environment state is (s_t).
* Each agent receives a private observation (o_{i,t}).
* Agents communicate before selecting actions.

The objective is to maximize the expected return:

[
J(\pi) =
\mathbb{E}
\left[
\sum_{t=0}^{T}
\gamma^t r(s_t, a_t)
\right].
]

---

# 2. Communication Model

Each agent encodes its observation into a message:

[
x_{i,t} = f_i(o_{i,t}),
]

where ( f_i ) is a message function.

The messages are broadcast to other agents to facilitate coordination.

---

## 2.1 Bounded Message Norm

Assume the message function has bounded ( \ell_2 )-norm:

[
|x_{i,t}|_2 \le C,
]

where (C) is a known constant.

This boundedness is crucial for the privacy analysis.

---

# 3. Differential Privacy Mechanism

To preserve privacy, each agent perturbs its message with Gaussian noise:

[
m_{i,t} = x_{i,t} + \xi_{i,t},
]

where

[
\xi_{i,t} \sim \mathcal{N}(0, \sigma_i^2 I).
]

The perturbed messages (m_{i,t}) are shared among agents.

---

# 4. Privacy Definition

Agent (i)'s communication satisfies ((\epsilon_i, \delta))-Differential Privacy if for any two neighboring observation sequences and any measurable set (S):

[
\Pr(M_i \in S)
\le
e^{\epsilon_i}
\Pr(M_i' \in S)
+
\delta.
]

Privacy is defined over the **entire communication trajectory**, not a single message.

---

# 5. Privacy Guarantee (Theorem 5.1)

Let:

* ( \gamma_1, \gamma_2 \in (0,1) ),
* ( C ) be the ( \ell_2 )-norm bound of message functions,
* ( N ) be the number of agents,
* ( \epsilon_i ) be the privacy budget of agent (i),
* ( \delta > 0 ).

Then the communication of agent (i) satisfies ((\epsilon_i, \delta))-DP if the noise variance is chosen as:

[
\sigma_i^2
==========

\frac{
14 \gamma_2^2 \gamma_1^2 N C^2 \alpha
}{
\beta \epsilon_i
},
]

where:

[
\alpha
======

\frac{
\log(1/\delta)
}{
\epsilon_i (1-\beta)
}
+
1,
]

and ( \beta \in (0,1) ).

Additionally, define

[
\sigma^2 = \frac{\sigma_i^2}{4C^2},
]

which must satisfy:

[
\sigma^2 \ge 0.7.
]

---

# 6. Key Characteristics of DPMAC Privacy

## 6.1 Variance Scaling

The variance scales as:

[
\sigma_i^2 \propto \frac{1}{\epsilon_i}.
]

This differs from the classical Gaussian mechanism, where:

[
\sigma^2 \propto \frac{1}{\epsilon^2}.
]

---

## 6.2 Stability-Based Analysis

The privacy guarantee relies on:

* Contraction properties of the learning dynamics.
* Bounded message norms.
* Stability of the communication system.

Thus, DPMAC does **not** use the classical sensitivity-based Gaussian DP proof.

---

# 7. Learning with Private Communication

Each agent selects actions using a policy:

[
a_{i,t}
\sim
\pi_i(a \mid o_{i,t}, m_t).
]

The training objective remains:

[
\max_\pi
\mathbb{E}
\left[
\sum_t
\gamma^t
r(s_t, a_t)
\right].
]

However, privacy noise increases:

* Information distortion
* Gradient variance
* Coordination difficulty

---

# 8. Privacy–Performance Trade-off

Since

[
\sigma_i^2 \propto \frac{1}{\epsilon_i},
]

smaller ( \epsilon_i ) (stronger privacy) implies:

* Larger noise variance
* Greater signal distortion
* Reduced coordination performance

This yields an inherent privacy–utility trade-off.

---

# 9. Assumptions Summary

DPMAC relies on:

1. Bounded message norm ( |x_{i,t}|_2 \le C )
2. Contraction constants ( \gamma_1, \gamma_2 \in (0,1) )
3. Gaussian noise injection
4. Stability-based privacy analysis

Privacy level is **exogenously specified** and not optimized dynamically.

