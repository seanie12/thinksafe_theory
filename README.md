# Rebuttal response on theory

**Response to the theoretical concern.**
We contribute three results that together justify the design of filtered SFT with self-generation and refusal steering.
**(1)** Self-generation is KL-optimal for reasoning preservation: the student's own safety-filtered distribution is the unique safe distribution minimizing forward KL to the student (Lemma 1).
**(2)** Filtering is the least disruptive way to achieve perfect safety: it is the smallest $\chi^2$ step from the source that places all mass on safe outputs (Proposition 1, specialized from Russo, 2026).
**(3)** Refusal steering reduces generation cost without distorting the target: under a structural assumption (a label-only tilt), refusal steering preserves the accepted target *exactly* while providing up to an $\omega$-fold speedup, with the largest gains on the hardest prompts (Corollary 1).

**Setup and notation.**
Fix a prompt $x$ and let $\mathcal{Y}$ denote the (finite) response space. Two models play distinct roles:

- the **student** $p_{\mathrm{ref}}(\cdot\mid x)$: the frozen model being safety-tuned;
- the **source** $\pi(\cdot\mid x)$: the model that generates candidate training responses (may equal $p_{\mathrm{ref}}$ for self-generation, or differ, e.g. a teacher).

A binary **safety filter** $\varphi(x,y)\in\{0,1\}$ classifies each response as safe or unsafe. *Safety filtering* a model means conditioning it on safe outputs. For the source, define its acceptance rate and **safety-filtered distribution**:

$$\alpha_{\pi}(x) := \mathbb{P}_{y\sim \pi(\cdot\mid x)}[\varphi(x,y)=1], \qquad \pi^+(y\mid x) := \frac{\pi(y\mid x)\,\varphi(x,y)}{\alpha_{\pi}(x)}.$$

For the student, define the same quantities:

$$\alpha_{\mathrm{ref}}(x) := \mathbb{P}_{y\sim p_{\mathrm{ref}}(\cdot\mid x)}[\varphi(x,y)=1], \qquad p_{\mathrm{ref}}^+(y\mid x) := \frac{p_{\mathrm{ref}}(y\mid x)\,\varphi(x,y)}{\alpha_{\mathrm{ref}}(x)},$$

whenever $\alpha_{\mathrm{ref}}(x)>0$.

**Reasoning preservation.**
The KL divergence $D_{\mathrm{KL}}(\pi^+(\cdot\mid x)\\,\Vert \\,p_{\mathrm{ref}}(\cdot\mid x))$ is a proxy for distributional discrepancy between the training target and the frozen student: for any bounded score $s(y)\in[0,1]$, Pinsker's inequality gives

$$\bigl|\mathbb{E}_{y\sim \pi^+}[s(y)] - \mathbb{E}_{y\sim p_{\mathrm{ref}}}[s(y)]\bigr| \le \sqrt{\tfrac{1}{2}\,D_{\mathrm{KL}}(\pi^+(\cdot\mid x)\\,\Vert \\,p_{\mathrm{ref}}(\cdot\mid x))},$$

so smaller KL means the safety-filtered source is less likely to shift the student's native reasoning.

---

### 1. Self-generation minimizes distributional discrepancy.

**Lemma 1** (Student-relative safe projection). Assume $\alpha_{\mathrm{ref}}(x)>0$. For any distribution $r(\cdot\mid x)$ supported on $\{y\in\mathcal{Y}:\varphi(x,y)=1\}$,

$$D_{\mathrm{KL}}(r(\cdot\mid x)\\,\Vert \\,p_{\mathrm{ref}}(\cdot\mid x)) = -\log \alpha_{\mathrm{ref}}(x) + D_{\mathrm{KL}}(r(\cdot\mid x)\\,\Vert \\,p_{\mathrm{ref}}^+(\cdot\mid x)).$$

Since the first term is independent of $r$ and $D_{\mathrm{KL}}(r(\cdot\mid x)\\,\Vert \\,p_{\mathrm{ref}}^+(\cdot\mid x))\ge 0$ with equality iff $r=p_{\mathrm{ref}}^+(\cdot\mid x)$, the unique safe distribution minimizing $D_{\mathrm{KL}}(r(\cdot\mid x)\\,\Vert \\,p_{\mathrm{ref}}(\cdot\mid x))$ is $r^*=p_{\mathrm{ref}}^+(\cdot\mid x)$.

Applying Lemma 1 to $r=\pi^+(\cdot\mid x)$ yields:

$$D_{\mathrm{KL}}(\pi^+(\cdot\mid x)\\,\Vert \\,p_{\mathrm{ref}}(\cdot\mid x)) = \underbrace{-\log \alpha_{\mathrm{ref}}(x)}_{\text{unavoidable safe-filtering cost}} + \underbrace{D_{\mathrm{KL}}(\pi^+(\cdot\mid x)\\,\Vert \\,p_{\mathrm{ref}}^+(\cdot\mid x))}_{\text{extra discrepancy from using source } \pi \text{ instead of } p_{\mathrm{ref}}}.$$

When the source is the student itself ($\pi=p_{\mathrm{ref}}$), $\pi^+=p_{\mathrm{ref}}^+$ and the mismatch term vanishes: $D_{\mathrm{KL}}(\pi^+\|p_{\mathrm{ref}})=-\log\alpha_{\mathrm{ref}}$, the smallest KL attainable by *any* safe policy. A teacher source ($\pi=p_T$) incurs an additional cross-model gap $D_{\mathrm{KL}}(p_T^+\|p_{\mathrm{ref}}^+)\ge 0$.

---

### 2. Filtering is the least disruptive way to achieve safety.

**Proposition 1** (Source-centered policy improvement; one-step specialization of Russo, 2026). Assume $\alpha_{\pi}(x)>0$. The accepted conditional $\pi^+(\cdot\mid x)$ is the unique optimizer of

$$\max_{r\in\Delta(\mathcal{Y})}\;\mathbb{E}_{y\sim r}[\varphi(x,y)] \qquad\text{subject to}\qquad \chi^2(r \\,\Vert \\, \pi(\cdot\mid x)) \le \frac{1-\alpha_{\pi}(x)}{\alpha_{\pi}(x)}.$$

*That is, filtering is the smallest $\chi^2$-ball step from the source that achieves perfect safety reward.*

For any source $\pi$, the safety-filtered distribution $\pi^+$ is the most conservative modification of $\pi$ that places all mass on safe outputs. This holds regardless of whether the source is the student, a teacher, or a steered model.

---

### 3. Refusal steering reduces generation cost and eliminates gradient imbalance across harmful prompts.

**Assumption 1** (Refusal tilt). For a harmful prompt $x_h$, there exists $\omega(x_h)> 1$ such that the refusal instruction reweights safe outputs by $\omega(x_h)$ while leaving unsafe outputs unchanged:

$$\begin{equation*}p_{\mathrm{ref}}(y\mid I_{\mathrm{refusal}},x_h) \propto \begin{cases} \omega(x_h)\cdot p_{\mathrm{ref}}(y\mid x_h) & \text{if } \varphi(x_h,y)=1, \\\\ p_{\mathrm{ref}}(y\mid x_h) & \text{if } \varphi(x_h,y)=0. \end{cases}
\end{equation*}$$

*Relative probabilities within the safe set and within the unsafe set are preserved; only the odds between the two groups change.*

**Corollary 1** (Refusal steering as cost-reducing preconditioning). Let $\pi_I(\cdot\mid x_h):=p_{\mathrm{ref}}(\cdot\mid I_{\mathrm{refusal}},x_h)$ with acceptance rate $\alpha_I(x_h)$. Under Assumption 1,

$$\pi_I^+(\cdot\mid x_h)=p_{\mathrm{ref}}^+(\cdot\mid x_h), \qquad \alpha_I(x_h)=\frac{\omega(x_h)\,\alpha_{\mathrm{ref}}(x_h)}{1+\bigl(\omega(x_h)-1\bigr)\,\alpha_{\mathrm{ref}}(x_h)} \ge \alpha_{\mathrm{ref}}(x_h),$$

with strict inequality when $\alpha_{\mathrm{ref}}(x_h)<1$. Hence the accepted target is unchanged while the acceptance rate increases (strictly so whenever $0<\alpha_{\mathrm{ref}}(x_h)<1$). Collecting $m$ accepted traces requires $\alpha_I(x_h)/\alpha_{\mathrm{ref}}(x_h)\ge 1$ times fewer expected generations with steering; for small $\alpha_{\mathrm{ref}}(x_h)$ this speedup approaches $\omega(x_h)$.

Since $\pi_I^+=p_{\mathrm{ref}}^+$ exactly, refusal steering achieves the same KL minimum as benign self-generation ($\pi=p_{\mathrm{ref}}$): $D_{\mathrm{KL}}(\pi_I^+\|p_{\mathrm{ref}})=-\log\alpha_{\mathrm{ref}}(x_h)$. Moreover, prompts with lower acceptance rates receive a *larger* multiplicative boost: for any $0<a\le b\le 1$ and common $\omega>1$, $f_\omega(a)/f_\omega(b) > a/b$ where $f_\omega(a):=\omega a/\bigl(1+(\omega-1)a\bigr)$. Thus steering can reduce the *multiplicative* acceptance-rate imbalance across harmful prompts, especially when $\omega$ is large enough to raise all rates above a common threshold.

**References**

D. Russo. Success Conditioning as Policy Improvement: The Optimization Problem Solved by Imitating Success. arXiv preprint arXiv:2601.18175, 2026.
