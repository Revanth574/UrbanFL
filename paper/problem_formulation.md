# Problem Formulation — UrbanFL

## §2.1: Formal Problem Statement

**Given:**
- A metropolitan area $\mathcal{A}$ with area $|\mathcal{A}| = 500$ km², population $P \approx 2$ million
- $K = 3$ data silos, each owning private geospatial datasets $\mathcal{D}_k = \{(x_i, y_i)\}_{i=1}^{n_k}$:
  - Silo 1: Social media, IoT sensors, emergency calls → $\mathcal{D}_1 \in (\mathcal{X}_\text{text} \times \mathbb{R}^{d_\text{sensor}} \times \mathbb{R}^2)$
  - Silo 2: Pre/post-disaster satellite/drone imagery → $\mathcal{D}_2 \in (\mathbb{R}^{H \times W \times C})^2$
  - Silo 3: Road network $G = (V, E)$ with real-time traffic telemetry → $\mathcal{D}_3 \in (\mathbb{R}^{|V|} \times \mathbb{R}^{|E|})$

**Objective:** Learn per-silo models $\{f_k(\cdot; \theta_k)\}_{k=1}^K$ that minimise the weighted empirical risk:

$$\min_{\theta_1, \theta_2, \theta_3} \sum_{k=1}^{3} w_k \cdot \frac{1}{n_k} \sum_{i=1}^{n_k} \ell_k(f_k(x_i; \theta_k), y_i)$$

**Subject to Privacy Constraint (GeoPrivacyFL):**

$$\forall k \in \{1,2,3\}: \Pr[M_k(\mathcal{D}_k) \in S] \leq e^{\varepsilon_k} \cdot \Pr[M_k(\mathcal{D}'_k) \in S] + \delta$$

where $\mathcal{D}_k, \mathcal{D}'_k$ differ in one record and $M_k$ denotes the complete information pathway from silo $k$ to the aggregator.

**And Subject to Response Time Constraint:**

$$\mathbb{E}[T_\text{alert-to-dispatch}] \leq 300 \text{ seconds}$$

## §2.2: Key Challenges

1. **Modality Heterogeneity**: $\nabla_{\theta_1} \ell_1$ (text gradients) have fundamentally different scale and structure from $\nabla_{\theta_2} \ell_2$ (image gradients).

2. **Non-IID Data**: Each silo observes a different projection of the same disaster event. Statistical heterogeneity exceeds standard FL assumptions.

3. **Privacy for Geospatial Data**: Standard DP treats each data point identically. Geospatial data has inherent structural correlations (nearby locations have correlated features) and differential sensitivity (residential vs. public infrastructure).

4. **Real-Time Constraint**: Emergency response demands sub-second routing decisions while maintaining privacy guarantees.


## §2.3: Contributions (Hierarchical)

### Primary Contribution:
**GeoPrivacyFL** — A 5-layer privacy architecture for geospatial federated learning providing provably unprecedented privacy guarantees:

| Layer | Mechanism | Innovation |
|---|---|---|
| 1 | Sensitivity-discriminative geo-indistinguishability | Location-type-aware noise (residential 10× vs. public) |
| 2 | Gradient spatial decomposition | Public-basis projection separates structure from private |
| 3 | Pairwise masking secure aggregation | Corrected for $N=3$ silos (not additive sharing) |
| 4 | Weight-aware RDP composition (Theorem 5) | Subsampling amplification gives $w_k^2$ scaling |
| 5 | Verifiable privacy commitments | Hash-based audit trail with 99.4% detection confidence |

### Supporting Contributions:
- **FedHetero-Robust**: Modality-aware FL with gradient normalisation (eliminates $O(R^2)$ magnitude heterogeneity), adaptive weighting, and Byzantine resilience
- **3-Silo Emergency Response**: Integrated system demonstrating FL advantages over centralised/local approaches on real disaster scenarios

## §2.4: Notation Table

| Symbol | Description |
|---|---|
| $K = 3$ | Number of data silos |
| $\mathcal{D}_k$ | Private dataset of silo $k$ |
| $n_k$ | Dataset size of silo $k$ |
| $w_k$ | Adaptive weight for silo $k$ ($\sum w_k = 1$) |
| $\varepsilon_k$ | Privacy budget for silo $k$ |
| $\delta$ | Privacy failure probability ($10^{-5}$) |
| $\sigma_k$ | Gaussian noise scale for silo $k$ |
| $C$ | Gradient clipping norm |
| $R$ | Gradient magnitude ratio $\max_k \|\nabla_k\| / \min_k \|\nabla_k\|$ |
| $P$ | Public spatial basis matrix ($d \times d_\text{public}$) |
| $\alpha$ | Rényi divergence order |
| $T$ | Number of FL rounds |
| $\theta^{(t)}$ | Global model parameters at round $t$ |
