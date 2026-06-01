# Related Work — UrbanFL

## §3: Related Work

### 3.1 Federated Learning: Heterogeneity Solutions

**FedAvg** (McMahan et al., 2017) established the foundational FL protocol but assumes IID data across clients. Its convergence degrades substantially under non-IID distributions.

**FedProx** (Li et al., 2020a) adds a proximal term $\frac{\mu}{2}\|w - w^t\|^2$ to the local objective, constraining client drift. However, this uniformly penalizes all divergence without distinguishing modality-specific versus data-specific divergence.

**SCAFFOLD** (Karimireddy et al., 2020) uses control variates $c_i$ to correct client drift, achieving variance reduction. We include SCAFFOLD as a baseline (§5.1) and show that FedHetero-Robust outperforms it specifically on modality-heterogeneous settings (≠ data heterogeneity alone).

**FedNova** (Wang et al., 2020) normalizes updates by the number of local steps, addressing objective inconsistency. FedHetero-Robust extends this idea by normalizing by gradient magnitude per modality.

**CreamFL** (Yu et al., 2023) handles multi-modal FL via knowledge distillation across heterogeneous models. UrbanFL's Semantic Bridge is inspired by this approach but operates in a shared embedding space rather than requiring a public dataset.

**FedMA** (Wang et al., 2020b) matches and averages neural network layers using Bayesian non-parametric methods. While powerful, it requires architecture similarity across clients, which doesn't hold for our 3-silo setting (NLP, Vision, Traffic).

> **Gap filled by UrbanFL**: No prior work addresses FL where clients have fundamentally different architectures AND data modalities (text vs. image vs. graph) while providing per-modality adaptive privacy guarantees.

### 3.2 Privacy-Preserving Federated Learning

**DP-FTRL** (Kairouz et al., 2021) applies differential privacy to FL via follow-the-regularized-leader. It provides strong privacy but uses uniform ε across all clients.

**Federated Learning with Local Differential Privacy** (Liu et al., 2020) applies LDP but at high utility cost (local noise is much larger than central DP).

**Secure Aggregation** (Bonawitz et al., 2017; Bell et al., 2020) uses secret sharing to hide individual updates from the server. We adapt pairwise masking (suitable for $N \geq 2$) rather than Shamir's secret sharing (requires $N \geq 2t+1$, problematic for $N=3$).

**RDP Composition** (Mironov, 2017) provides tighter composition via Rényi divergence. Our Theorem 5 extends this with weight-aware bounds using a subsampling amplification analogy.

> **Gap filled by GeoPrivacyFL**: No existing work combines (1) input-level geo-indistinguishability, (2) gradient spatial decomposition, (3) secure aggregation, (4) per-silo adaptive DP with weight-aware RDP, and (5) verifiable commitments — all optimized for geospatial data.

### 3.3 Geo-Indistinguishability and Location Privacy

**d_χ-privacy** (Chatzikokolakis et al., 2013) extends DP to metric spaces, providing privacy proportional to geographic distance. **Planar Laplace** (Andrés et al., 2013) instantiates this for 2D locations.

**LDGI** (Cunningham et al., 2024) develops location-dependent geo-indistinguishability with sensitivity-discriminative noise. GeoPrivacyFL's Layer 1 builds directly on LDGI, extending it into the FL training pipeline (not just query answering).

**Snap-to-grid analysis** (Alvim et al., 2018): Discretization of continuous location data to road networks introduces bounded privacy overhead $O(\log |V|)$. We formally analyze this in Layer 1.

### 3.4 Disaster Response Systems

**FEMA IPAWS** (2022) provides the US Integrated Public Alert & Warning System. Alert-to-action latency averages 15 minutes. UrbanFL targets < 5 minutes.

**xBD** (Gupta et al., 2019) is the largest building damage assessment dataset from satellite imagery. We use xBD as ground truth for Silo 2 validation.

**CrisisNLP** (Imran et al., 2016) and **HumAID** (Alam et al., 2021) provide crisis-annotated social media data. We use these for Silo 1 NLP model training.

**Bureau of Public Roads (BPR) function** (1964) is the standard travel time model relating flow to capacity. UrbanFL uses BPR in Silo 3.

### 3.5 Byzantine-Resilient Aggregation

**Krum** (Blanchard et al., 2017) selects the update closest to the majority, tolerating up to $f < n/2 - 1$ Byzantine clients. We include Krum as an alternative to trimmed mean.

**Coordinate-wise Trimmed Mean** (Yin et al., 2018) trims extreme values per coordinate, providing statistical optimality against Byzantine attacks.

**FLTrust** (Cao et al., 2021) uses a trusted root dataset at the server. UrbanFL avoids this assumption (no trusted data).

### 3.6 Concurrent & Closely Related Work

| System | FL | Multi-Modal | Geo-Privacy | Byzantine | Verified |
|---|---|---|---|---|---|
| FedAvg | ✓ | ✗ | ✗ | ✗ | ✗ |
| SCAFFOLD | ✓ | ✗ | ✗ | ✗ | ✗ |
| CreamFL | ✓ | ✓ | ✗ | ✗ | ✗ |
| DP-FTRL | ✓ | ✗ | ✗ | ✗ | ✗ |
| Geo-Ind. | ✗ | ✗ | ✓ | ✗ | ✗ |
| **UrbanFL** | **✓** | **✓** | **✓** | **✓** | **✓** |
