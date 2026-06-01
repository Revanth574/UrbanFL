# Algorithm Pseudocode — FedHetero-Robust + GeoPrivacyFL

## Algorithm 1: FedHetero-Robust with GeoPrivacyFL

```
Algorithm 1: FedHetero-Robust with GeoPrivacyFL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: K silos with datasets {D_k}, initial model θ⁰, 
       privacy budgets {ε_k}, δ, gradient clip norm C,
       public spatial basis P, warmup rounds W

Output: Trained models {θ_k^T}, privacy certificate

1:  for round t = 1 to T do
2:    ── Local Training (each silo k in parallel) ──
3:    θ_k ← θ^(t-1)                          ▷ Receive global model
4:    for local epoch e = 1 to E_k do
5:      g_k ← ∇L_k(θ_k; B_k)                ▷ Compute local gradient on mini-batch
6:    end for
7:    Δ_k ← θ_k - θ^(t-1)                    ▷ Model update delta
8:
9:    ── GeoPrivacyFL Pipeline (per silo) ──
10:   ε_k ← ε_max(1 - s_k) + ε_min·s_k       ▷ Layer 4a: sensitivity-adaptive ε
11:   σ_k ← CalibrateNoise(ε_k, δ, C)         ▷ Via RDP → (ε,δ)-DP conversion
12:   Δ̃_k ← Clip(Δ_k, C)                     ▷ Per-sample gradient clipping
13:
14:   if t > W then                             ▷ Layer 2: post-warmup decomposition
15:     g_struct ← P·Pᵀ·Δ̃_k                   ▷ Project onto public basis
16:     g_private ← Δ̃_k - g_struct
17:     η ← N(0, σ²_k·C²·I) × √(d_priv/d)     ▷ Reduced noise on private only
18:     Δ̂_k ← g_struct + g_private + η
19:   else
20:     Δ̂_k ← Δ̃_k + N(0, σ²_k·C²·I)          ▷ Full noise during warmup
21:   end if
22:
23:   ── Layer 3: Pairwise Masking ──
24:   for each silo j ≠ k do
25:     r_{k,j} ← PRF(seed_{k,j}, t)           ▷ Deterministic mask from shared seed
26:   end for
27:   Δ̂'_k ← Δ̂_k + Σ_{j≠k} r_{k,j}           ▷ Masked update (masks cancel in sum)
28:
29:   ── Layer 5: Commit ──
30:   h_k ← Hash(‖Δ_k‖, ‖η‖, ε_k, σ_k)        ▷ Verifiable commitment
31:   Send (Δ̂'_k, h_k) to aggregator
32:   BudgetManager.consume(k, ε_k/T)           ▷ Track cumulative budget
33:
34:   ── Aggregation (Server) ──
35:   Δ_total ← Σ_k Δ̂'_k                       ▷ Pairwise masks cancel automatically
36:
37:   ── FedHetero-Robust ──
38:   for each silo k do
39:     n_k ← ‖Δ_total_k‖₂                     ▷ Track gradient norms
40:     Δ_total_k ← Δ_total_k / n_k             ▷ Modality-aware normalisation
41:   end for
42:
43:   w_k ← Softmax(ΔL_k × q_k × decay(τ_k))   ▷ Adaptive weights
44:
45:   ── Byzantine Aggregation ──
46:   Δ_agg ← CoordTrimmedMean({w_k·Δ_total_k}) ▷ Trim extreme coordinates
47:
48:   θ^(t) ← θ^(t-1) + Δ_agg                   ▷ Update global model
49:
50:   ── Concept Drift Check ──
51:   if KL(p^new_k ‖ p^old_k) > threshold then
52:     E_k ← E_k / 3                           ▷ Reduce local epochs, increase comm
53:   end if
54: end for
55:
56: Return {θ^T}, BudgetManager.certificate()
```

## Algorithm 2: Theorem 5 — Weight-Aware RDP Composition

```
Algorithm 2: Weight-Aware RDP Composition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Per-silo RDP ε_k(α), weights w_k, δ
Output: Global (ε_total, δ)-DP guarantee

1:  for each α in {2, 3, 5, 10, 20, 50, 100} do
2:    ε_rdp(α) ← 0
3:    for each silo k do
4:      if w_k < 1/√α then
5:        ε_rdp(α) += min(ε_k(α), 2·w²_k·α·ε_k(α)/(α-1))  ▷ Amplification regime
6:      else
7:        ε_rdp(α) += w_k · ε_k(α)                            ▷ Standard regime
8:      end if
9:    end for
10:   ε_dp(α) ← ε_rdp(α) - (ln δ + ln(α-1))/(α-1) - ln α/(α-1)
11: end for
12: Return min_α ε_dp(α)
```

## Complexity Analysis

| Operation | Time | Space | Per Round |
|---|---|---|---|
| Local training (Silo 1, NLP) | $O(n_1 \cdot d_\text{model})$ | $O(d_\text{model})$ | ~120ms |
| Local training (Silo 2, Vision) | $O(n_2 \cdot H \cdot W \cdot C)$ | $O(d_\text{model})$ | ~480ms |
| Gradient decomposition | $O(d \cdot d_\text{pub})$ | $O(d \cdot d_\text{pub})$ | ~2ms |
| Pairwise masking | $O(d \cdot K)$ | $O(d)$ | ~1ms |
| DP noise addition | $O(d)$ | $O(d)$ | ~1ms |
| Byzantine trimmed mean | $O(K \cdot d \cdot \log K)$ | $O(K \cdot d)$ | ~3ms |
| Response optimization | $O(\|V\| \log \|V\|)$ | $O(\|V\| + \|E\|)$ | ≤500ms |
