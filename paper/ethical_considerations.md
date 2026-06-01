# Ethical Considerations

## §7: Ethical Considerations & Broader Impacts

### 7.1 Privacy as a Design Principle, Not an Afterthought

UrbanFL makes privacy preservation its **central innovation**, not a post-hoc addition. The GeoPrivacyFL architecture provides formal (ε, δ)-DP guarantees at every layer of the system. We argue this is ethically necessary for any system that processes:

- **Emergency call origins** (reveals home addresses of vulnerable people)
- **Social media geolocations** (reveals movement patterns)
- **IoT sensor telemetry** (reveals neighbourhood-level activity patterns)

### 7.2 Sensitivity-Discriminative Privacy

GeoPrivacyFL's Layer 1 models the insight that not all locations deserve equal privacy protection:

| Location Type | Sensitivity | Noise Level | Rationale |
|---|---|---|---|
| Public infrastructure | 0.1 | Low | Bridges, public buildings — already public knowledge |
| Commercial traffic | 0.5 | Medium | Commercial vehicle GPS — moderate commercial sensitivity |
| Residential zones | 0.9 | High | Home addresses — high individual privacy value |
| Emergency call origin | 1.0 | Maximum | Caller potentially in mortal danger — maximum protection |

This differential treatment is both ethically grounded (more noise where privacy matters most) and practically beneficial (less noise where it doesn't improve downstream utility).

### 7.3 Dual-Use Concerns

**Surveillance risk:** Any city monitoring system could theoretically be repurposed for surveillance. We mitigate this through:
1. **Formal privacy guarantees** that are mathematically verifiable
2. **Verifiable commitments** (Layer 5) allowing external audit
3. **Hard budget enforcement** preventing unlimited data extraction
4. **Open-source implementation** enabling public scrutiny

**Bias in training data:** CrisisNLP and social media datasets underrepresent communities with lower digital access. We acknowledge this limitation and recommend:
- Supplementing social media with IoT and emergency call data (which UrbanFL already does)
- Calibrating the credibility scorer to avoid penalizing low-digital-access neighbourhoods

### 7.4 Deployment Considerations

| Consideration | UrbanFL Approach |
|---|---|
| Data sovereignty | Federated learning — data never leaves silos |
| Consent | IoT/traffic data used in aggregate; emergency calls are public safety data |
| Transparency | All algorithms open-source; verifiable privacy commitments |
| Accountability | Privacy budget certificates provide auditable guarantees |
| Equity | Sensitivity-discriminative noise ensures residential areas get MORE protection |

### 7.5 Responsible AI Statement

We strongly recommend that any deployment of UrbanFL or similar systems:
1. Undergoes independent privacy audit before operational deployment
2. Includes human-in-the-loop decision making for all resource allocation
3. Publishes annual privacy budget reports to the public
4. Maintains an external ethics advisory board including affected community representatives
5. Never be used for law enforcement or immigration enforcement purposes

### 7.6 Environmental Impact

- **Model sizes:** Silo 1 (NLP): ~240MB, Silo 2 (Vision): ~48MB, Silo 3 (Routing): ~5MB
- **Training compute:** Estimated 24 GPU-hours on A100 for full convergence (150 rounds)
- **Inference:** Real-time on consumer GPUs (RTX 3070 or equivalent)
- **Carbon footprint:** Approximately 12 kg CO₂e for full training run (comparable to a short car trip)
