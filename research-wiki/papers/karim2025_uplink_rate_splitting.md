---
type: paper
node_id: paper:karim2025_uplink_rate_splitting
title: "Uplink Rate Splitting Multiple Access with Imperfect Channel State Information and Interference Cancellation"
authors: ["Farjam Karim", "Nurul Huda Mahmood", "Arthur S. de Sena", "Deepak Kumar", "Bruno Clerckx", "Matti Latva-aho"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2501.19019"
  doi: null
  s2: null
tags: ["rsma", "uplink", "imperfect-csi", "low-snr", "prior-work", "novelty-2026-09"]
added: 2026-09-03T16:05:04Z
---

# Uplink Rate Splitting Multiple Access with Imperfect Channel State Information and Interference Cancellation

## One-line thesis
P5 (co-closest neighbor). UPLINK RSMA under imperfect CSI: finds CSIR error dominates SIC error AT LOW SNR and a small common fraction is optimal (closed-form outage/throughput analysis). Supplies the exact low-SNR CSIR-dominance FACT the L2-4 diagnosis leans on. Delta: NOT RL; proposes no training fix, no distribution alignment / DR, no constrained-EE agent.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> This article investigates the performance of uplink rate splitting multiple access (RSMA) in a two-user scenario, addressing an under-explored domain compared to its downlink counterpart. With the increasing demand for uplink communication in applications like the Internet-of-Things, it is essential to account for practical imperfections, such as inaccuracies in channel state information at the receiver (CSIR) and limitations in successive interference cancellation (SIC), to provide realistic assessments of system performance. Specifically, we derive closed-form expressions for the outage probability, throughput, and asymptotic outage behavior of uplink users, considering imperfect CSIR and SIC. We validate the accuracy of these derived expressions using Monte Carlo simulations. Our findings reveal that at low transmit power levels, imperfect CSIR significantly affects system performance more severely than SIC imperfections. However, as the transmit power increases, the impact of imperfect CSIR diminishes, while the influence of SIC imperfections becomes more pronounced. Moreover, we highlight the impact of the rate allocation factor on user performance. Finally, our comparison with non-orthogonal multiple access (NOMA) highlights the outage performance trade-offs between RSMA and NOMA. RSMA proves to be more effective in managing imperfect CSIR and enhances performance through strategic message splitting, resulting in more robust communication.

