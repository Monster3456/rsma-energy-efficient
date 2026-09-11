import sys
import time
from pathlib import Path

aris = Path((Path.home() / ".aris" / "repo").read_text(encoding="utf-8").strip())
sys.path.insert(0, str(aris / "tools"))
import research_wiki as rw  # noqa: E402

WIKI = "research-wiki"

# (arxiv_id, thesis = overlap/delta from the verified novelty dossier, tags)
PAPERS = [
    (
        "2203.03690",
        "P1. Robust design of RSMA under imperfect CSI via CONVENTIONAL robust optimization (worst-case rate bounds), DOWNLINK. "
        "Closest prior work to idea:l2-4-csir-align-audit / idea:l2-1-csir-domain-rand. Delta: not RL; downlink not uplink; "
        "no train/eval distribution mismatch, no domain randomization, no queue/delivery constraint, no EE agent.",
        ["rsma", "imperfect-csi", "robust-optimization", "downlink", "prior-work", "novelty-2026-09"],
    ),
    (
        "2507.06602",
        "P2 (co-closest neighbor). Studies RL generalization across RAN topology/traffic/mobility/channel and EXPLICITLY uses domain "
        "randomization. Delta: mismatch attributed to diverse environments, NOT CSI-error-scale at low SNR; downlink link adaptation; "
        "no RSMA / uplink-EE / queue-delivery constraint; does not diagnose a fixed-eps-vs-sigma_est train/eval gap.",
        ["domain-randomization", "rl-generalization", "ran", "prior-work", "novelty-2026-09"],
    ),
    (
        "2510.25393",
        "P3. Model-free robust beamforming (SAC) in LEO satellite DOWNLINK under imperfect CSIT (AoD/phase errors). Delta: no "
        "SNR-dependent error link, no train/eval distribution-mismatch diagnosis; not RSMA / uplink / EE / constrained.",
        ["robust-rl", "beamforming", "satellite", "imperfect-csi", "prior-work", "novelty-2026-09"],
    ),
    (
        "2302.02302",
        "P4. Robust generalization for a learned wireless CHANNEL-ESTIMATION model. Delta: about the estimator network, not an RL "
        "control policy; no RSMA, no EE, no queue constraint, no train/eval sigma_est mismatch inside a decision-making loop.",
        ["channel-estimation", "robust-generalization", "prior-work", "novelty-2026-09"],
    ),
    (
        "2501.19019",
        "P5 (co-closest neighbor). UPLINK RSMA under imperfect CSI: finds CSIR error dominates SIC error AT LOW SNR and a small common "
        "fraction is optimal (closed-form outage/throughput analysis). Supplies the exact low-SNR CSIR-dominance FACT the L2-4 diagnosis "
        "leans on. Delta: NOT RL; proposes no training fix, no distribution alignment / DR, no constrained-EE agent.",
        ["rsma", "uplink", "imperfect-csi", "low-snr", "prior-work", "novelty-2026-09"],
    ),
    (
        "2506.04291",
        "P6. A Lyapunov drift-plus-penalty drift term tailored for RL (queue stability). Matches our constrained-RL + DPP + queue setting. "
        "Delta: about the drift-term FORM for value stability, not CSI-error distribution mismatch; no RSMA / imperfect-CSI robustness.",
        ["lyapunov", "drift-plus-penalty", "queue-stability", "rl", "prior-work", "novelty-2026-09"],
    ),
    (
        "2412.14297",
        "P7. Distributionally robust policy learning under concept drifts (generic RL theory). Matches the 'distributionally robust policy' "
        "framing of claim C3. Delta: generic RL, not wireless; no CSI / sigma_est, no RSMA, no SNR-dependent error, no sim channel model.",
        ["distributionally-robust", "rl-theory", "concept-drift", "prior-work", "novelty-2026-09"],
    ),
]

ok, failed = [], []
for i, (aid, thesis, tags) in enumerate(PAPERS):
    if i:
        time.sleep(3)  # arXiv API asks for ~3s between requests
    for attempt in range(3):
        try:
            p = rw.ingest_paper(WIKI, arxiv_id=aid, thesis=thesis, tags=tags, update_on_exist=True)
            print(f"ingested {aid} -> {p.name}")
            ok.append(aid)
            break
        except Exception as e:  # transient arXiv reset (WinError 10054) etc.
            if attempt < 2:
                wait = 3 * (attempt + 1)
                print(f"  retry {aid} in {wait}s ({type(e).__name__})")
                time.sleep(wait)
            else:
                print(f"  SKIP {aid}: fetch failed after 3 attempts ({type(e).__name__}) — not fabricating metadata")
                failed.append(aid)

if ok:
    rw.rebuild_query_pack(WIKI)
print(f"PAPER INGEST: {len(ok)}/7 succeeded" + (f"; failed={failed}" if failed else "") + ".")
