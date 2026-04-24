"""Kimi K2.6 expert-pruning pipeline.

Routing-aware expert pruning (REAP-style) with absorb-merge.

Target model: moonshotai/Kimi-K2.6 (KimiK25ForConditionalGeneration,
text backbone = DeepseekV3 variant, 61 layers, 384 routed + 1 shared
expert, top-8 routing, MLA attention).

Pipeline:
  build_calib  -> assembles mixed-domain calibration corpus
  profile      -> streams shards, captures routing freq/weighted_freq/
                  coact/output_energy per layer
  score        -> computes per-expert importance
  prune        -> drops experts + absorb-merges coact neighbors + rewrites
                  router weight rows + re-saves FP8 compressed-tensors shards
  eval         -> multi-domain gate (code/tool/agent/pentest/general/zh)
"""
