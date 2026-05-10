# MTP Runtime Examples

Small, low-RAM helpers for checking whether a bundle has MTP and whether a
profile is realistic for a target memory size.

## Inspect A Bundle

```sh
python3 jang-tools/examples/mtp/inspect_mtp_bundle.py /path/to/model-or-bundle
```

This reads config and safetensors index metadata only. It does not load model
weights.

## Estimate Hy3 Fit

```sh
python3 jang-tools/examples/mtp/estimate_jangtq_fit.py \
  /Users/eric/models/Tencent/Hy3-preview \
  --profile JANGTQ_K \
  --device-gb 128
```

The estimator is intentionally conservative. It should guide conversion order,
not replace measured runtime memory proofs.
