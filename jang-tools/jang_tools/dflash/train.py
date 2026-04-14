"""PyTorch distillation trainer for JANG-DFlash.

Reads safetensors shards produced by ``jang_tools.dflash.distill_data``
and trains a ``JangDFlashDrafter`` against the target's chosen tokens
under the weighted masked CE loss (``dflash_loss``, Eq. 4 of the
DFlash paper).

Designed to run on the RTX 5090 server where PyTorch has CUDA. The
checkpoint is saved as a plain PT state dict; convert to MLX
safetensors via ``jang_tools.dflash.convert_to_mlx``.

Usage:
    python -m jang_tools.dflash.train \\
        --data /data/dflash-distill-v1 \\
        --out /data/dflash-drafter-v1 \\
        --batch 16 --max-steps 2000 --lr 3e-4
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .config import JangDFlashConfig
from .drafter import JangDFlashDrafter, dflash_loss


class DistillDataset(Dataset):
    """Loads one safetensors shard per ``__getitem__``. Each shard
    contains ``h_taps: [num_taps, T, hidden]`` and ``tokens: [T]``.

    The collate step reshapes ``h_taps`` to ``[T, num_taps * hidden]``
    and samples a random block window per shard. Block sampling is
    done in ``collate`` rather than here so the dataset stays
    shard-atomic.
    """

    def __init__(self, root: str, min_tokens: int = 0):
        self.files = sorted(glob.glob(f"{root}/*.safetensors"))
        if not self.files:
            raise FileNotFoundError(f"no .safetensors files found under {root}")
        self.min_tokens = min_tokens

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from safetensors import safe_open  # local — optional dep on 5090

        path = self.files[idx]
        with safe_open(path, framework="pt") as sf:
            h_taps = sf.get_tensor("h_taps").to(torch.bfloat16)  # [K, T, hidden]
            tokens = sf.get_tensor("tokens").long()               # [T]
        # [K, T, hidden] -> [T, K*hidden]
        T = h_taps.shape[1]
        h_taps = h_taps.permute(1, 0, 2).reshape(T, -1).contiguous()
        return h_taps, tokens


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: JangDFlashConfig,
    seed_generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one random block window per shard. Returns:

        h_ctx:    [N, B, tap_dim]   bfloat16 — anchor-aligned tap slices
        block_in: [N, B]             int64 — [anchor_tok, MASK, ..., MASK]
        targets:  [N, B]             int64 — the target's chosen tokens
    """
    B = cfg.block_size
    h_ctx_list: list[torch.Tensor] = []
    in_list: list[torch.Tensor] = []
    tgt_list: list[torch.Tensor] = []
    for h_taps, tokens in batch:
        T = tokens.shape[0]
        if T < B:
            continue
        anchor = int(torch.randint(0, T - B + 1, (1,), generator=seed_generator).item())
        block_tokens = tokens[anchor : anchor + B].clone()
        block_input = block_tokens.clone()
        block_input[1:] = cfg.mask_token_id
        h_ctx_list.append(h_taps[anchor : anchor + B])  # [B, tap_dim]
        in_list.append(block_input)
        tgt_list.append(block_tokens)

    if not h_ctx_list:
        # Empty fallback — upstream loop will just skip this batch.
        return (
            torch.empty(0, B, cfg.tap_dim, dtype=torch.bfloat16),
            torch.empty(0, B, dtype=torch.long),
            torch.empty(0, B, dtype=torch.long),
        )

    return (
        torch.stack(h_ctx_list),
        torch.stack(in_list),
        torch.stack(tgt_list),
    )


def _to_device(
    t: torch.Tensor, device: torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor:
    if dtype is not None:
        return t.to(device=device, dtype=dtype)
    return t.to(device=device)


def train(args: argparse.Namespace) -> None:
    cfg = JangDFlashConfig(
        block_size=args.block_size,
        loss_gamma=args.loss_gamma,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(
            "[train] WARNING: CUDA not available; training on CPU will be "
            "unusably slow for real runs. Intended hardware is the 5090.",
            file=sys.stderr,
        )

    drafter = JangDFlashDrafter(cfg).to(device).to(torch.bfloat16)
    opt = torch.optim.AdamW(
        drafter.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    ds = DistillDataset(args.data, min_tokens=cfg.block_size)
    print(f"[train] {len(ds)} shards in {args.data}", file=sys.stderr)

    # Dedicated generator so shuffle and anchor sampling are
    # reproducible per run.
    seed_generator = torch.Generator().manual_seed(args.seed)

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(b, cfg, seed_generator),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    step = 0
    running_loss = 0.0
    running_count = 0
    t0 = time.time()
    for epoch in range(args.max_epochs):
        for batch in loader:
            h_ctx, block_in, tgt = batch
            if h_ctx.shape[0] == 0:
                continue  # skipped batch (shards too short)

            h_ctx = _to_device(h_ctx, device, torch.bfloat16)
            block_in = _to_device(block_in, device)
            tgt = _to_device(tgt, device)

            logits = drafter(block_in, h_taps=h_ctx)
            loss = dflash_loss(logits, tgt, cfg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.grad_clip)
            opt.step()

            running_loss += loss.item()
            running_count += 1
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                avg = running_loss / max(running_count, 1)
                print(
                    f"[train] epoch {epoch} step {step} "
                    f"loss {loss.item():.4f} avg {avg:.4f} "
                    f"dt {dt:.1f}s",
                    file=sys.stderr,
                )
                running_loss = 0.0
                running_count = 0

            if step % args.save_every == 0:
                ckpt_path = out / f"drafter-step{step}.pt"
                torch.save(drafter.state_dict(), ckpt_path)
                print(f"[train] saved checkpoint {ckpt_path}", file=sys.stderr)

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    final = out / "drafter.pt"
    torch.save(drafter.state_dict(), final)
    print(f"[train] final checkpoint saved to {final}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m jang_tools.dflash.train",
        description="PyTorch distillation trainer for JANG-DFlash.",
    )
    p.add_argument("--data", required=True, help="Dataset directory (safetensors shards).")
    p.add_argument("--out", required=True, help="Output directory for checkpoints.")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--loss-gamma", type=float, default=7.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
