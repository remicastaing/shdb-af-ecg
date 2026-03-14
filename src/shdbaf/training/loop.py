from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shdbaf.training.metrics import evaluate_partial
from shdbaf.training.profiling import StepWindow, samples_per_sec
from shdbaf.training.system import mem_rss_mb, sys_mem_available_mb, sync_if_needed


@dataclass(frozen=True)
class EpochResult:
    train_loss_epoch: float
    epoch_duration_s: float
    samples_per_sec_epoch: float
    global_step: int


def train_one_epoch(
    *,
    epoch: int,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    labels: list[str],
    device: str,
    batch_size: int,
    steps_per_epoch: int,
    log_every: int,
    eval_every: int,
    eval_batches: int,
    global_step: int,
) -> EpochResult:
    """
    Train 1 epoch:
      - logs train loss + timings + mémoire + samples_per_sec_step (MLflow)
      - live val (val_f1_macro_step, val_acc_step) toutes eval_every steps
      - calcule samples_per_sec_epoch (moyenne des segments loggés)
    """
    model.train()
    epoch_t0 = time.time()

    running_loss = 0.0
    seen = 0

    win = StepWindow()
    epoch_batch_ms_samples: list[float] = []
    prev_end = time.perf_counter()

    for step, (x, y) in enumerate(train_loader):
        t_arrive = time.perf_counter()
        data_wait = t_arrive - prev_end

        # to_device
        t0 = time.perf_counter()
        x = x.to(device)
        y = y.to(device)
        sync_if_needed(device)
        to_dev = time.perf_counter() - t0

        # compute
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        sync_if_needed(device)
        compute = time.perf_counter() - t1

        batch_total = time.perf_counter() - t_arrive
        win.append(data_wait, to_dev, compute, batch_total)

        bs = y.size(0)
        running_loss += float(loss.item()) * bs
        seen += bs

        # logs périodiques
        if step % log_every == 0:
            avg = running_loss / max(seen, 1)
            mean_data_ms, mean_to_dev_ms, mean_compute_ms, mean_batch_ms = win.means_ms()
            sps = samples_per_sec(batch_size, mean_batch_ms)

            print(
                f"[epoch {epoch + 1}] step={step} "
                f"batch_loss={loss.item():.4f} avg_loss={avg:.4f} | "
                f"data={mean_data_ms:.1f}ms to_dev={mean_to_dev_ms:.1f}ms "
                f"compute={mean_compute_ms:.1f}ms batch={mean_batch_ms:.1f}ms "
                f"sps={sps:.0f}/s"
            )

            mlflow.log_metric("train_batch_loss", float(loss.item()), step=global_step)
            mlflow.log_metric("train_avg_loss", float(avg), step=global_step)

            mlflow.log_metric("mem_rss_mb_step", float(mem_rss_mb()), step=global_step)
            mlflow.log_metric("mem_avail_mb_step", float(sys_mem_available_mb()), step=global_step)

            mlflow.log_metric("time_data_wait_ms_step", mean_data_ms, step=global_step)
            mlflow.log_metric("time_to_device_ms_step", mean_to_dev_ms, step=global_step)
            mlflow.log_metric("time_compute_ms_step", mean_compute_ms, step=global_step)
            mlflow.log_metric("time_batch_ms_step", mean_batch_ms, step=global_step)

            mlflow.log_metric("samples_per_sec_step", float(sps), step=global_step)

            epoch_batch_ms_samples.append(mean_batch_ms)
            win.reset()

        # live val partielle
        if eval_every and (global_step > 0) and (global_step % eval_every == 0):
            acc_s, f1_s = evaluate_partial(
                model=model,
                loader=val_loader,
                device=device,
                labels=labels,
                max_batches=eval_batches,
            )
            mlflow.log_metric("val_acc_step", float(acc_s), step=global_step)
            mlflow.log_metric("val_f1_macro_step", float(f1_s), step=global_step)
            print(f"[live val] step={global_step} val_f1_macro_step={f1_s:.4f} val_acc_step={acc_s:.4f}")
            model.train()

        global_step += 1
        prev_end = time.perf_counter()

        if steps_per_epoch and (step + 1) >= steps_per_epoch:
            break

    epoch_duration_s = time.time() - epoch_t0
    train_loss_epoch = running_loss / max(seen, 1)

    if epoch_batch_ms_samples:
        mean_epoch_batch_ms = float(np.mean(epoch_batch_ms_samples))
        sps_epoch = samples_per_sec(batch_size, mean_epoch_batch_ms)
    else:
        sps_epoch = 0.0

    return EpochResult(
        train_loss_epoch=float(train_loss_epoch),
        epoch_duration_s=float(epoch_duration_s),
        samples_per_sec_epoch=float(sps_epoch),
        global_step=global_step,
    )