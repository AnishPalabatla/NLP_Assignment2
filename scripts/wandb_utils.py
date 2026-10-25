from __future__ import annotations

import atexit
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb


def start_wandb_run(
    stage: str,
    config: dict[str, Any],
    run_dir: str | Path,
):
    run_id = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "NLP_Assignment2"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=f"{stage}-{run_id}",
        group=os.environ.get("WANDB_RUN_GROUP") or run_id,
        job_type=stage,
        config=config,
        dir=str(run_dir),
    )
    atexit.register(run.finish)
    return run


def log_wandb_metrics(
    run,
    metrics: dict[str, Any],
) -> None:
    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool))
    }
    if scalar_metrics:
        run.log(scalar_metrics)
