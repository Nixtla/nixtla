#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to clean up finetuned models older than a specified duration.
Usage: python scripts/cleanup_finetuned_models.py [--older-than MINUTES] [--dry-run]
"""

import os
import sys
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone

from nixtla.nixtla_client import NixtlaClient


def cleanup_models(older_than_minutes: int = 30, dry_run: bool = False) -> int:
    """
    Clean up finetuned models older than specified duration.

    Args:
        older_than_minutes: Delete models older than this many minutes (default: 30).
        dry_run: If True, only list models without deleting them.

    Returns:
        Number of models deleted.
    """
    # Collect all credential pairs from environment
    credentials = [
        (os.environ.get("NIXTLA_API_KEY_CUSTOM"), os.environ.get("NIXTLA_BASE_URL_CUSTOM")),
        (os.environ.get("NIXTLA_API_KEY"), os.environ.get("NIXTLA_BASE_URL")),
        (os.environ.get("NIXTLA_API_KEY_FOR_SF"), os.environ.get("NIXTLA_BASE_URL")),
    ]

    # Filter out None/empty credentials
    credentials = [(key, url) for key, url in credentials if key and url]

    if not credentials:
        print("[WARNING] No credentials found. Skipping cleanup.")
        return 0

    now = datetime.now(timezone.utc)
    cutoff_time = now - timedelta(minutes=older_than_minutes)

    total_deleted = 0
    total_failed = 0

    for api_key, base_url in credentials:
        try:
            client = NixtlaClient(api_key=api_key, base_url=base_url)
            models = client.finetuned_models()
        except Exception:
            # Silently skip if unable to connect
            continue

        # Filter models older than cutoff time
        old_models = []
        for model in models:
            model_time = model.created_at
            if model_time.tzinfo is None:
                model_time = model_time.replace(tzinfo=timezone.utc)

            if model_time < cutoff_time:
                old_models.append(model)

        if not old_models:
            continue

        if dry_run:
            total_deleted += len(old_models)
            continue

        # Delete old models
        for model in old_models:
            try:
                client.delete_finetuned_model(model.id)
                total_deleted += 1
            except Exception:
                total_failed += 1

    if dry_run:
        print(f"[DRY RUN] Found {total_deleted} model(s) older than {older_than_minutes} minute(s)")
        print("[DRY RUN] Models would be deleted without --dry-run flag")
    else:
        print(f"[OK] Cleaned up {total_deleted} finetuned model(s)")
        if total_failed:
            print(f"[WARNING] {total_failed} model(s) failed to delete")

    return total_deleted


if __name__ == "__main__":
    parser = ArgumentParser(description="Clean up old finetuned models")
    parser.add_argument(
        "--older-than",
        type=int,
        default=30,
        help="Delete models older than this many minutes (default: 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List models without deleting")
    args = parser.parse_args()

    cleanup_models(older_than_minutes=args.older_than, dry_run=args.dry_run)
    sys.exit(0)
