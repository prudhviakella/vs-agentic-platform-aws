"""
pinecone_load.py — Import vectors from a dump file into Pinecone
================================================================
Students run this to populate their own Pinecone index from the
dump file provided by the instructor.

Usage:
  pip install pinecone tqdm
  export PINECONE_API_KEY=pcsk_...       # student's own API key
  export CLINICAL_TRIALS_INDEX=clinical-trials-index

  # Download the dump file from the shared link, then:
  python3 pinecone_load.py --file pinecone_dump.json

  # Load one namespace only:
  python3 pinecone_load.py --file pinecone_dump.json --namespace clinical-trials-index

  # Dry run (no writes):
  python3 pinecone_load.py --file pinecone_dump.json --dry-run
"""

import os
import json
import time
import argparse
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from typing import Optional

BATCH_UPSERT = 100   # upsert this many vectors per batch

def load(api_key: str, index_name: str, dump_file: str,
         target_namespace: Optional[str] = None, dry_run: bool = False):

    # ── 1. Read dump file ──────────────────────────────────────────────────
    print(f"\nReading dump file: {dump_file}")
    with open(dump_file, "r") as f:
        data = json.load(f)

    source_index = data["index_name"]
    dimension    = data["dimension"]
    namespaces   = data["namespaces"]

    total_vectors = sum(len(v) for v in namespaces.values())
    print(f"Source index  : {source_index}")
    print(f"Dimension     : {dimension}")
    print(f"Namespaces    : {list(namespaces.keys()) or ['(default)']}")
    print(f"Total vectors : {total_vectors:,}")

    if dry_run:
        print("\n⚠️  DRY RUN — no data will be written to Pinecone")

    # ── 2. Connect and ensure index exists ────────────────────────────────
    pc = Pinecone(api_key=api_key)

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        print(f"\nCreating index '{index_name}' (dimension={dimension})...")
        if not dry_run:
            pc.create_index(
                name      = index_name,
                dimension = dimension,
                metric    = "cosine",
                spec      = ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            print("Waiting for index to be ready", end="")
            while True:
                status = pc.describe_index(index_name).status
                if status.get("ready"):
                    print(" ✅")
                    break
                print(".", end="", flush=True)
                time.sleep(3)
    else:
        print(f"\nIndex '{index_name}' already exists ✅")

    if not dry_run:
        index = pc.Index(index_name)

    # ── 3. Upsert each namespace ──────────────────────────────────────────
    ns_list = [target_namespace] if target_namespace else list(namespaces.keys())
    total_upserted = 0

    for ns in ns_list:
        if ns not in namespaces:
            print(f"\n⚠️  Namespace '{ns}' not found in dump file — skipping")
            continue

        vectors  = namespaces[ns]
        ns_label = ns or "(default)"
        print(f"\n── Namespace: {ns_label}  ({len(vectors):,} vectors)")

        if not vectors:
            print("   Empty — skipping")
            continue

        for i in tqdm(range(0, len(vectors), BATCH_UPSERT),
                      desc=f"   Upserting {ns_label}", unit="batch"):
            batch = vectors[i : i + BATCH_UPSERT]

            upsert_records = [
                {
                    "id":       v["id"],
                    "values":   v["values"],
                    "metadata": v["metadata"],
                }
                for v in batch
            ]

            if not dry_run:
                try:
                    upsert_kwargs = {"vectors": upsert_records}
                    if ns:
                        upsert_kwargs["namespace"] = ns
                    index.upsert(**upsert_kwargs)
                    total_upserted += len(batch)
                except Exception as e:
                    print(f"\n   ⚠️  Upsert error on batch {i}: {e}")
                    time.sleep(2)
                    continue
            else:
                total_upserted += len(batch)

        print(f"   {'Would upsert' if dry_run else 'Upserted'} {len(vectors):,} vectors into {ns_label}")

    # ── 4. Verify ─────────────────────────────────────────────────────────
    if not dry_run:
        print(f"\nVerifying...")
        time.sleep(5)   # give Pinecone a moment to index
        stats = index.describe_index_stats()
        print(f"Total vectors in index: {stats.total_vector_count:,}")
        print(f"Namespaces: {list(stats.namespaces.keys())}")

    print(f"\n{'[DRY RUN] Would have upserted' if dry_run else 'Done ✅  Upserted'} "
          f"{total_upserted:,} vectors into '{index_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Pinecone dump into your index")
    parser.add_argument("--file",      required=True,  help="Path to pinecone_dump.json")
    parser.add_argument("--index",     default=os.environ.get("CLINICAL_TRIALS_INDEX", "clinical-trials-index"))
    parser.add_argument("--api-key",   default=os.environ.get("PINECONE_API_KEY"))
    parser.add_argument("--namespace", default=None,   help="Load one namespace only (default: all)")
    parser.add_argument("--dry-run",   action="store_true", help="Preview only — no writes")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Set PINECONE_API_KEY env var or pass --api-key")

    load(
        api_key          = args.api_key,
        index_name       = args.index,
        dump_file        = args.file,
        target_namespace = args.namespace,
        dry_run          = args.dry_run,
    )