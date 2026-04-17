"""
pinecone_dump.py — Export all vectors + metadata from Pinecone
================================================================
Dumps every namespace in your index to a JSON file.
Students can then run pinecone_load.py to import into their own index.

Usage:
  pip install pinecone tqdm
  export PINECONE_API_KEY=pcsk_...
  export CLINICAL_TRIALS_INDEX=clinical-trials-index

  python3 pinecone_dump.py                          # dump all namespaces
  python3 pinecone_dump.py --namespace clinical-trials-index  # one namespace only
  python3 pinecone_dump.py --out my_dump.json       # custom output file
"""

import os
import json
import time
import argparse
from tqdm import tqdm
from pinecone import Pinecone
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────
BATCH_FETCH = 100        # fetch this many vectors per batch (Pinecone max: 1000)
BATCH_LIST  = 100        # list this many IDs per page

def dump(api_key: str, index_name: str, output_file: str,
         target_namespace: Optional[str] = None):

    pc    = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # ── 1. Get index stats ────────────────────────────────────────────────
    print(f"\nConnecting to index: {index_name}")
    stats = index.describe_index_stats()
    total = stats.total_vector_count
    namespaces = stats.namespaces or {}

    print(f"Total vectors : {total:,}")
    print(f"Namespaces    : {list(namespaces.keys()) or ['(default)']}")
    print(f"Dimension     : {stats.dimension}")

    # ── 2. Decide which namespaces to dump ────────────────────────────────
    if target_namespace:
        ns_list = [target_namespace]
    else:
        ns_list = list(namespaces.keys()) if namespaces else [""]

    # ── 3. Dump each namespace ────────────────────────────────────────────
    dump_data = {
        "index_name":  index_name,
        "dimension":   stats.dimension,
        "namespaces":  {}
    }

    total_exported = 0

    for ns in ns_list:
        ns_label = ns or "(default)"
        ns_count = namespaces.get(ns, {}).vector_count if ns else total
        print(f"\n── Namespace: {ns_label}  ({ns_count:,} vectors)")

        vectors = []

        # ── Step 1: list all IDs in this namespace ────────────────────────
        print(f"   Listing IDs...")
        all_ids = []
        list_kwargs = {"namespace": ns, "limit": BATCH_LIST} if ns else {"limit": BATCH_LIST}

        try:
            for id_batch in index.list(**list_kwargs):
                # index.list() yields lists of IDs
                if isinstance(id_batch, list):
                    all_ids.extend(id_batch)
                else:
                    all_ids.append(id_batch)
        except Exception as e:
            print(f"   ⚠️  list() failed: {e}")
            print(f"   Trying list_paginated() fallback...")
            try:
                pagination_token = None
                while True:
                    kwargs = {"limit": BATCH_LIST, "namespace": ns} if ns else {"limit": BATCH_LIST}
                    if pagination_token:
                        kwargs["pagination_token"] = pagination_token
                    resp = index.list_paginated(**kwargs)
                    all_ids.extend([v.id for v in resp.vectors])
                    if resp.pagination and resp.pagination.next:
                        pagination_token = resp.pagination.next
                    else:
                        break
            except Exception as e2:
                print(f"   ⚠️  list_paginated() also failed: {e2}")
                print(f"   Skipping namespace {ns_label}")
                continue

        print(f"   Found {len(all_ids):,} IDs")

        # ── Step 2: fetch vectors in batches ──────────────────────────────
        print(f"   Fetching vectors (batch size: {BATCH_FETCH})...")
        fetch_kwargs_base = {"namespace": ns} if ns else {}

        for i in tqdm(range(0, len(all_ids), BATCH_FETCH),
                      desc=f"   {ns_label}", unit="batch"):
            batch_ids = all_ids[i : i + BATCH_FETCH]
            try:
                resp = index.fetch(ids=batch_ids, **fetch_kwargs_base)
                for vec_id, vec_data in resp.vectors.items():
                    vectors.append({
                        "id":       vec_id,
                        "values":   vec_data.values,
                        "metadata": vec_data.metadata or {},
                    })
            except Exception as e:
                print(f"\n   ⚠️  Fetch error on batch {i}-{i+BATCH_FETCH}: {e}")
                time.sleep(2)
                continue

        print(f"   Exported {len(vectors):,} vectors from {ns_label}")
        dump_data["namespaces"][ns] = vectors
        total_exported += len(vectors)

    # ── 4. Write to file ──────────────────────────────────────────────────
    print(f"\nWriting {total_exported:,} total vectors to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(dump_data, f)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Done ✅  {output_file}  ({size_mb:.1f} MB)")
    print(f"\nShare this file with students — they run pinecone_load.py to import it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump Pinecone index to JSON")
    parser.add_argument("--index",     default=os.environ.get("CLINICAL_TRIALS_INDEX", "clinical-trials-index"))
    parser.add_argument("--api-key",   default=os.environ.get("PINECONE_API_KEY"))
    parser.add_argument("--namespace", default=None, help="Dump one namespace only (default: all)")
    parser.add_argument("--out",       default="pinecone_dump.json")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Set PINECONE_API_KEY env var or pass --api-key")

    dump(
        api_key          = args.api_key,
        index_name       = args.index,
        output_file      = args.out,
        target_namespace = args.namespace,
    )