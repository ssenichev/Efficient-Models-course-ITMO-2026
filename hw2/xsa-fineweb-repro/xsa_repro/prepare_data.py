"""Bounded streaming FineWeb preparation; synthetic mode requires no network."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import unicodedata

import numpy as np

from .config import load_config
from .data import verify_manifest
from .utils import canonical_hash, file_hash, utc_now, versions, write_json


def serial_parquet_reader(stream):
    """Keep HF shard/shuffle machinery, disable threaded partial Parquet scans.

    datasets 3.x/4.x ArrowExamplesIterable captures its builder's table generator.
    Replace that per-stream callback, without mutating the installed library.
    Explicitly reject incompatible builders instead of silently changing data.
    """
    import builtins
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets.iterable_dataset import ArrowExamplesIterable
    from datasets.packaged_modules.parquet.parquet import Parquet
    iterable = getattr(stream, "_ex_iterable", None)
    if not isinstance(iterable, ArrowExamplesIterable):
        raise RuntimeError("Unsupported datasets streaming layout; expected ArrowExamplesIterable")
    original = iterable.generate_tables_fn
    builder = getattr(original, "__self__", None)
    if not isinstance(builder, Parquet) or builder.config.filters is not None:
        raise RuntimeError("Serial reader requires an unfiltered HF Parquet builder")
    # HF substitutes xopen in the builder module, retaining auth/storage options.
    file_open = original.__func__.__globals__.get("open", builtins.open)

    def tables(files):
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            with file_open(file, "rb") as source:
                reader = pq.ParquetFile(source)
                try:
                    if reader.metadata.num_row_groups == 0:
                        continue
                    batch_size = builder.config.batch_size or reader.metadata.row_group(0).num_rows
                    for batch_idx, batch in enumerate(reader.iter_batches(
                            batch_size=batch_size, columns=builder.config.columns, use_threads=False)):
                        yield f"{file_idx}_{batch_idx}", builder._cast_table(pa.Table.from_batches([batch]))
                finally:
                    reader.close()

    iterable.generate_tables_fn = tables
    return stream


def document_hash(text):
    normalized = unicodedata.normalize("NFC", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def document_split(digest, salt, val_fraction):
    value = int.from_bytes(hashlib.sha256((salt + ":" + digest).encode()).digest(), "big")
    return "val" if value < int(val_fraction * (1 << 256)) else "train"


class SyntheticTokenizer:
    """ASCII byte tokenizer for software tests, never for FineWeb experiments."""
    name, n_vocab = "synthetic-byte-v1", 256

    def encode_ordinary(self, text):
        return list(text.encode("ascii"))


def synthetic_documents():
    for i in range(100000):
        yield {"id": f"synthetic-{i}", "text": f"Document {i}: the small model predicts the next token. " * 3}


def tokenizer_metadata(tokenizer, eos):
    if isinstance(tokenizer, SyntheticTokenizer):
        fingerprint = canonical_hash({"algorithm": "ascii-bytes", "eot": eos})
    else:
        # Exact BPE vocabulary/ranks and regex, not just a mutable encoding name.
        ranks = sorted((key.hex(), value) for key, value in tokenizer._mergeable_ranks.items())
        fingerprint = canonical_hash({"ranks": ranks, "special": tokenizer._special_tokens, "regex": tokenizer._pat_str})
    return {"name": tokenizer.name, "vocab_size": tokenizer.n_vocab,
            "sha256": fingerprint, "eot_id": eos}


def prepare_documents(documents, config, tokenizer, revision, source, preparation_runtime=None):
    d = config["data"]
    output = Path(d["processed_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite corpus at {output}; use a new processed_dir")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    targets = {"train": d["train_tokens_target"], "val": d["val_tokens_target"]}
    counts = {s: {"tokens": 0, "documents": 0} for s in targets}
    seen_count, duplicate_count = 0, 0
    db = sqlite3.connect(stage / "document_audit.sqlite")
    db.execute("CREATE TABLE documents (sha256 TEXT PRIMARY KEY, split TEXT NOT NULL, accepted INTEGER NOT NULL)")
    binary, indices = {}, {}
    try:
        binary = {s: (stage / f"{s}.bin").open("wb") for s in targets}
        indices = {s: (stage / f"{s}.documents.jsonl").open("w", encoding="utf-8") for s in targets}
        for document in documents:
            text = document.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            seen_count += 1
            digest = document_hash(text)
            split = document_split(digest, d["split_salt"], d["val_fraction_documents"])
            accepted = counts[split]["tokens"] < targets[split]
            inserted = db.execute("INSERT OR IGNORE INTO documents VALUES (?, ?, ?)", (digest, split, int(accepted))).rowcount
            if not inserted:
                duplicate_count += 1
                continue
            if seen_count % 1000 == 0:
                db.commit()
            if not accepted:
                continue
            ids = tokenizer.encode_ordinary(text) + [d["eos_id"]]
            if min(ids) < 0 or max(ids) >= config["model"]["vocab_size"]:
                raise ValueError("Tokenizer does not match model vocabulary")
            # Keep original text for tokenization; normalization is only for dedup/split.
            np.asarray(ids, dtype="<u2").tofile(binary[split])
            indices[split].write(json.dumps({"sha256": digest, "id": document.get("id"), "tokens": len(ids),
                                            "start": counts[split]["tokens"]}, ensure_ascii=False) + "\n")
            counts[split]["tokens"] += len(ids)
            counts[split]["documents"] += 1
            if all(counts[s]["tokens"] >= targets[s] for s in targets):
                break
        if any(counts[s]["tokens"] < targets[s] for s in targets):
            raise RuntimeError("Source exhausted before both token targets were reached")
        db.commit()
        for f in [*binary.values(), *indices.values()]:
            f.close()
        db.close()
        for s in counts:
            counts[s].update(sha256=file_hash(stage / f"{s}.bin"), bytes=(stage / f"{s}.bin").stat().st_size,
                             document_index_sha256=file_hash(stage / f"{s}.documents.jsonl"))
        manifest = {"format_version": 1, "source": source, "revision": revision, "created_utc": utc_now(),
                    "data_config": d, "config_sha256": canonical_hash(config), "data_config_sha256": canonical_hash(d),
                    "tokenizer": tokenizer_metadata(tokenizer, d["eos_id"]), "dtype": "uint16-little-endian",
                    "normalization_for_hash_only": "NFC+strip", "split_rule": "sha256(salt:content_sha256) < fraction*2^256",
                    "splits": counts, "documents_scanned": seen_count, "duplicates_skipped": duplicate_count,
                    "document_audit_sha256": file_hash(stage / "document_audit.sqlite"), "versions": versions(),
                    "preparation_runtime": preparation_runtime or {},
                    "packing": "continuous original-text tokens + one EOT after each document; document-boundary overshoot",
                    "synthetic": d["source"] == "synthetic"}
        write_json(stage / "manifest.json", manifest)
        verify_manifest(stage)
        stage.rename(output)
        print(json.dumps({"output": str(output), "splits": counts, "revision": revision}))
        return manifest
    except BaseException:
        for f in [*binary.values(), *indices.values()]:
            f.close()
        db.close()
        shutil.rmtree(stage)
        raise


def prepare(config):
    d = config["data"]
    if Path(d["processed_dir"]).exists():
        raise FileExistsError(f'Refusing to overwrite corpus at {d["processed_dir"]}; use a new processed_dir')
    if d["source"] == "synthetic":
        return prepare_documents(synthetic_documents(), config, SyntheticTokenizer(), "synthetic-v1", "local synthetic ASCII corpus")
    if (d["source"], d["dataset_id"], d["subset"], d["tokenizer"], d["eos_id"], config["model"]["vocab_size"]) != (
            "fineweb", "HuggingFaceFW/fineweb", "sample-10BT", "gpt2", 50256, 50257):
        raise ValueError("FineWeb reproduction requires sample-10BT with GPT-2 BPE")
    from huggingface_hub import HfApi
    from datasets import load_dataset
    import tiktoken
    import pyarrow as pa
    # Arrow's threaded reader over Python file handles can deadlock at native
    # interpreter shutdown (observed with pyarrow 20 on Apple Silicon). A single
    # reader thread avoids this; tokenization, split and stream order are unchanged.
    pa.set_cpu_count(1)
    # Resolve once through the API and always pass the immutable full SHA to datasets.
    revision = HfApi().dataset_info(d["dataset_id"], revision=d["revision"] or "main").sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset API did not return a full commit SHA")
    if d["revision"] and re.fullmatch(r"[0-9a-f]{40}", d["revision"]) and revision != d["revision"]:
        raise ValueError("Dataset revision mismatch")
    stream = load_dataset(d["dataset_id"], name=d["subset"], split="train", streaming=True, revision=revision)
    stream = serial_parquet_reader(stream)
    stream = stream.shuffle(seed=d["shuffle_seed"], buffer_size=d["shuffle_buffer_size"])
    return prepare_documents(stream, config, tiktoken.get_encoding("gpt2"), revision,
                             "https://huggingface.co/datasets/HuggingFaceFW/fineweb",
                             {"arrow_cpu_threads": pa.cpu_count(), "arrow_use_threads": False,
                              "reader": "HF shard order + serial ParquetFile.iter_batches", "pyarrow_version": pa.__version__})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
