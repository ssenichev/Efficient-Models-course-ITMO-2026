import json
import numpy as np
import pytest
import torch
from xsa_repro.data import TokenStream, WindowSampler, verify_manifest
from xsa_repro.prepare_data import prepare, prepare_documents, SyntheticTokenizer, document_hash, document_split
from xsa_repro.config import training_plan, load_config


def test_pipeline_and_shift(smoke_config):
    manifest = prepare(smoke_config)
    d = smoke_config["data"]["processed_dir"]
    assert verify_manifest(d)[0] == manifest
    stream = TokenStream(f"{d}/train.bin", 16, 256)
    x, y = stream.batch([0, (stream.num_windows - 1) * 16])
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert y[-1, -1] == stream.tokens[(stream.num_windows - 1) * 16 + 16]
    with pytest.raises(IndexError):
        stream.batch([len(stream.tokens) - 16])
    with pytest.raises(IndexError):
        stream.batch([-1])
    assert np.all(np.diff(stream.validation_starts()) == 16)
    hashes = [set(json.loads(line)["sha256"] for line in open(f"{d}/{s}.documents.jsonl")) for s in ("train", "val")]
    assert not hashes[0] & hashes[1]
    assert manifest["splits"]["train"]["tokens"] >= 4096
    with pytest.raises(FileExistsError):
        prepare(smoke_config)


def test_dedup_normalization_and_original_tokens(smoke_config):
    c = smoke_config
    c["data"]["train_tokens_target"] = 50
    c["data"]["val_tokens_target"] = 50
    assert document_hash(" e\u0301 ") == document_hash("é")
    # Different whitespace hashes identically, but accepted original text is preserved.
    docs = [{"text": "  unique first document  "}, {"text": "unique first document"}]
    docs += [{"text": f"sample {i} " * 8} for i in range(100)]
    manifest = prepare_documents(iter(docs), c, SyntheticTokenizer(), "test", "local")
    assert manifest["duplicates_skipped"] == 1
    digest = document_hash(docs[0]["text"])
    split = document_split(digest, c["data"]["split_salt"], 0.2)
    tokens = np.fromfile(f'{c["data"]["processed_dir"]}/{split}.bin', dtype="<u2")
    assert list(tokens[:len(docs[0]["text"])]) == list(docs[0]["text"].encode())


def test_sampler_pair_and_restore(smoke_config):
    prepare(smoke_config)
    stream = TokenStream(f'{smoke_config["data"]["processed_dir"]}/train.bin', 16, 256)
    a, b = WindowSampler(stream, 42), WindowSampler(stream, 42)
    for _ in range(5):
        assert np.array_equal(a.sample(7), b.sample(7))
    state = a.state_dict()
    expected = a.sample(9)
    b.load_state_dict(state)
    assert np.array_equal(expected, b.sample(9))


def test_manifest_detects_corruption(smoke_config):
    prepare(smoke_config)
    path = f'{smoke_config["data"]["processed_dir"]}/train.bin'
    with open(path, "r+b") as f:
        f.write(b"\x00\x00")
    with pytest.raises(ValueError, match="binary differs"):
        verify_manifest(smoke_config["data"]["processed_dir"])


def test_main_plan_and_ablation():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    main = load_config(root / "configs/main.yaml")
    assert training_plan(main) == {"grad_accum_steps": 32, "total_steps": 1526, "actual_tokens_budget": 100007936}
    for length in (256, 512, 1024):
        c = load_config(root / f"configs/context_{length}.yaml")
        assert c["model"]["block_size"] == length
        assert c["data"] == main["data"]
        assert training_plan(c)["actual_tokens_budget"] == 30015488
    main["train"]["micro_batch_size"] = 3
    with pytest.raises(ValueError, match="divisible"):
        training_plan(main)


def test_fineweb_streaming_contract(smoke_config, monkeypatch):
    from types import SimpleNamespace
    from xsa_repro.prepare_data import synthetic_documents
    import datasets
    import huggingface_hub
    import tiktoken
    import xsa_repro.prepare_data as preparation
    monkeypatch.setattr(preparation, "serial_parquet_reader", lambda stream: stream)
    calls = {}
    sha = "9bb295ddab0e05d785b879661af7260fed5140fc"
    def info(dataset_id, revision):
        calls["api"] = (dataset_id, revision)
        return SimpleNamespace(sha=sha)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=info))
    class FakeStream:
        def shuffle(self, **kwargs):
            calls["shuffle"] = kwargs
            return synthetic_documents()
    def load(dataset_id, **kwargs):
        calls["load"] = (dataset_id, kwargs)
        return FakeStream()
    monkeypatch.setattr(datasets, "load_dataset", load)
    class FakeGPT2:
        name, n_vocab = "gpt2", 50257
        _mergeable_ranks, _special_tokens, _pat_str = {b"a": 0}, {"<|endoftext|>": 50256}, "fake-test-regex"
        def encode_ordinary(self, text):
            return list(text.encode("ascii"))
    def get_encoding(name):
        calls["tokenizer"] = name
        return FakeGPT2()
    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)
    c = smoke_config
    c["data"].update(source="fineweb", tokenizer="gpt2", revision=sha, eos_id=50256)
    c["model"]["vocab_size"] = 50257
    manifest = prepare(c)
    assert calls["load"] == ("HuggingFaceFW/fineweb", {"name": "sample-10BT", "split": "train", "streaming": True, "revision": sha})
    assert calls["api"] == ("HuggingFaceFW/fineweb", sha)
    assert calls["shuffle"] == {"seed": 123, "buffer_size": 32}
    assert calls["tokenizer"] == "gpt2" and manifest["revision"] == sha
    assert manifest["tokenizer"]["eot_id"] == 50256
    assert not manifest["synthetic"]
    assert manifest["preparation_runtime"]["arrow_cpu_threads"] == 1
    assert manifest["preparation_runtime"]["arrow_use_threads"] is False


def test_arrow_single_reader_shutdown(tmp_path):
    import subprocess
    import sys
    # Exercise a partial read of the real HF iterable, with our serial callback.
    # A timeout catches the actual native-shutdown regression, not just a counter.
    script = """
import sys
import os
os.environ['HF_HOME'] = sys.argv[2]
os.environ['HF_DATASETS_OFFLINE'] = '1'
import pyarrow as pa
import pyarrow.parquet as pq
from types import SimpleNamespace
from datasets.iterable_dataset import ArrowExamplesIterable
from datasets.packaged_modules.parquet.parquet import Parquet
from xsa_repro.prepare_data import serial_parquet_reader
pa.set_cpu_count(1)
pq.write_table(pa.table({'text': ['sample document'] * 3000}), sys.argv[1])
# Test the real HF Arrow iterable without spawning an unrelated shared-memory
# manager (IterableDataset's epoch tensor otherwise requires OS IPC permissions).
builder = Parquet(cache_dir=sys.argv[2])
examples = ArrowExamplesIterable(builder._generate_tables, {'files': [[sys.argv[1]]]})
serial_parquet_reader(SimpleNamespace(_ex_iterable=examples))
iterator = iter(examples)
assert next(iterator)[1]['text'] == 'sample document'
iterator.close()
print('reader closed')
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "partial.parquet"), str(tmp_path / "hf")],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "reader closed" in result.stdout


def test_exhausted_source_does_not_publish_partial_corpus(smoke_config):
    from pathlib import Path
    output = Path(smoke_config["data"]["processed_dir"])
    with pytest.raises(RuntimeError, match="exhausted"):
        prepare_documents(iter([{"text": "tiny"}]), smoke_config, SyntheticTokenizer(), "test", "local")
    assert not output.exists()
    assert not list(output.parent.glob(output.name + ".partial-*"))


def test_memmap_rejects_invalid_tokens(tmp_path):
    path = tmp_path / "bad.bin"
    np.asarray([0, 1, 256, 2, 3], dtype="<u2").tofile(path)
    with pytest.raises(ValueError, match="outside"):
        TokenStream(path, 4, 256)
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="malformed"):
        TokenStream(path, 4, 256)
