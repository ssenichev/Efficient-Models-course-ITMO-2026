# Exclusive Self Attention

Обучение NanoGPT-подобной модели с обычным causal attention (SA) и Exclusive Self Attention (XSA) на FineWeb. Обе модели обучаются с нуля; внутри пары совпадают начальные веса, корпус, sampler, оптимизатор и токенный бюджет.

XSA применяется отдельно к каждой голове после causal SDPA и до выходной проекции:

```python
v_hat = F.normalize(v, dim=-1, eps=1e-6)
z = y - (y * v_hat).sum(dim=-1, keepdim=True) * v_hat
```

Градиент проходит через `v`. RoPE применяется к Q/K. Модель: 6 слоёв, ширина 512, 8 голов, GELU FFN ×4, LayerNorm после embedding и перед блоками, связанные embedding/lm head. Размер — 44 647 936 параметров.

## Установка

Python ≥3.10, PyTorch ≥2.5. Команды выполняются из `hw2/xsa-fineweb-repro`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Для FineWeb нужен доступ к Hugging Face; при первом запуске загружается GPT-2 BPE. Тесты и smoke работают без сети. MPS-тесты пропускаются, если Metal недоступен. Для CUDA предварительно установите подходящую сборку PyTorch.

## Структура

```text
configs/              архитектура, данные и бюджеты экспериментов
xsa_repro/
  model.py            GPT, RoPE, SA/XSA
  prepare_data.py     FineWeb, split, дедупликация, токенизация
  data.py             memmap, sampler, проверка корпуса
  train.py            обучение, checkpoints, resume
  evaluate.py         validation NLL и perplexity
  diagnostics.py      cosine до/после проекции
  plot.py             графики и сводные таблицы
  benchmark.py        измерение скорости и памяти
  sweep.py            последовательный запуск пар и seeds
  config.py           загрузка конфигов и расчёт бюджета
  utils.py            provenance, RNG, устройства
tests/                unit tests и end-to-end smoke
```

Данные, веса, кэши и результаты создаются локально в `data/`, `runs/`, `reports/` и исключены из Git.

## Офлайн smoke

Тест проходит всю цепочку: подготовка синтетического корпуса → обучение SA/XSA → validation → диагностика → графики. Он также проверяет одинаковые начальные веса и отклонение несопоставимых пар.

```bash
pytest -q tests/test_smoke.py::test_offline_end_to_end
```

Для проверки CLI на чистых выходных директориях:

```bash
python -m xsa_repro.prepare_data --config configs/smoke.yaml
python -m xsa_repro.train --config configs/smoke.yaml --attention sa --seed 42
python -m xsa_repro.train --config configs/smoke.yaml --attention xsa --seed 42
python -m xsa_repro.plot \
  --run-dirs runs/synthetic_smoke/sa_seed42 runs/synthetic_smoke/xsa_seed42 \
  --out-dir reports/smoke
```

Smoke использует byte tokenizer, модель 2×32 и четыре optimizer updates. Это проверка программы, а не качества на FineWeb.

## Эксперименты на Apple Silicon

MPS-конфиги используют FP32, microbatch=1 и `compile=false`. Накопление градиентов сохраняет заданный global batch. Проверка скорости на синтетических токенах выполняет настоящие forward/backward/AdamW updates:

```bash
python -m xsa_repro.benchmark --config configs/mac_10m.yaml \
  --attention both --warmup-steps 2 --steps 5 --out reports/benchmark.json
```

Три SA/XSA-пары по 10M токенов, T=256:

```bash
python -m xsa_repro.prepare_data --config configs/mac_10m.yaml
for seed in 42 43 44; do
  python -m xsa_repro.train --config configs/mac_10m.yaml --attention sa --seed "$seed"
  python -m xsa_repro.train --config configs/mac_10m.yaml --attention xsa --seed "$seed"
done
python -m xsa_repro.plot \
  --run-dirs runs/xsa_mac_45m_10m_ctx256/{sa,xsa}_seed{42,43,44} \
  --out-dir reports/10m
```

Контекстные абляции по 30M токенов, global batch=65536, seed=42:

```bash
python -m xsa_repro.prepare_data --config configs/mac_main.yaml
for context in 256 512 1024; do
  python -m xsa_repro.train --config "configs/mac_context_${context}.yaml" --attention sa --seed 42
  python -m xsa_repro.train --config "configs/mac_context_${context}.yaml" --attention xsa --seed 42
done
python -m xsa_repro.plot \
  --run-dirs runs/xsa_mac_context{256,512,1024}_30m/{sa,xsa}_seed42 \
  --out-dir reports/context
```

`mac_pilot.yaml` задаёт 1M-пару. `mac_main.yaml` — 100M-пару при T=1024. Полная последовательная серия (18 запусков: 10M и 100M на трёх seeds, три контекста по 30M) запускается отдельно:

```bash
python -m xsa_repro.sweep --plan configs/mac_sweep.json
```

Очередь хранит состояние в `reports/sweep_status.json`, исходники и логи — в `runs/sweep/`. Повторный вызов продолжает совместимые запуски. Файл `runs/sweep/STOP` останавливает очередь после текущей команды; удалите его перед продолжением. На macOS очередь удерживает машину от idle sleep через `caffeinate`.

Поля плана: `groups` (config/seeds), `workspace`, `artifacts_dir`, `minimum_free_gib`, `command_timeout_hours`. Необязательный `deadline` в ISO 8601 с timezone сохраняет снимок состояния в этот момент; очередь продолжает работу. После проверки завершённой пары очередь удаляет промежуточные checkpoints, сохраняя initial/latest/best/final и используемые версии best.

## Основная 100M-пара

`main.yaml` выбирает CUDA → MPS → CPU. Auto precision на CUDA — BF16 при поддержке, иначе FP16; на MPS/CPU — FP32.

```bash
python -m xsa_repro.prepare_data --config configs/main.yaml
python -m xsa_repro.train --config configs/main.yaml --attention sa --seed 42
python -m xsa_repro.train --config configs/main.yaml --attention xsa --seed 42
python -m xsa_repro.plot \
  --run-dirs runs/xsa_fineweb_45m/{sa,xsa}_seed42 \
  --out-dir reports/main
```

Global batch=65536, AdamW, LR 6e−4 → 6e−5, 100 warmup updates, cosine schedule, clip=1.0. Бюджет округляется вверх до полного batch: 100 007 936 токенов, 1526 updates. При уменьшении microbatch меняйте его одинаково для SA/XSA, сохраняя global batch.

В конце обучения автоматически выполняются full validation и диагностика. Отдельная оценка сохранённой модели:

```bash
python -m xsa_repro.evaluate --run-dir runs/xsa_fineweb_45m/sa_seed42 --full-val
python -m xsa_repro.diagnostics --run-dir runs/xsa_fineweb_45m/sa_seed42
```

Остановка после пяти updates с сохранением исходного schedule и продолжение:

```bash
python -m xsa_repro.train --config configs/main.yaml --attention sa --seed 42 --max-steps 5
python -m xsa_repro.train --config configs/main.yaml --attention sa --seed 42 \
  --resume runs/xsa_fineweb_45m/sa_seed42/latest.pt
```

`--max-steps` — абсолютный номер update. Существующие данные и runs не перезаписываются. Для нового эксперимента задайте новые `experiment` и, если меняется корпус, `data.processed_dir`. Resume проверяет исходники, config и corpus; после изменения кода используйте версию из `source_snapshot` соответствующей очереди.

## Протокол и результаты

- FineWeb `sample-10BT`, revision `9bb295ddab0e05d785b879661af7260fed5140fc`. Потоковый shuffle с фиксированным seed и ограниченным буфером.
- Split и дедупликация по SHA-256 от `NFC(text).strip()` до packing. Токенизируется исходный текст: GPT-2 BPE + EOT 50256. Train/validation документные hashes не пересекаются. Manifest содержит SHA корпуса, tokenizer fingerprint и версии библиотек.
- Train sampler: PCG64(seed), выбор окон с возвращением, starts кратны T. Обработанный бюджет отличается от размера корпуса. SA/XSA получают одинаковую последовательность окон.
- Validation: все полные непересекающиеся окна; NLL взвешена по target tokens, PPL=exp(NLL). Промежуточная оценка использует фиксированный subset. Основное сравнение — `final.pt` при равном бюджете; `best_val.pt` выбирается по subset.
- Диагностика: cosine(Y,V) для SA/XSA и cosine(Z,V) после XSA. Первый токен окна и векторы с norm≤1e−6 исключаются. При norm(V)<eps проекция регуляризована.
- Checkpoints сохраняют optimizer/scheduler/scaler, RNG и sampler. Logs содержат tokens, NLL/PPL, LR, throughput и память. MPS memory — сэмплированная нижняя граница peak; CUDA memory — high-water mark.

`plot` проверяет сопоставимость пар, строит восемь PNG, CSV/JSON и Markdown по измеренным метрикам. Для нескольких seeds выводит среднее и sample standard deviation парной ΔNLL. Разные корпуса, бюджеты и T сравнивайте внутри соответствующих SA/XSA-пар. Малый масштаб эксперимента отличается от исходной статьи.

## Источники

- Shuangfei Zhai, [Exclusive Self Attention, arXiv:2603.09078v1](https://arxiv.org/abs/2603.09078).
- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT), MIT. Реализация здесь самостоятельная.
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/); исходный контент также регулируется условиями [Common Crawl](https://commoncrawl.org/terms-of-use).
- [tiktoken](https://github.com/openai/tiktoken), MIT.
