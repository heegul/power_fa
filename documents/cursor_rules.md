# Cursor Rules – Power & Frequency Allocation Simulator

This document must stay in the repository root so that Cursor's LLM always loads it as additional context.  It condenses the agreed‐upon rules for coding style, project organisation, testing, security, and deployment.  **Any code or documentation generated inside Cursor should comply with everything written here.**

---

## 0. Core Principles

1. Single, reusable simulation framework; algorithms are plug-ins.
2. Reproducibility via global config & seed.
3. Maximum CPU utilisation on a 64 GB Mac-mini.
4. Clear, automated visualisation of results.
5. High code quality (SOLID, modular, documented, tested).

## 1. Repository Layout (Canonical)

See README for tree diagram.  Major directories:

* `src/simulator/` – environment, scenario, engine, metrics, visualise.
* `src/algorithms/` – each algorithm (FS, ML, heuristics) as a module.
* `src/config.py` – central `SimulationConfig` dataclass & seed utilities.
* `src/cli.py` – entry-point with sub-commands `run`, `batch`, `visualize`.
* `tests/` – pytest unit & integration tests (≥80 % coverage target).

## 2. Coding Standards

* **Python ≥3.9**, `typing` annotations mandatory.
* Follow SOLID and clean-architecture practices; functions < 50 LOC when possible.
* External config via YAML/ENV, never hard-code experiment parameters.
* Lint with **black**, **flake8**, and **isort**.
* Include doctrings in [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## 3. Testing & CI

* Adopt TDD: write tests before or alongside implementation.
* Use `pytest` with markers: `unit`, `integration`, `slow`.
* GitHub Actions pipeline: install deps → lint → tests → optional Docker build.

## 4. Parallel Execution Rules

* Prefer `ray` (`ray.init(num_cpus=os.cpu_count())`) but fall back to `ProcessPoolExecutor`.
* Ensure deterministic randomness: pass explicit `seed` to each parallel worker and set
  `os.environ["PYTHONHASHSEED"]`.
* Limit nested parallelism by setting env vars:
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_MAX_THREADS=1`.

## 5. Security & Robustness

* Validate algorithm outputs (power bounds, FA indices).
* Catch divide-by-zero errors in SINR calc.
* Log warnings instead of crashing on recoverable issues.

## 6. Visualisation Mandates

* Provide at least:
  * Sum-rate **CDF** or histogram for each algorithm.
  * **Power map**: scatter plot of Tx positions, marker size ∝ power, colour ∝ FA.
* Save plots to `plots/` as both PNG and vector (SVG/PDF).

## 7. Documentation Duties

* Keep **README.md** updated with new CLI flags and directory changes.
* For big design decisions, update `docs/changelog.md` (to be created when needed).

## 8. Dependency Management

* All libs pinned in `requirements.txt` (no `>=` versions—use `==`).
* Optional/extra dependencies can live in `requirements-ml.txt`.

## 9. Rule Updates

This rule file is a living document. Every substantial change to project guidelines **must** be reflected here and committed in the same PR. 

**Usage Example:**
```bash
python -m src.simulator.scenario generate_samples \
    --n_samples 1000 \
    --n_pairs 4 \
    --area_size_m 100 \
    --channel_type urban \
    --seed 42 \
    --out_path samples_urban_4pairs.npy
```

- The output `.npy` file will contain a tensor of shape `(n_samples, n_pairs, n_pairs)` with channel gains in dB.
- You can change `--channel_type` to `suburban` or `rural` for different pathloss exponents.

Let me know if you want a CLI wrapper, integration into your main CLI, or sample code for loading these datasets in PyTorch! 