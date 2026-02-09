# Algorithm Unit Tests

## What was done

Added **89 pytest tests** for all 5 algorithms in `algorithm/`. Tests run in ~3s on CPU.

## Files created

| File | Purpose |
|------|---------|
| `tests/conftest.py` | Shared fixtures: MockDenoiser, MockPipeline, MockState, MockLogger |
| `tests/test_ema.py` | 14 tests for EMA weight averaging and lifecycle |
| `tests/test_repa.py` | 17 tests for REPA MLP, hooks, loss wrapping (DINOv3 mocked) |
| `tests/test_tread.py` | 27 tests for token routing, gather/scatter, PE handling |
| `tests/test_sprint.py` | 14 tests for fusion layer, mask token, dense stash |
| `tests/test_contrastive_flow_matching.py` | 12 tests for contrastive loss formula and wrapping |
| `pyproject.toml` | Added pytest config and test dependency |

## Test approach

- **Unit tests**: Individual method verification (e.g., `compute_ema` formula, `_sample_indices` coverage, MLP shape)
- **Integration tests**: Full forward + loss + backward through mock pipeline with algorithms active
- **REPA encoder**: Mocked via `monkeypatch` to avoid DINOv3 downloads
- **Import workaround**: `conftest.py` pre-loads `dataset.constants` to bypass broken `dataset/__init__.py`
- **REPA dtype**: Integration tests use bfloat16 pipeline to match REPA's MLP dtype

## How to run

```bash
conda activate photoroom
python -m pytest tests/ -v
```
