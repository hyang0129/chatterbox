---
id: BUG-2
title: "asyncio.get_event_loop() is deprecated in Python 3.10+ — use get_running_loop()"
severity: Medium
status: open
depends_on: BUG-1
---

## Problem

This issue is closely coupled to BUG-1. The current code does not call
`asyncio.get_event_loop()` today, but the BUG-1 fix requires adding
`run_in_executor` calls inside async endpoints. The correct form is
`asyncio.get_running_loop().run_in_executor(...)`.

If the wrong form (`asyncio.get_event_loop()`) is used instead:

- **Python 3.10** — emits `DeprecationWarning` when no current event loop exists in the
  calling thread (e.g. in tests or in a `run_in_executor` callback).
- **Python 3.12+** — `get_event_loop()` raises `RuntimeError` inside a running coroutine
  when there is no current loop in the calling thread.

The project targets Python 3.11, so this is a `DeprecationWarning` today and a hard error
on the next Python upgrade.

## Implementation Plan

This issue is addressed as part of BUG-1. When adding `run_in_executor` calls:

**Use this form (correct):**

```python
wav = await asyncio.get_running_loop().run_in_executor(
    None,
    lambda: model.generate(...),
)
```

**Never this form (deprecated):**

```python
loop = asyncio.get_event_loop()          # deprecated inside a coroutine
wav = await loop.run_in_executor(None, ...)
```

`asyncio.get_running_loop()` is always safe inside an `async def` function — it raises
`RuntimeError` immediately if called outside a running loop, making misuse visible at
development time rather than silently degrading in production.

### Checklist

- [ ] All `run_in_executor` calls in `app/main.py` use `asyncio.get_running_loop()`
- [ ] No `asyncio.get_event_loop()` calls remain anywhere in `app/`
- [ ] `ruff check .` passes with no warnings

## Testing Plan

### Static analysis

```bash
# Must return no matches after BUG-1 + BUG-2 are implemented
grep -rn "get_event_loop" app/
```

### Unit test

Add to `tests/test_api.py` (or a new `tests/test_async.py`):

```python
import asyncio

def test_synthesize_uses_running_loop(client, mock_model, monkeypatch):
    """get_running_loop() must be used, not get_event_loop()."""
    called_with_running_loop = []

    original = asyncio.get_running_loop

    def patched_get_running_loop():
        called_with_running_loop.append(True)
        return original()

    monkeypatch.setattr(asyncio, "get_running_loop", patched_get_running_loop)
    r = client.post("/tts", json={"text": "Loop check."})
    assert r.status_code == 200
    assert called_with_running_loop, "get_running_loop() was never called"
```

This test will fail if someone accidentally reverts to `get_event_loop()`.
