---
id: BUG-3
title: "Serverless (direct) mode thread-safety is undocumented — callers may use it unsafely"
severity: Medium
status: open
---

## Problem

The serverless usage pattern (`ChatterboxTurboTTS.from_pretrained(device="cuda")` once,
then `model.generate()` per segment) is not thread-safe. Nothing in the README,
docstrings, or API reference communicates this constraint.

Callers that share a single model instance across threads (e.g. via a
`ThreadPoolExecutor`) will hit:

- Race conditions on internal model state (non-deterministic audio)
- CUDA stream conflicts
- Potential CUDA OOM from overlapping allocations

**Specifically:** the video-agent pipeline (`chatterbox_direct` backend) calls
`model.generate()` for each audio segment in a run. If the orchestrator uses
`ThreadPoolExecutor` for parallel stage execution, multiple `generate()` calls can
overlap on the same model instance. This is silent data corruption — no exception is
raised in the common case.

## Implementation Plan

### 1. Add thread-safety warning to README serverless section

Insert the following callout immediately before the serverless code example:

```markdown
> **Thread safety:** A single `ChatterboxTurboTTS` instance must not be used from
> multiple threads concurrently. If you need concurrent synthesis, create one model
> instance per thread, or serialise all `generate()` calls behind a `threading.Lock`.
> On a single GPU with limited VRAM, serial single-instance use is strongly recommended.
```

### 2. Add docstring warning to `ChatterboxTurboTTS`

Add to the class docstring (or `generate()` method docstring) in the
`chatterbox-tts` package source:

```python
class ChatterboxTurboTTS:
    """
    Single-step text-to-speech model.

    Thread safety
    -------------
    Instances are NOT thread-safe. Do not share one instance across threads
    without external serialisation (e.g. ``threading.Lock``). Concurrent
    ``generate()`` calls on a single instance may cause CUDA OOM or produce
    non-deterministic audio.
    """
```

### 3. Document the `serial=True` requirement for `chatterbox_direct` in the video-agent

The video-agent integration plan already notes this constraint. The README should
cross-reference it:

> Pipeline callers using `chatterbox_direct` backend mode must run the audio stage
> serially (no `ThreadPoolExecutor`). Violating this causes non-deterministic output
> and potential CUDA OOM.

## Testing Plan

### Documentation review (no code changes needed for tests)

Verify:

1. `README.md` serverless section contains the thread-safety callout
2. `ChatterboxTurboTTS` class or `generate()` method docstring mentions thread safety

### Regression test (validates constraint, no GPU required)

Add to `tests/test_thread_safety.py`:

```python
import threading
from unittest.mock import MagicMock, call

def test_generate_called_serially_not_concurrently(mock_model):
    """
    Validate that if generate() is called from the same thread one after another
    (the correct usage), it works. This test documents the expected serial pattern.
    """
    texts = [
        "Did you know Allied soldiers called any enemy tank a Tiger?",
        "The American M4 Sherman was produced over 49,000 times.",
        "The Soviet T-34 shocked German engineers.",
    ]
    results = []
    for text in texts:
        wav = mock_model.generate(text)
        results.append(wav)

    assert mock_model.generate.call_count == 3
    assert len(results) == 3
```

The above documents the **correct** serial usage pattern that callers should follow.

### Fixture texts

The segment narrations from the video-agent WW2 tanks script serve as canonical test
input (see `tests/fixtures/ww2_tanks_segments.json`):

```json
[
  "Did you know Allied soldiers called any enemy tank a Tiger — regardless of its actual type — because the Tiger I was so feared?",
  "The American M4 Sherman was produced over 49,000 times — winning through sheer quantity over German armor quality.",
  "The Soviet T-34 shocked German engineers so much they considered copying its sloped armor for their own future tank designs.",
  "Germany's Panzer VIII Maus weighed 188 tonnes — so heavy it could only cross rivers by driving along the submerged riverbed.",
  "Britain's Hobart's Funnies were Churchill variants that could flail mines, bridge gaps, and lay roads — revolutionizing assault engineering."
]
```
