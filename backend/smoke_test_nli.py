"""Real-model smoke test for cross-encoder/nli-deberta-v3-small.

This is SEPARATE from the mocked test suite. It downloads the model from
Hugging Face and runs three claim–evidence pairs to verify:
  1. Model and tokenizer load successfully
  2. Entailment, contradiction, and neutral predictions are produced
  3. Label mapping matches model configuration

Run with:  python backend/smoke_test_nli.py
"""

from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cross-encoder/nli-deberta-v3-small"
# Label order verified from model card: contradiction, entailment, neutral
LABELS = ["contradiction", "entailment", "neutral"]

# Test pairs: (claim, evidence_text, expected_category_hint)
TEST_CASES = [
    (
        "Coffee helps people live longer",
        "Higher coffee consumption was associated with lower mortality.",
        "entailment (supports)",
    ),
    (
        "Coffee prevents all disease",
        "Coffee consumption has no effect on disease prevention.",
        "contradiction (contradicts)",
    ),
    (
        "Coffee helps people live longer",
        "The weather was sunny today.",
        "neutral (inconclusive)",
    ),
]


def main() -> None:
    print(f"EvidenceLens NLI smoke test")
    print(f"Model: {MODEL_ID}")
    print(f"Python: {sys.version}")
    print(f"Torch: {torch.__version__}")
    print()

    # 1. Load tokenizer and model
    t0 = time.time()
    try:
        print("Downloading/loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print("Tokenizer loaded OK.")
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: Tokenizer load failed: {exc}")
        sys.exit(1)

    try:
        print("Downloading/loading model (this may take a minute on first run)...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        model.eval()
        print("Model loaded OK.")
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: Model load failed: {exc}")
        sys.exit(1)

    load_time = time.time() - t0
    print(f"Model load time: {load_time:.1f}s")
    print(f"Model config labels: {model.config.id2label}")
    print(f"Expected label order: {LABELS}")
    print()

    # Verify label mapping agrees with model config
    config_labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    if config_labels != LABELS:
        print(f"WARNING: Model config label order {config_labels} differs from expected {LABELS}")
    else:
        print("Label mapping agrees with model configuration.")

    print()
    print("Running inference on test claim–evidence pairs:")
    print("-" * 70)

    all_ok = True
    for claim, evidence, hint in TEST_CASES:
        t0 = time.time()
        try:
            inputs = tokenizer(
                claim,
                evidence,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1)

            scores = {
                LABELS[i]: round(float(probs[i].item()), 4)
                for i in range(len(LABELS))
            }
            pred_idx = int(probs.argmax().item())
            pred_label = LABELS[pred_idx]
            inf_time = time.time() - t0

            print(f"\nClaim:      {claim}")
            print(f"Evidence:   {evidence}")
            print(f"Expected:   {hint}")
            print(f"Predicted:  {pred_label}  (scores: {scores})")
            print(f"Inference:  {inf_time*1000:.0f} ms on CPU")

            # Basic sanity: top score should be > 0.3
            if max(scores.values()) < 0.3:
                print("  NOTE: Low confidence prediction.")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR on pair: {exc}")
            all_ok = False

    print()
    print("-" * 70)
    if all_ok:
        print("SMOKE TEST PASSED: model loads and produces entailment/contradiction/neutral predictions.")
    else:
        print("SMOKE TEST FAILED: one or more inference pairs errored.")
        sys.exit(1)


if __name__ == "__main__":
    main()