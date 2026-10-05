"""Evaluate frozen portable policies on one untouched teacher-observation split."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_action_policy import PolicyNet, load, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    assert not Path(a.report).exists(), "refusing to overwrite frozen assessment"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load(a.test)
    results = []
    for path in a.models.split(","):
        raw = Path(path).read_bytes()
        width = {b"CABOPL01": 24, b"CABOPL02": 64}[raw[:8]]
        assert len(raw) == (7500 if width == 24 else 25100)
        flat = np.frombuffer(raw[8:], dtype="<f4").copy()
        assert np.isfinite(flat).all() and (np.abs(flat) <= 100).all()
        net, offset = PolicyNet(width).to(device), 0
        with torch.no_grad():
            for layer in net.layers:
                for parameter in [layer.weight, layer.bias]:
                    size = parameter.numel()
                    parameter.copy_(torch.from_numpy(flat[offset:offset + size].reshape(parameter.shape)))
                    offset += size
        assert offset == len(flat)
        assessment = metrics(net, data, device)
        results.append({"model": path, "sha256": hashlib.sha256(raw).hexdigest(), "metrics": assessment})
    report = {"purpose": "Teacher prediction only, not match strength", "device": str(device),
              "data": data["manifest"], "models": results}
    Path(a.report).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
