"""CPU smoke test for the ResNet18 temporal MTL architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import CoronaryTemporalMTL


def _print_shapes(outputs: dict[str, torch.Tensor]) -> None:
    for task, logits in outputs.items():
        print(f"{task}: {tuple(logits.shape)}")


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = CoronaryTemporalMTL(pretrained=False).to(device)
    model.eval()

    inputs = {
        "occlusion_images": torch.randn(2, 15, 1, 512, 512, device=device),
        "frame_quality_images": torch.randn(2, 1, 512, 512, device=device),
        "dominance_images": torch.randn(2, 1, 512, 512, device=device),
    }

    with torch.no_grad():
        outputs = model(inputs)
        print("All-task forward output shapes:")
        _print_shapes(outputs)

        assert outputs["occlusion"].shape == (2, 2)
        assert outputs["frame_quality"].shape == (2, 2)
        assert outputs["dominance"].shape == (2, 2)

        partial_inputs = [
            {"occlusion_images": inputs["occlusion_images"]},
            {"frame_quality_images": inputs["frame_quality_images"]},
            {"dominance_images": inputs["dominance_images"]},
        ]
        expected_keys = ["occlusion", "frame_quality", "dominance"]

        for partial_input, expected_key in zip(partial_inputs, expected_keys):
            partial_outputs = model(partial_input)
            print(f"\nPartial forward output shapes ({expected_key}):")
            _print_shapes(partial_outputs)
            assert set(partial_outputs) == {expected_key}
            assert partial_outputs[expected_key].shape == (2, 2)

    print("\nModel smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
