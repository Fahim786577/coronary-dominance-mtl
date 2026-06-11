"""CPU smoke test for supported temporal MTL backbones."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import CoronaryTemporalMTL, SingleFrameTeacher, VideoTeacher


BACKBONES = ("resnet18", "mobilenet_v2", "densenet121")


def _print_shapes(outputs: dict[str, torch.Tensor]) -> None:
    for task, logits in outputs.items():
        print(f"{task}: {tuple(logits.shape)}")


def _fake_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "occlusion_images": torch.randn(1, 2, 1, 512, 512, device=device),
        "frame_quality_images": torch.randn(1, 1, 512, 512, device=device),
        "dominance_images": torch.randn(1, 1, 512, 512, device=device),
    }


def _test_mtl_model(backbone_name: str, device: torch.device) -> None:
    model = CoronaryTemporalMTL(backbone_name=backbone_name, pretrained=False).to(device)
    model.eval()
    inputs = _fake_inputs(device)

    with torch.no_grad():
        outputs = model(inputs)
        print(f"\n{backbone_name} all-task forward output shapes:")
        _print_shapes(outputs)

        assert outputs["occlusion"].shape == (1, 2)
        assert outputs["frame_quality"].shape == (1, 2)
        assert outputs["dominance"].shape == (1, 2)

        partial_inputs = [
            {"occlusion_images": inputs["occlusion_images"]},
            {"frame_quality_images": inputs["frame_quality_images"]},
            {"dominance_images": inputs["dominance_images"]},
        ]
        expected_keys = ["occlusion", "frame_quality", "dominance"]

        for partial_input, expected_key in zip(partial_inputs, expected_keys):
            partial_outputs = model(partial_input)
            print(f"{backbone_name} partial forward output shapes ({expected_key}):")
            _print_shapes(partial_outputs)
            assert set(partial_outputs) == {expected_key}
            assert partial_outputs[expected_key].shape == (1, 2)


def _test_teachers(backbone_name: str, device: torch.device) -> None:
    single_frame_teacher = SingleFrameTeacher(
        backbone_name=backbone_name,
        pretrained=False,
    ).to(device)
    video_teacher = VideoTeacher(
        backbone_name=backbone_name,
        pretrained=False,
    ).to(device)
    single_frame_teacher.eval()
    video_teacher.eval()

    frame_images = torch.randn(1, 1, 512, 512, device=device)
    clip_images = torch.randn(1, 2, 1, 512, 512, device=device)

    with torch.no_grad():
        single_logits = single_frame_teacher(frame_images)
        video_logits = video_teacher(clip_images)

    print(f"{backbone_name} teacher output shapes:")
    print(f"single_frame_teacher: {tuple(single_logits.shape)}")
    print(f"video_teacher: {tuple(video_logits.shape)}")
    assert single_logits.shape == (1, 2)
    assert video_logits.shape == (1, 2)


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")

    for backbone_name in BACKBONES:
        _test_mtl_model(backbone_name, device)
        _test_teachers(backbone_name, device)

    print("\nModel smoke test passed for all supported backbones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
