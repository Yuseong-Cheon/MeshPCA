#!/usr/bin/env python3
"""Create base and moving-link masks for one laptop RGB frame with SAM3."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sam3-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image

    sys.path.insert(0, str(args.sam3_root.resolve()))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    image = Image.open(args.image).convert("RGB")
    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.05)

    def best(*prompts):
        for prompt in prompts:
            state = processor.set_text_prompt(prompt, processor.set_image(image))
            scores = state["scores"].float().cpu().numpy()
            if len(scores):
                index = int(np.argmax(scores))
                return (state["masks"][index, 0].cpu().numpy().astype(bool),
                        float(scores[index]), prompt)
        raise RuntimeError(f"SAM3 found no mask for {prompts!r}")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        base, base_score, base_prompt = best("laptop base", "laptop keyboard")
        moving, moving_score, moving_prompt = best("laptop screen", "laptop lid")

    overlap = base & moving
    base &= ~moving
    masks = {"base": base, "moving_link": moving}
    pixels = {name: int(mask.sum()) for name, mask in masks.items()}
    if any(count < 1000 for count in pixels.values()) or overlap.mean() > 0.05:
        raise RuntimeError(f"suspicious automatic laptop masks: {pixels}")

    view = np.asarray(image).copy()
    colors = {"base": (30, 120, 255), "moving_link": (255, 60, 30)}
    for name, mask in masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / f"{name}.png")
        view[mask] = (0.35 * view[mask]
                      + 0.65 * np.asarray(colors[name])).astype(np.uint8)
    Image.fromarray(view).save(args.output / "preview.png")
    summary = {
        "prompts": {"base": base_prompt, "moving_link": moving_prompt},
        "scores": {"base": base_score, "moving_link": moving_score},
        "pixels": pixels,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
