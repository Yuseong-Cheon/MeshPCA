#!/usr/bin/env python3
"""Create base/support/head masks for one desk-lamp RGB frame with SAM3."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def select_boxes(image, names):
    import tkinter as tk
    from PIL import ImageTk

    scale = min(1040 / image.width, 585 / image.height, 1.0)
    shown = image.resize((round(image.width * scale), round(image.height * scale)))
    root = tk.Tk()
    root.title("SAM 램프 박스 선택")
    photo = ImageTk.PhotoImage(shown)
    canvas = tk.Canvas(root, width=shown.width, height=shown.height,
                       cursor="cross")
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.pack()
    label = tk.Label(root, font=("Sans", 14, "bold"))
    label.pack(pady=4)
    boxes, current, start = [], None, None
    colors = ("#ff6414", "#28d228", "#f0281e")

    def update_label():
        label.config(text=f"{len(boxes) + 1}/3 {names[len(boxes)]}: "
                          "왼쪽 드래그 후 확정")

    def press(event):
        nonlocal start, current
        start = (event.x, event.y)
        if current is not None:
            canvas.delete(current[0])
        item = canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                       outline=colors[len(boxes)], width=3)
        current = (item, event.x, event.y, event.x, event.y)

    def drag(event):
        nonlocal current
        if start is None:
            return
        x = np.clip(event.x, 0, shown.width - 1)
        y = np.clip(event.y, 0, shown.height - 1)
        canvas.coords(current[0], start[0], start[1], x, y)
        current = (current[0], start[0], start[1], x, y)

    def confirm():
        nonlocal current, start
        if current is None:
            return
        _, x0, y0, x1, y1 = current
        x0, x1 = sorted((x0 / scale, x1 / scale))
        y0, y1 = sorted((y0 / scale, y1 / scale))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return
        boxes.append((x0, y0, x1, y1))
        current = start = None
        if len(boxes) == len(names):
            root.destroy()
        else:
            update_label()

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", drag)
    tk.Button(root, text="확정", command=confirm, width=18).pack(pady=(0, 6))
    update_label()
    root.mainloop()
    if len(boxes) != len(names):
        raise RuntimeError("박스 선택이 완료되지 않았습니다.")
    return dict(zip(names, boxes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image

    sys.path.insert(0, str(args.sam3_root.resolve()))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    image = Image.open(args.image).convert("RGB")
    boxes = (select_boxes(image, ("base", "support", "head"))
             if args.manual else None)
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

    def from_box(name):
        x0, y0, x1, y1 = boxes[name]
        box = [(x0 + x1) / (2 * image.width),
               (y0 + y1) / (2 * image.height),
               (x1 - x0) / image.width, (y1 - y0) / image.height]
        state = processor.add_geometric_prompt(
            box, True, processor.set_image(image))
        scores = state["scores"].float().cpu().numpy()
        if not len(scores):
            raise RuntimeError(f"SAM3 found no mask in the {name} box")
        index = int(np.argmax(scores))
        mask = state["masks"][index, 0].cpu().numpy().astype(bool)
        region = np.zeros(mask.shape, dtype=bool)
        region[round(y0):round(y1), round(x0):round(x1)] = True
        return mask & region, float(scores[index]), "manual box"

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if boxes is None:
            base, base_score, base_prompt = best("lamp base", "desk lamp base")
            head, head_score, head_prompt = best(
                "lamp light bar", "lamp shade", "lamp head")
            whole, whole_score, whole_prompt = best("lamp support", "desk lamp")
            base &= whole
            head &= whole & ~base
            support = whole & ~(base | head)
        else:
            base, base_score, base_prompt = from_box("base")
            support, whole_score, whole_prompt = from_box("support")
            head, head_score, head_prompt = from_box("head")
            support &= ~base
            head &= ~(base | support)
    masks = {"base": base, "support": support, "head": head}
    pixels = {name: int(mask.sum()) for name, mask in masks.items()}
    if (any(count < 1000 for count in pixels.values())
            or (boxes is None and whole.mean() > 0.3)):
        raise RuntimeError(f"suspicious lamp masks: {pixels}")

    view = np.asarray(image).copy()
    colors = {"base": (20, 100, 255), "support": (40, 210, 40),
              "head": (240, 40, 30)}
    for name, mask in masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / f"{name}.png")
        view[mask] = (0.35 * view[mask]
                      + 0.65 * np.asarray(colors[name])).astype(np.uint8)
    Image.fromarray(view).save(args.output / "preview.png")
    summary = {
        "method": ("SAM3 manual box masks" if boxes is not None else
                   "SAM3 text masks: support = whole lamp - base - head"),
        "prompts": {"base": base_prompt, "head": head_prompt,
                    "whole": whole_prompt},
        "scores": {"base": base_score, "head": head_score,
                   "whole": whole_score},
        "pixels": pixels,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
