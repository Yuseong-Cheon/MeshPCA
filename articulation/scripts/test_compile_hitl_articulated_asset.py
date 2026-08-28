#!/usr/bin/env python3
"""Minimal compiler regression check."""

import json
import tempfile
from pathlib import Path

import trimesh

from compile_hitl_articulated_asset import compile_asset


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        a = trimesh.creation.box([0.2, 0.1, 0.02]); a.export(root / "a.ply")
        b = trimesh.creation.box([0.2, 0.02, 0.1]); b.apply_translation([0, 0.05, 0.05]); b.export(root / "b.ply")
        manifest = {
            "mode": "hitl", "robot_name": "test", "links": [
                {"name": "a", "visual": str(root / "a.ply"), "collision": str(root / "a.ply")},
                {"name": "b", "visual": str(root / "b.ply"), "collision": str(root / "b.ply")},
            ], "joints": [{
                "name": "a_to_b", "parent": "a", "child": "b", "type": "revolute",
                "origin": "contact", "axis": [1, 0, 0], "limits_source": "unverified_default",
            }],
        }
        path = root / "manifest.json"; path.write_text(json.dumps(manifest))
        audit = compile_asset(path, root / "out")
        assert audit["status"] == "GEOMETRY_PASS_PHYSICS_PROVISIONAL"
        assert audit["root_link"] == "a" and len(audit["joints"]) == 1
        manifest["links"][1].pop("collision")
        manifest["links"][1]["auto_shell_source"] = str(root / "b.ply")
        path.write_text(json.dumps(manifest))
        shell_audit = compile_asset(path, root / "shell_out")
        assert shell_audit["links"][1]["collision_backend"]["backend"] == "automatic_morphology_shell"
        print('{"status":"PASS","links":2,"joints":1}')


if __name__ == "__main__":
    main()
