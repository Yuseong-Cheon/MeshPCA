import numpy as np
import vedo
import trimesh
from typing import List, Dict, Any
from functools import partial
import pymeshlab
from sklearn.cluster import KMeans
from vtkmodules.vtkRenderingCore import vtkRenderer

from functions.lib.geometry import normalize, rotmat


if not hasattr(vtkRenderer, "AddActor2D"):
    def _add_button(self, fnc=None, states=("On", "Off"), c=("w", "w"),
                    bc=("green4", "red4"), pos=(0.7, 0.1), size=24,
                    font="Courier", bold=True, italic=False, alpha=1, angle=0):
        if not self.interactor:
            return None
        button = vedo.Button(fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle)
        self.renderer.AddViewProp(button.actor)
        button.function_id = button.actor.AddObserver("PickEvent", button.function)
        self.buttons.append(button)
        return button

    vedo.Plotter.add_button = _add_button

# =============================================================================
# Helper: Convert Trimesh list to Vedo
# =============================================================================
def trimesh_list_to_vedo_mesh(parts: List[trimesh.Trimesh]) -> vedo.Mesh | None:
    """Convert list of trimesh.Trimesh into a single vedo.Mesh (or None if empty)."""
    if not parts:
        return None
    v_all = []
    f_all = []
    off = 0
    for tm in parts:
        v = np.asarray(tm.vertices, dtype=float)
        f = np.asarray(tm.faces, dtype=int)
        v_all.append(v)
        f_all.append(f + off)
        off += v.shape[0]
    V = np.vstack(v_all)
    F = np.vstack(f_all)
    return vedo.Mesh([V, F])


def spatial_seed_belongings(parts: List[trimesh.Trimesh], categories_num: int) -> list[int]:
    """Seed a serial link chain; HITL remains authoritative."""
    if len(parts) < categories_num:
        return [0] * (len(parts) + 1)
    centers = np.vstack([part.centroid for part in parts])
    weights = np.maximum([abs(float(part.volume)) for part in parts], 1e-12)
    labels = KMeans(categories_num, random_state=0, n_init=20).fit_predict(
        centers, sample_weight=weights)
    cluster_centers = np.vstack([
        np.average(centers[labels == category], axis=0, weights=weights[labels == category])
        for category in range(categories_num)
    ])
    cluster_volumes = np.array([
        weights[labels == category].sum() for category in range(categories_num)
    ])
    order = [int(np.argmax(cluster_volumes))]
    while len(order) < categories_num:
        remaining = [category for category in range(categories_num) if category not in order]
        order.append(min(remaining, key=lambda category: np.linalg.norm(
            cluster_centers[category] - cluster_centers[order[-1]])))
    category_for_cluster = {cluster: category for category, cluster in enumerate(order, 1)}
    return [0, *[category_for_cluster[int(label)] for label in labels]]

# =============================================================================
# Visualization — part selection callbacks
# =============================================================================
def recolor_parts(vmeshes, belongings, mode, rand_colors):
    palette = ("gray", "dodgerblue", "orange", "mediumseagreen", "violet", "gold")
    vmeshes[0].c("white").alpha(0.05)
    for i, vmesh in enumerate(vmeshes[1:], start=1):
        vmesh.c(palette[belongings[i] % len(palette)]).alpha(0.85)


def selection_status_text(belongings, mode, categories_num, error="", category_names=None):
    counts = {category: belongings[1:].count(category) for category in range(categories_num + 1)}
    names = category_names or [f"Category {i}" for i in range(1, categories_num + 1)]
    colors = ("blue", "orange", "green", "violet", "gold")
    ready = all(counts[category] > 0 for category in range(1, categories_num + 1))
    return ("STEP 1/3 — RORA COARSE HULL ASSIGNMENT\n"
            + " | ".join(f"C{i}={name}" for i, name in enumerate(names, 1)) + "\n"
            + "Colors: gray=Ambiguous | "
            + " | ".join(f"{colors[(i - 1) % len(colors)]}={name}"
                         for i, name in enumerate(names, 1)) + "\n"
            f"Current brush: {'Ambiguous' if mode == 0 else names[mode - 1]}\n"
            + "   ".join([f"Ambiguous: {counts[0]}"] + [
                f"C{category}: {counts[category]}" for category in range(1, categories_num + 1)
            ]) + "\n"
            f"Selected hull: {belongings[0] if belongings[0] else '-'}   "
            f"READY TO SAVE: {'YES' if ready else 'NO'}\n"
            "Gray Ambiguous hulls are excluded from hard seeds and assigned later by geometry.\n"
            f"Left click=assign | 0..{categories_num}=brush | Right click=undo | "
            "drag=rotate | wheel=zoom | S=save | Esc=cancel"
            + (f"\nERROR: {error}" if error else ""))


def update_selection_ui(vmeshes, belongings, shared, rand_colors, plt, error=""):
    recolor_parts(vmeshes, belongings, shared["mode"], rand_colors)
    shared["status_actor"].text(selection_status_text(
        belongings, shared["mode"], shared["categories_num"], error,
        shared.get("category_names")))
    plt.render()


def on_left_click(event, vmeshes, belongings, prevs, shared, rand_colors, plt):
    vmesh = event.actor
    mode = shared['mode']
    if vmesh is None:
        return
    if vmesh in vmeshes[1:]:
        idx = vmesh.idx_part
        prevs[idx] = belongings[idx]
        belongings[idx] = mode
        belongings[0] = idx
        update_selection_ui(vmeshes, belongings, shared, rand_colors, plt)


def on_right_click(event, vmeshes, belongings, prevs, shared, rand_colors, plt):
    vmesh = event.actor
    if vmesh is None:
        return
    if vmesh in vmeshes[1:]:
        idx = vmesh.idx_part
        belongings[idx] = prevs[idx]
        belongings[0] = idx
        update_selection_ui(vmeshes, belongings, shared, rand_colors, plt)


def on_tab(event, vmeshes, belongings, prevs, shared, categories_num, rand_colors, plt):
    if event.keypress in ('\t', 'Tab'):
        shared['mode'] = shared['mode'] % categories_num + 1
        mode = shared['mode']
        update_selection_ui(vmeshes, belongings, shared, rand_colors, plt)
        print(f"[info]: {'default mode' if mode==0 else f'selection mode #{mode}'}")


def on_key(event, vmeshes, belongings, prevs, shared, rand_colors, categories_num, plt):
    """S/Enter saves; ESC/q cancels; Tab cycles the link brush."""
    k = getattr(event, "keypress", None)
    if k in ("q", "Q", "Esc", "Escape", "\x1b"):
        shared["cancelled"] = True
        plt.close()
        return
    if k in ("s", "S", "Return", "Enter"):
        finish_part_selection(
            None, None, shared=shared, vmeshes=vmeshes, belongings=belongings,
            rand_colors=rand_colors, plt=plt)
        return
    if k in ("r", "R"):
        belongings[1:] = [0] * (len(belongings) - 1)
        belongings[0] = 0
        update_selection_ui(vmeshes, belongings, shared, rand_colors, plt)
        return
    if str(k).isdigit() and 0 <= int(k) <= categories_num:
        shared["mode"] = int(k)
        update_selection_ui(vmeshes, belongings, shared, rand_colors, plt)
        return
    if k in ('\t', 'Tab'):
        on_tab(event, vmeshes, belongings, prevs, shared, categories_num, rand_colors, plt)


def set_selection_mode(_widget, _event, *, mode, shared, vmeshes, belongings, rand_colors, plt):
    shared["mode"] = mode
    update_selection_ui(vmeshes, belongings, shared, rand_colors, plt)


def reset_part_selection(_widget, _event, *, shared, vmeshes, belongings, rand_colors, plt):
    belongings[1:] = [0] * (len(belongings) - 1)
    belongings[0] = 0
    update_selection_ui(vmeshes, belongings, shared, rand_colors, plt)


def finish_part_selection(_widget, _event, *, shared, vmeshes, belongings, rand_colors, plt):
    used = set(belongings[1:])
    required = set(range(1, shared["categories_num"] + 1))
    if (0 in used and not shared.get("allow_unassigned", False)) or not required.issubset(used):
        names = shared.get("category_names") or [str(i) for i in range(1, shared["categories_num"] + 1)]
        missing = [names[index - 1] for index in sorted(required - used)]
        detail = (f"Missing safe core: {', '.join(missing)}. " if missing else "")
        detail += "Ambiguous hulls are allowed; each link still needs one safe core."
        update_selection_ui(
            vmeshes, belongings, shared, rand_colors, plt,
            detail,
        )
        return
    shared["finished"] = True
    print("[RORA HITL] STEP 1/3 part assignment confirmed; continuing to joint axes")
    plt.close()

def select_parts_interactive(parts: List[trimesh.Trimesh], verts: np.ndarray, faces: np.ndarray,
                             categories_num: int, display: bool = True,
                             allow_unassigned: bool = False,
                             category_names: List[str] | None = None,
                             initial_belongings: List[int] | None = None) -> tuple[List[List[trimesh.Trimesh]], Any, Any]:
    """Interactive part selection UI. Returns (categories, belongings, rand_colors)."""
    if not display:
        # Default behavior if no display: assign based on something or just return empty?
        # Assuming if no display, we shouldn't fail but interactive selection is impossible.
        # This function is inherently interactive. If display=False, maybe skip?
        print("[warn]: select_parts_interactive called with display=False. Skipping UI.")
        # Fallback or error? For now return empty or simple grouping?
        # But this returns 'categories'. We can't proceed without categories.
        # If this is called, user probably expects interaction.
        # But for automation, we might need a way to bypass.
        # For now, just return empty categories and let the caller handle it.
        return [[] for _ in range(categories_num + 1)], [], []

    vmeshes, rand_colors = [], []
    base_actor = vedo.Mesh([verts, faces])
    if hasattr(base_actor, "pickable"):
        base_actor.pickable(False)
    vmeshes.append(base_actor); rand_colors.append((255, 255, 255))
    for i, part in enumerate(parts, start=1):
        actor = vedo.Mesh([part.vertices, part.faces])
        color = np.random.rand(3); actor.c(color).alpha(0.9)
        actor.properties.SetEdgeVisibility(True)
        actor.properties.SetEdgeColor(0.08, 0.08, 0.08)
        actor.properties.SetEdgeOpacity(0.7)
        actor.properties.SetEdgeWidth(1.0)
        actor.idx_part = i
        vmeshes.append(actor); rand_colors.append(color)

    if categories_num < 1:
        raise ValueError("categories_num must be positive")
    if category_names is not None and len(category_names) != categories_num:
        raise ValueError("category_names must match categories_num")
    # Crossing hulls are only proposals. With ambiguity enabled the user selects a
    # few unmistakable link cores; graph-cut assigns the untouched original surface.
    belongings = (list(initial_belongings) if initial_belongings is not None else
                   ([0] * (len(parts) + 1) if allow_unassigned
                    else spatial_seed_belongings(parts, categories_num)))
    if len(belongings) != len(parts) + 1:
        raise ValueError("initial_belongings must include background plus every part")
    prevs = belongings.copy()
    shared = {"mode": 1, "finished": False, "cancelled": False,
              "categories_num": categories_num,
              "allow_unassigned": allow_unassigned, "category_names": category_names}

    plt = vedo.Plotter(
        title="RORA HITL 1/3 — Assign hulls; gray Ambiguous is auto-assigned later",
        pos=(80, 60), size=(1200, 850),
    )
    shared["status_actor"] = vedo.Text2D(
        selection_status_text(belongings, shared["mode"], categories_num, "", category_names),
        pos="top-left", s=0.7, bg="white", c="black", alpha=0.88,
    )
    plt.add_callback("LeftButtonPress", partial(on_left_click,  vmeshes=vmeshes, belongings=belongings, prevs=prevs, shared=shared, rand_colors=rand_colors, plt=plt))
    plt.add_callback("RightButtonPress", partial(on_right_click, vmeshes=vmeshes, belongings=belongings, prevs=prevs, shared=shared, rand_colors=rand_colors, plt=plt))
    plt.add_callback("KeyPress",        partial(on_key,        vmeshes=vmeshes, belongings=belongings, prevs=prevs, shared=shared, rand_colors=rand_colors, categories_num=categories_num, plt=plt))
    button_args = dict(shared=shared, vmeshes=vmeshes, belongings=belongings,
                       rand_colors=rand_colors, plt=plt)
    for category, x in zip(range(1, categories_num + 1), np.linspace(0.52, 0.88, categories_num)):
        colors = ("Blue", "Orange", "Green", "Violet", "Gold")
        color = colors[(category - 1) % len(colors)]
        label = f"{category}: {color} = {category_names[category - 1]}" if category_names else (
            "Select 1 / root" if category == 1 else f"Select {category}")
        plt.add_button(partial(set_selection_mode, mode=category, **button_args),
                       states=(label,), pos=(float(x), 0.12), size=13)
    if allow_unassigned:
        plt.add_button(partial(set_selection_mode, mode=0, **button_args),
                       states=("Ambiguous / auto later",), pos=(0.30, 0.12), size=13)
    plt.add_button(partial(reset_part_selection, **button_args), states=("Reset",), pos=(0.68, 0.06), size=14)
    plt.add_button(partial(finish_part_selection, **button_args), states=("Save & continue",), pos=(0.84, 0.06), size=14)
    recolor_parts(vmeshes, belongings, shared["mode"], rand_colors)
    plt.show([*vmeshes, shared["status_actor"]], interactive=True)
    if not shared["finished"]:
        raise RuntimeError("Part selection was closed without a valid Save parts action")
    
    categories = [[] for _ in range(categories_num + 1)]
    for i in range(1, len(belongings)):
        categories[belongings[i]].append(parts[i - 1])
        
    return categories, belongings, rand_colors


# =============================================================================
# Vector selection (per-RLP)
# =============================================================================
def vector_status_text(rlp):
    candidate_stage = rlp.get("ui_stage") == "candidate_selection"
    joint = (f"{rlp.get('parent_name', rlp['a']['parent'])} -> "
             f"{rlp.get('child_name', rlp['a']['child'])}")
    position = f"joint {rlp.get('joint_index', 1)}/{rlp.get('joint_count', 1)}"
    lines = ([f"STEP 2/3 — AXIS & TYPE | {position} | {joint}",
              "1) Click one numbered gray axis  2) Choose Revolute/Prismatic  3) Confirm",
              "Selected axis turns yellow. Original PLY is translucent white.", "Candidates:"]
             if candidate_stage else
             [f"STEP 3/3 — DIRECTION & LIMITS | {position} | {joint}",
             "Red=parent, blue=child, green=child descendants, yellow=axis",
             "Angles are relative to the captured pose (captured pose = 0)",
              "Right-click surface=set center | Move: W/S X, D/A Y, Space/Z Z | ,/. step",
              "Rotate axis: I/U, J/H, N/B",
              "1) Preview motion  2) Confirm direction  3) Confirm limits  4) Finish & save",
              "Candidates:"])
    if rlp.get("axis_deferred_to_multistate"):
        lines.append("Axis: deferred to multi-state RGB-D motion fitting")
    if "motion_limits" in rlp:
        lines.append(f"Collision-safe preview: {rlp['motion_limits'][0]:.0f} .. {rlp['motion_limits'][1]:.0f}")
    lines.append(f"Center move step: {1000 * rlp.get('translation_step_m', 0.001):.3f} mm")
    selected = []
    for idx, vector in enumerate(rlp.get("vectors", [])):
        n = normalize(np.asarray(vector.get("n", [0, 0, 0]), dtype=float))
        state = vector.get("state", "None")
        source = vector.get("source", vector.get("method", "rora"))
        lines.append(f"  {idx}: {source}  n=[{n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}]  {state}")
        if state in ("Revolute", "Prismatic"):
            selected.append((idx, state))
    types = {state for _, state in selected}
    if not selected:
        result = "Output: no joint"
    elif len(types) > 1:
        result = "Output: INVALID mixed joint types"
    elif len(selected) > 1:
        result = f"Output: {'Spherical' if selected[0][1] == 'Revolute' else 'Planar'} (multiple axes)"
    else:
        result = f"Output: {selected[0][1]} using candidate {selected[0][0]}"
    picked = [v for v in rlp.get("vectors", []) if v.get("state") in ("Revolute", "Prismatic")]
    if len(picked) == 1:
        center = np.asarray(picked[0].get("center", rlp.get("center", [0, 0, 0])), float)
        result += (f"\norigin=[{center[0]:+.6f}, {center[1]:+.6f}, {center[2]:+.6f}] m")
        unit = "mm" if picked[0]["state"] == "Prismatic" else "deg"
        limits = picked[0].get(f"limits_{unit}", [-50.0, 50.0] if unit == "mm" else [-120.0, 120.0])
        direction = "YES" if picked[0].get("positive_direction_confirmed") else "NO"
        limits_ok = "YES" if picked[0].get("limits_confirmed") else "NO"
        result += (f"\nPositive direction confirmed: {direction}  "
                   f"limits=[{limits[0]:.0f}, {limits[1]:.0f}] {unit} confirmed: {limits_ok}")
        if picked[0].get("static_shell_collision_override_confirmed"):
            result += "\nStatic-shell overlap explicitly accepted: YES"
    if rlp.get("limit_validation_message"):
        result += f"\n{rlp['limit_validation_message']}"
    action = ("Click Confirm axis & type to continue" if candidate_stage
              else "Finish & save writes joint_selection.json after all joints")
    return "\n".join([*lines, "", result, action])


def selected_vector(rlp):
    picked = [v for v in rlp.get("vectors", []) if v.get("state") in ("Revolute", "Prismatic")]
    return picked[0] if len(picked) == 1 else None


def set_selected_pose(widget, _event, *, rlp, component, idx2actor, vec_actors,
                      seg_len, radius, status_actor, plt):
    """Numerical origin XYZ and axis yaw/pitch sliders share one update path."""
    vector = selected_vector(rlp)
    if vector is None:
        return
    center = np.asarray(vector.get("center", rlp.get("center", [0, 0, 0])), float)
    axis = normalize(np.asarray(vector["n"], float))
    if component in (0, 1, 2):
        center[component] = float(widget.value)
        vector["center"] = center.tolist(); rlp["center"] = center.tolist()
    else:
        yaw = np.arctan2(axis[1], axis[0])
        pitch = np.arcsin(np.clip(axis[2], -1.0, 1.0))
        if component == "yaw": yaw = np.deg2rad(float(widget.value))
        else: pitch = np.deg2rad(float(widget.value))
        axis = np.asarray([np.cos(pitch) * np.cos(yaw),
                           np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        vector["n"] = axis.tolist()
    vector["positive_direction_confirmed"] = False
    vector["limits_confirmed"] = False
    index = next(i for i, candidate in enumerate(rlp["vectors"]) if candidate is vector)
    if index in idx2actor:
        rebuild_tube_actor(plt, old_actor=idx2actor[index], center=center, n=axis,
                           seg_len=seg_len, radius=radius, state=vector["state"], vi=index,
                           vec_actors=vec_actors, idx2actor=idx2actor)
    status_actor.text(vector_status_text(rlp)); plt.render()


def reset_selected_pose(_widget, _event, *, rlp, idx2actor, vec_actors,
                        seg_len, radius, status_actor, plt):
    vector = selected_vector(rlp)
    if vector is None or "_edit_reset" not in vector:
        return
    vector["center"] = list(vector["_edit_reset"]["center"])
    vector["n"] = list(vector["_edit_reset"]["n"])
    vector["positive_direction_confirmed"] = False
    vector["limits_confirmed"] = False
    index = next(i for i, candidate in enumerate(rlp["vectors"]) if candidate is vector)
    rebuild_tube_actor(plt, old_actor=idx2actor[index], center=np.asarray(vector["center"]),
                       n=np.asarray(vector["n"]), seg_len=seg_len, radius=radius,
                       state=vector["state"], vi=index, vec_actors=vec_actors,
                       idx2actor=idx2actor)
    status_actor.text(vector_status_text(rlp)); plt.render()


def flip_selected_direction(_widget, _event, *, rlp, status_actor, plt, idx2actor,
                            vec_actors, seg_len, radius):
    vector = selected_vector(rlp)
    if vector is None:
        return
    vector["n"] = (-normalize(np.asarray(vector["n"], dtype=float))).tolist()
    vector["positive_direction_confirmed"] = False
    vector["limits_confirmed"] = False
    vector["static_shell_collision_override_confirmed"] = False
    rlp.pop("limit_validation_message", None)
    index = next(i for i, candidate in enumerate(rlp.get("vectors", [])) if candidate is vector)
    if index in idx2actor:
        rebuild_tube_actor(
            plt, old_actor=idx2actor[index],
            center=np.asarray(vector.get("center", rlp.get("center", [0, 0, 0])), dtype=float),
            n=np.asarray(vector["n"], dtype=float), seg_len=seg_len, radius=radius,
            state=vector["state"], vi=index, vec_actors=vec_actors, idx2actor=idx2actor,
        )
    status_actor.text(vector_status_text(rlp))
    plt.render()


def confirm_selected_direction(_widget, _event, *, rlp, status_actor, plt):
    vector = selected_vector(rlp)
    if vector is None:
        return
    vector["positive_direction_confirmed"] = True
    status_actor.text(vector_status_text(rlp))
    plt.render()


def confirm_selected_limits(_widget, _event, *, rlp, status_actor, plt):
    vector = selected_vector(rlp)
    if vector is None:
        return
    unit = "mm" if vector["state"] == "Prismatic" else "deg"
    limits = vector.get(f"limits_{unit}", [-50.0, 50.0] if unit == "mm" else [-120.0, 120.0])
    valid = bool(limits[0] < 0 < limits[1])
    if valid and rlp.get("limit_validator"):
        valid, message = rlp["limit_validator"](vector)
        rlp["limit_validation_message"] = message
        if valid:
            vector["static_shell_collision_override_confirmed"] = False
    vector["limits_confirmed"] = valid
    status_actor.text(vector_status_text(rlp))
    plt.render()


def accept_static_shell_overlap(_widget, _event, *, rlp, status_actor, plt):
    vector = selected_vector(rlp)
    if vector is None or not rlp.get("limit_validator"):
        return
    valid, message = rlp["limit_validator"](vector)
    unit = "mm" if vector["state"] == "Prismatic" else "deg"
    limits = vector.get(f"limits_{unit}", [])
    vector["static_shell_collision_override_confirmed"] = not valid
    vector["limits_confirmed"] = bool(len(limits) == 2 and limits[0] < 0 < limits[1])
    rlp["limit_validation_message"] = (message + " — HITL static-shell override"
                                        if not valid else message + " — no override needed")
    status_actor.text(vector_status_text(rlp))
    plt.render()


def set_selected_limit(widget, _event, *, rlp, index, status_actor):
    vector = selected_vector(rlp)
    if vector is None:
        return
    unit = "mm" if vector["state"] == "Prismatic" else "deg"
    limits = list(vector.get(f"limits_{unit}", [-50.0, 50.0] if unit == "mm" else [-120.0, 120.0]))
    limits[index] = float(widget.value)
    vector[f"limits_{unit}"] = limits
    vector["limits_confirmed"] = False
    vector["static_shell_collision_override_confirmed"] = False
    rlp.pop("limit_validation_message", None)
    status_actor.text(vector_status_text(rlp))


def finish_hitl(_widget, _event, *, rlp, status_actor, plt):
    vector = selected_vector(rlp)
    if vector is None:
        status_actor.text("Select exactly one joint axis first")
        plt.render()
        return
    if rlp.get("ui_stage") != "candidate_selection" and not (
            vector.get("positive_direction_confirmed") and vector.get("limits_confirmed")):
        status_actor.text("Confirm direction and limits before Finish & save")
        plt.render()
        return
    rlp["hitl_finished"] = True
    print(f"[RORA HITL] confirmed {rlp.get('ui_stage')} for "
          f"{rlp.get('parent_name', rlp['a']['parent'])} -> "
          f"{rlp.get('child_name', rlp['a']['child'])}")
    plt.close()


def defer_axis_to_multistate(_widget, _event, *, rlp, plt):
    """Keep the confirmed link pair while supplying RORA a temporary axis."""
    vectors = rlp.get("vectors", [])
    if not vectors:
        raise RuntimeError("RORA produced no temporary axis for this link pair")
    for vector in vectors:
        vector["state"] = "None"
    vectors[0]["state"] = "Revolute"
    vectors[0]["source"] = "temporary_axis_pending_multistate_fit"
    rlp["selected_candidate_index"] = 0
    rlp["candidate_joint_type"] = "Revolute"
    rlp["axis_deferred_to_multistate"] = True
    rlp["hitl_finished"] = True
    plt.close()


def reselect_joint_axis(_widget, _event, *, rlp, plt):
    rlp["hitl_reselect"] = True
    plt.close()


def preview_selected_motion(widget, _event, *, rlp, moving_actors, original_vertices, plt):
    vector = selected_vector(rlp)
    if vector is None:
        return
    axis = normalize(np.asarray(vector["n"], dtype=float))
    origin = np.asarray(vector.get("center", rlp.get("center", [0, 0, 0])), dtype=float)
    value = float(widget.value)
    for actor, vertices in zip(moving_actors, original_vertices):
        if vector["state"] == "Prismatic":
            actor.vertices = vertices + axis * value / 1000
        else:
            rotation = rotmat(axis, np.deg2rad(value))
            actor.vertices = (vertices - origin) @ rotation.T + origin
    plt.render()


def place_selected_center(event, *, rlp, plt, idx2actor, vec_actors, seg_len,
                          radius, status_actor):
    vector = selected_vector(rlp)
    point = getattr(event, "picked3d", None)
    if vector is None or point is None:
        return
    center = np.asarray(point, dtype=float)
    vector["center"] = center.tolist()
    rlp["center"] = center.tolist()
    vector["positive_direction_confirmed"] = False
    vector["limits_confirmed"] = False
    vector["static_shell_collision_override_confirmed"] = False
    index = next(i for i, candidate in enumerate(rlp["vectors"]) if candidate is vector)
    rebuild_tube_actor(
        plt, old_actor=idx2actor[index], center=center, n=vector["n"],
        seg_len=seg_len, radius=radius, state=vector["state"], vi=index,
        vec_actors=vec_actors, idx2actor=idx2actor,
    )
    status_actor.text(vector_status_text(rlp))
    plt.render()


def on_vector_click(event, *, vec_actors, rlp, click_counter, status_actor=None):
    """Select one candidate explicitly; retain legacy cycling outside HITL stages."""
    act = getattr(event, "actor", None)
    if act is None or act not in vec_actors:
        return
    st = getattr(act, "vec_state", "None")
    idx = getattr(act, "vec_idx", None)
    if idx is None:
        return

    if rlp.get("ui_stage") == "candidate_selection":
        state = rlp.get("candidate_joint_type", "Revolute")
        for actor in vec_actors:
            actor.c("gray").alpha(0.9)
            for gizmo in getattr(actor, "hinge_gizmos", []):
                gizmo.c("gray").alpha(0.18)
            actor.vec_state = "None"
            rlp["vectors"][actor.vec_idx]["state"] = "None"
        act.c("yellow" if state == "Revolute" else "blue").alpha(1.0)
        for gizmo in getattr(act, "hinge_gizmos", []):
            gizmo.c("yellow" if state == "Revolute" else "blue").alpha(1.0)
        act.vec_state = state
        rlp["vectors"][idx]["state"] = state
        rlp["selected_candidate_index"] = idx
        if status_actor is not None:
            status_actor.text(vector_status_text(rlp))
        event.plotter.render() if hasattr(event, "plotter") else None
        return

    if rlp.get("ui_stage") == "final_motion_confirmation":
        return

    if st == "None":
        act.c("yellow").alpha(1.0)
        setattr(act, "vec_state", "Revolute")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            click_counter['count'] = int(click_counter.get('count', 0)) + 1
            rlp['vectors'][idx]['state'] = "Revolute"
            rlp['vectors'][idx]['order'] = int(click_counter['count'])
    elif st == "Revolute":
        act.c("blue").alpha(1.0)
        setattr(act, "vec_state", "Prismatic")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            rlp['vectors'][idx]['state'] = "Prismatic"
    else:
        act.c("gray").alpha(0.9)
        setattr(act, "vec_state", "None")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            rlp['vectors'][idx]['state'] = "None"
            rlp['vectors'][idx]['order'] = None
    if status_actor is not None:
        status_actor.text(vector_status_text(rlp))
    event.plotter.render() if hasattr(event, "plotter") else None


def set_candidate_joint_type(_widget, _event, *, joint_type, rlp, vec_actors,
                             status_actor, plt):
    rlp["candidate_joint_type"] = joint_type
    if rlp.get("ui_stage") == "final_motion_confirmation":
        vector = selected_vector(rlp)
        if vector is not None:
            vector["state"] = joint_type
            vector.setdefault("limits_mm" if joint_type == "Prismatic" else "limits_deg",
                              [-50.0, 50.0] if joint_type == "Prismatic" else [-120.0, 120.0])
            vector["positive_direction_confirmed"] = False
            vector["limits_confirmed"] = False
            vector["static_shell_collision_override_confirmed"] = False
            for actor in vec_actors:
                actor.vec_state = joint_type
                actor.c("yellow" if joint_type == "Revolute" else "blue").alpha(1.0)
                for gizmo in getattr(actor, "hinge_gizmos", []):
                    gizmo.c("yellow" if joint_type == "Revolute" else "blue").alpha(1.0)
        status_actor.text(vector_status_text(rlp))
        plt.render()
        return
    index = rlp.get("selected_candidate_index")
    if index is not None:
        for actor in vec_actors:
            if actor.vec_idx == index:
                actor.vec_state = joint_type
                actor.c("yellow" if joint_type == "Revolute" else "blue").alpha(1.0)
                for gizmo in getattr(actor, "hinge_gizmos", []):
                    gizmo.c("yellow" if joint_type == "Revolute" else "blue").alpha(1.0)
        rlp["vectors"][index]["state"] = joint_type
    status_actor.text(vector_status_text(rlp))
    plt.render()


def hinge_gizmos(center, n, seg_len, radius, state):
    """Center marker and motion-plane ring for one hinge axis."""
    center, n = np.asarray(center, float), normalize(np.asarray(n, float))
    helper = np.asarray([1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    first = normalize(np.cross(n, helper))
    second = np.cross(n, first)
    angles = np.linspace(0.0, 2.0 * np.pi, 65)
    ring = center + 0.09 * seg_len * (
        np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    )
    color = "yellow" if state == "Revolute" else "blue" if state == "Prismatic" else "gray"
    alpha = 1.0 if state in ("Revolute", "Prismatic") else 0.18
    actors = [vedo.Sphere(center, r=3.0 * radius), vedo.Line(ring).lw(4)]
    for actor in actors:
        actor.c(color).alpha(alpha)
        if hasattr(actor, "pickable"):
            actor.pickable(False)
    return actors


def rebuild_tube_actor(plt,
                        *,
                        old_actor,
                        center: np.ndarray,
                        n: np.ndarray,
                        seg_len: float,
                        radius: float,
                        state: str,
                        vi: int,
                        vec_actors: list,
                        idx2actor: dict):
    """Recreate Tube, replace in scene, keep state & index."""
    center = np.asarray(center, dtype=float)
    half = 0.5 * seg_len * normalize(np.asarray(n, dtype=float))
    p0, p1 = center - half, center + half
    new_actor = vedo.Tube([p0, p1], r=radius, cap=True).lighting('off')
    new_actor.c("yellow" if state == "Revolute" else "blue" if state == "Prismatic" else "gray")
    new_actor.alpha(1.0 if state in ("Revolute", "Prismatic") else 0.9)
    setattr(new_actor, "vec_state", state)
    setattr(new_actor, "vec_idx", vi)
    new_actor.hinge_gizmos = hinge_gizmos(center, n, seg_len, radius, state)
    plt.add(new_actor, *new_actor.hinge_gizmos)
    plt.remove(old_actor, *getattr(old_actor, "hinge_gizmos", []))
    # sync collections
    j = vec_actors.index(old_actor) if old_actor in vec_actors else None
    if j is not None:
        vec_actors[j] = new_actor
    idx2actor[vi] = new_actor
    return new_actor


def on_rlp_key(event, *,
               plt,
               rlp: dict,
               vec_indices: List[int],
               idx2actor: dict,
               vec_actors: list,
               seg_len: float,
               radius: float,
               status_actor=None,
               angle_step_deg: float = 3.0,
               trans_step: float = 1.0):
    """
    Keyboard controls (apply to ALL vectors in this RLP window):
      Move:  W(+X), S(-X), D(+Y), A(-Y), Space/Z(+Z), X(-Z)
      Rotate: I(+Z), U(-Z), J(+Y), H(-Y), N(+X), B(-X)
      Exit: q / esc
    """
    kraw = (getattr(event, "keyPressed", None)
            or getattr(event, "key", None)
            or getattr(event, "symbol", None)
            or getattr(event, "keypress", "")
            or "")
    k = str(kraw).lower()

    if k in ("q", "esc", "escape"):
        plt.close()
        vedo.close()
        return

    if k in (",", "comma", "less"):
        rlp["translation_step_m"] = max(
            0.00005, rlp.get("translation_step_m", trans_step) / 2.0
        )
        if status_actor is not None:
            status_actor.text(vector_status_text(rlp)); plt.render()
        return
    if k in (".", "period", "greater"):
        rlp["translation_step_m"] = min(
            0.010, rlp.get("translation_step_m", trans_step) * 2.0
        )
        if status_actor is not None:
            status_actor.text(vector_status_text(rlp)); plt.render()
        return

    # translation
    trans_step = rlp.get("translation_step_m", trans_step)
    d = np.zeros(3, dtype=float)
    if   k == "w": d[:] = (+trans_step, 0.0, 0.0)
    elif k == "s": d[:] = (-trans_step, 0.0, 0.0)
    elif k == "d": d[:] = (0.0, +trans_step, 0.0)
    elif k == "a": d[:] = (0.0, -trans_step, 0.0)
    elif k in (" ", "space"): d[:] = (0.0, 0.0, +trans_step)
    elif k == "z": d[:] = (0.0, 0.0, -trans_step)

    # rotation
    R = None
    th = np.deg2rad(angle_step_deg)
    if   k == "i": R = rotmat([0,0,1], +th)
    elif k == "u": R = rotmat([0,0,1], -th)
    elif k == "j": R = rotmat([0,1,0], +th)
    elif k == "h": R = rotmat([0,1,0], -th)
    elif k == "n": R = rotmat([1,0,0], +th)
    elif k == "b": R = rotmat([1,0,0], -th)

    did_anything = False
    vecs = rlp.get("vectors", []) or []
    picked = selected_vector(rlp)
    active_indices = ([next(i for i, vector in enumerate(vecs) if vector is picked)]
                      if picked is not None else vec_indices)

    for vi in active_indices:
        if vi < 0 or vi >= len(vecs):
            continue
        v = vecs[vi]

        # update data
        if np.any(d):
            c = np.asarray(v.get("center", rlp.get("center", [0,0,0])), dtype=float)
            c2 = c + d
            v["center"] = c2.tolist()
            if "center" in rlp and isinstance(rlp["center"], (list, tuple)):
                rlp["center"] = (np.asarray(rlp["center"], dtype=float) + d).tolist()
            did_anything = True

        if R is not None and ("n" in v) and v["n"] is not None:
            n = normalize(np.asarray(v["n"], dtype=float))
            n2 = normalize(R @ n)
            v["n"] = n2.tolist()
            did_anything = True

        # update view (rebuild tube)
        if did_anything and vi in idx2actor:
            actor = idx2actor[vi]
            c_now = np.asarray(v.get("center", rlp.get("center", [0,0,0])), dtype=float)
            n_now = normalize(np.asarray(v.get("n"), dtype=float)) if v.get("n") is not None else None
            if n_now is not None and np.isfinite(n_now).all() and np.linalg.norm(n_now) > 1e-12:
                rebuild_tube_actor(
                    plt,
                    old_actor=actor,
                    center=c_now,
                    n=n_now,
                    seg_len=seg_len,
                    radius=radius,
                    state=getattr(actor, "vec_state", v.get("state", "None")),
                    vi=vi,
                    vec_actors=vec_actors,
                    idx2actor=idx2actor
                )

    if did_anything:
        plt.render()


def visualize_and_select_vectors_for_rlps(rlps: List[dict],
                                          parts: List[trimesh.Trimesh],
                                          categories: List[List[trimesh.Trimesh]],
                                          verts: np.ndarray,
                                          faces: np.ndarray,
                                          display: bool = True) -> None:
    """
    One window per RLP:
      - show base mesh + the two parts
      - draw all vectors (Tube)
      - click to toggle type
      - keyboard to move/rotate all vectors of THIS RLP
    """
    if not display:
        return

    scene_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    seg_len = scene_diag * 0.75 if np.isfinite(scene_diag) and scene_diag > 0 else 1.0
    trans_step = min(scene_diag * 0.002, 0.001) if np.isfinite(scene_diag) and scene_diag > 0 else 0.001
    radius = seg_len * 0.006

    for i, rlp in enumerate(rlps):
        rlp.setdefault("translation_step_m", trans_step)
        motion_lower, motion_upper = rlp.get("motion_limits", [-180.0, 180.0])
        mesh_actor = vedo.Mesh([verts, faces]).c("white").alpha(0.35)
        if hasattr(mesh_actor, "pickable"):
            mesh_actor.pickable(False)

        p_idx = int(rlp["a"]["parent"])
        q_idx = int(rlp["a"]["child"])
        moving_part_ids = rlp.get("moving_part_ids", [q_idx])
        part_ids = [pid for pid in (p_idx, *moving_part_ids)
                    if pid != 0 and 1 <= pid <= len(parts)]
        part_ids = list(dict.fromkeys(part_ids))

        part_actors, moving_actors, original_vertices = [], [], []
        for pid in part_ids:
            for tri in categories[pid]:
                color = ((0.85, 0.2, 0.2) if pid == p_idx else
                         (0.15, 0.45, 0.9) if pid == q_idx else
                         (0.15, 0.75, 0.55))
                actor = vedo.Mesh([tri.vertices, tri.faces]).c(color).alpha(0.82)
                if hasattr(actor, "pickable"):
                    actor.pickable(True)
                part_actors.append(actor)
                if pid in moving_part_ids:
                    moving_actors.append(actor)
                    original_vertices.append(np.asarray(tri.vertices, dtype=float).copy())

        # vector actors
        vecs = rlp.get('vectors', []) or []
        vec_actors = []
        gizmo_actors = []
        label_actors = []
        idx2actor = {}   # vi -> actor
        valid_indices = []

        for vi, v in enumerate(vecs):
            c = np.asarray(v.get("center", rlp.get("center", [0, 0, 0])), dtype=float)
            n = normalize(np.asarray(v.get("n"), dtype=float))
            if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
                continue
            half = 0.5 * seg_len * n
            p0, p1 = c - half, c + half
            st = v.get("state", "None")
            color = "yellow" if st == "Revolute" else "blue" if st == "Prismatic" else "gray"
            a = vedo.Tube([p0, p1], r=radius, cap=True).c(color).alpha(1.0).lighting('off')
            setattr(a, "vec_state", st)
            setattr(a, "vec_idx", vi)
            a.hinge_gizmos = hinge_gizmos(c, n, seg_len, radius, st)
            vec_actors.append(a)
            gizmo_actors.extend(a.hinge_gizmos)
            label = vedo.Text3D(str(vi), pos=p1, s=seg_len * 0.12, c="black")
            if hasattr(label, "pickable"):
                label.pickable(False)
            label_actors.append(label)
            idx2actor[vi] = a
            valid_indices.append(vi)

        stage = "2/3 Axis" if rlp.get("ui_stage") == "candidate_selection" else "3/3 Limits"
        plt_rlp = vedo.Plotter(
            title=(f"RORA HITL {stage} | joint {i + 1}/{len(rlps)} | "
                   f"parent {rlp.get('parent_name', p_idx)}=red, "
                   f"child {rlp.get('child_name', q_idx)}=blue"),
            axes=1, pos=(80, 60), size=(1200, 850),
        )
        status_actor = vedo.Text2D(vector_status_text(rlp), pos="top-left", s=0.7, bg="white", c="black", alpha=0.85)

        # Disable default VTK keybindings
        iren = plt_rlp.interactor
        for _ev in ("KeyPressEvent", "KeyReleaseEvent", "CharEvent"):
            iren.RemoveObservers(_ev)

        # mouse: toggle
        plt_rlp.add_callback(
            "LeftButtonPress",
            partial(on_vector_click, vec_actors=vec_actors, rlp=rlp, click_counter={'count': 0}, status_actor=status_actor)
        )
        plt_rlp.add_callback(
            "RightButtonPress",
            partial(place_selected_center, rlp=rlp, plt=plt_rlp,
                    idx2actor=idx2actor, vec_actors=vec_actors, seg_len=seg_len,
                    radius=radius, status_actor=status_actor),
        )
        # keyboard: move/rotate
        key_cb = partial(
            on_rlp_key,
            plt=plt_rlp,
            rlp=rlp,
            vec_indices=valid_indices,
            idx2actor=idx2actor,
            vec_actors=vec_actors,
            seg_len=seg_len,
            radius=radius,
            status_actor=status_actor,
            angle_step_deg=3.0,
            trans_step=trans_step
        )
        plt_rlp.add_callback("KeyPress",  key_cb)
        plt_rlp.add_callback("CharEvent", key_cb)

        if rlp.get("ui_stage") == "candidate_selection":
            rlp["candidate_joint_type"] = rlp.get("candidate_joint_type", "Revolute")
            plt_rlp.add_button(
                partial(set_candidate_joint_type, joint_type="Revolute", rlp=rlp,
                        vec_actors=vec_actors, status_actor=status_actor, plt=plt_rlp),
                states=("Type: Revolute",), pos=(0.66, 0.20), size=12,
            )
            plt_rlp.add_button(
                partial(set_candidate_joint_type, joint_type="Prismatic", rlp=rlp,
                        vec_actors=vec_actors, status_actor=status_actor, plt=plt_rlp),
                states=("Type: Prismatic",), pos=(0.84, 0.20), size=12,
            )
            plt_rlp.add_button(
                partial(finish_hitl, rlp=rlp, status_actor=status_actor, plt=plt_rlp),
                states=("Confirm axis & type",), pos=(0.84, 0.14), size=14,
            )
            preview_extent = min(60.0, max(15.0, scene_diag * 250.0))
            plt_rlp.add_slider(
                partial(preview_selected_motion, rlp=rlp, moving_actors=moving_actors,
                        original_vertices=original_vertices, plt=plt_rlp),
                -preview_extent, preview_extent, value=0,
                pos=((0.55, 0.06), (0.95, 0.06)),
                title="Preview selected revolute axis (degrees)",
            )
        else:
            vector = selected_vector(rlp) or {}
            vector.setdefault("_edit_reset", {
                "center": list(vector.get("center", rlp.get("center", [0, 0, 0]))),
                "n": list(vector.get("n", [1, 0, 0])),
            })
            unit = "mm" if vector.get("state") == "Prismatic" else "deg"
            limits = vector.get(f"limits_{unit}", [max(motion_lower, -120),
                                                     min(motion_upper, 120)])
            center = np.asarray(vector.get("center", rlp.get("center", [0, 0, 0])), float)
            axis_now = normalize(np.asarray(vector.get("n", [1, 0, 0]), float))
            lower_xyz, upper_xyz = verts.min(axis=0), verts.max(axis=0)
            for component, title, y in ((0, "origin X [m]", .42),
                                        (1, "origin Y [m]", .38),
                                        (2, "origin Z [m]", .34)):
                plt_rlp.add_slider(
                    partial(set_selected_pose, rlp=rlp, component=component,
                            idx2actor=idx2actor, vec_actors=vec_actors,
                            seg_len=seg_len, radius=radius,
                            status_actor=status_actor, plt=plt_rlp),
                    float(lower_xyz[component]), float(upper_xyz[component]),
                    value=float(center[component]), pos=((.55, y), (.95, y)), title=title,
                )
            plt_rlp.add_slider(
                partial(set_selected_pose, rlp=rlp, component="yaw", idx2actor=idx2actor,
                        vec_actors=vec_actors, seg_len=seg_len, radius=radius,
                        status_actor=status_actor, plt=plt_rlp),
                -180, 180, value=float(np.degrees(np.arctan2(axis_now[1], axis_now[0]))),
                pos=((.55, .30), (.74, .30)), title="axis yaw [deg]",
            )
            plt_rlp.add_slider(
                partial(set_selected_pose, rlp=rlp, component="pitch", idx2actor=idx2actor,
                        vec_actors=vec_actors, seg_len=seg_len, radius=radius,
                        status_actor=status_actor, plt=plt_rlp),
                -89.9, 89.9, value=float(np.degrees(np.arcsin(np.clip(axis_now[2], -1, 1)))),
                pos=((.76, .30), (.95, .30)), title="axis pitch [deg]",
            )
            plt_rlp.add_button(
                partial(reselect_joint_axis, rlp=rlp, plt=plt_rlp),
                states=("Reselect axis",), pos=(0.88, 0.26), size=14,
            )
            plt_rlp.add_button(
                partial(reset_selected_pose, rlp=rlp, idx2actor=idx2actor,
                        vec_actors=vec_actors, seg_len=seg_len, radius=radius,
                        status_actor=status_actor, plt=plt_rlp),
                states=("Reset origin/axis",), pos=(0.88, 0.22), size=12,
            )
            plt_rlp.add_button(
                partial(set_candidate_joint_type, joint_type="Revolute", rlp=rlp,
                        vec_actors=vec_actors, status_actor=status_actor, plt=plt_rlp),
                states=("Revolute",), pos=(0.56, 0.26), size=11,
            )
            plt_rlp.add_button(
                partial(set_candidate_joint_type, joint_type="Prismatic", rlp=rlp,
                        vec_actors=vec_actors, status_actor=status_actor, plt=plt_rlp),
                states=("Prismatic",), pos=(0.64, 0.26), size=11,
            )
            plt_rlp.add_button(
                partial(flip_selected_direction, rlp=rlp, status_actor=status_actor, plt=plt_rlp,
                        idx2actor=idx2actor, vec_actors=vec_actors, seg_len=seg_len, radius=radius),
                states=("Flip + direction",), pos=(0.72, 0.26), size=14,
            )
            plt_rlp.add_button(
                partial(confirm_selected_direction, rlp=rlp, status_actor=status_actor, plt=plt_rlp),
                states=("Confirm direction",), pos=(0.72, 0.20), size=14,
            )
            plt_rlp.add_button(
                partial(confirm_selected_limits, rlp=rlp, status_actor=status_actor, plt=plt_rlp),
                states=("Confirm limits",), pos=(0.72, 0.14), size=14,
            )
            plt_rlp.add_button(
                partial(accept_static_shell_overlap, rlp=rlp,
                        status_actor=status_actor, plt=plt_rlp),
                states=("Accept shell overlap",), pos=(0.72, 0.10), size=12,
            )
            plt_rlp.add_button(
                partial(finish_hitl, rlp=rlp, status_actor=status_actor, plt=plt_rlp),
                states=("Finish & save",), pos=(0.88, 0.14), size=14,
            )
            plt_rlp.add_slider(
                partial(preview_selected_motion, rlp=rlp, moving_actors=moving_actors,
                        original_vertices=original_vertices, plt=plt_rlp),
                motion_lower, motion_upper, value=0, pos=((0.55, 0.08), (0.95, 0.08)),
                title="Motion relative to captured pose (deg / mm)",
            )
            plt_rlp.add_slider(
                partial(set_selected_limit, rlp=rlp, index=0, status_actor=status_actor),
                motion_lower, 0, value=float(limits[0]), pos=((0.55, 0.05), (0.74, 0.05)), title="Observed lower",
            )
            plt_rlp.add_slider(
                partial(set_selected_limit, rlp=rlp, index=1, status_actor=status_actor),
                0, motion_upper, value=float(limits[1]), pos=((0.76, 0.05), (0.95, 0.05)), title="Observed upper",
            )

        plt_rlp.show(
            [*part_actors, *vec_actors, *gizmo_actors, *label_actors, status_actor],
            interactive=True,
        ).close()


# =============================================================================
# Visualization of boundary loops
# =============================================================================
def visualize_boundary_loops(
    ms: pymeshlab.MeshSet,
    categories: list[list[trimesh.Trimesh]],
    idx_part: int,
    loops: list[np.ndarray],
    *,
    boundary_indices: np.ndarray | None = None,
    show_original_mesh: bool = True,
    tube_radius: float = 0.4,
    sphere_radius: float = 0.8,
    boundary_point_size: float = 6.0,
    boundary_point_color = "black",
    background: str = "white",
    display: bool = True,
):
    """
    Visualize boundary loops; optionally overlay boundary vertex indices as points.
    """
    if not display:
        return

    # 0) get original mesh geometry
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)

    actors = []

    # 1) original mesh (semi-transparent)
    if show_original_mesh:
        m_orig = vedo.Mesh([verts, faces]).c("lightgray").alpha(0.25)
        m_orig.lighting("plastic")
        actors.append(m_orig)

    # 2) selected part mesh (if any)
    vpart = trimesh_list_to_vedo_mesh(categories[idx_part])
    if vpart is not None:
        vpart.c("dodgerblue").alpha(0.35).lw(0.5).lighting("plastic")
        actors.append(vpart)

    if len(loops) < 1: 
        if display: 
             plt = vedo.Plotter(bg=background, title="Boundary loops (Empty)")
             plt.show(actors, viewup="z").close()
        return

    # 3) draw loops (as tubes) + loop centers (small spheres)
    cmap = vedo.color_map(range(len(loops)), "Set1")  # distinct colors
    for i, loop in enumerate(loops):
        loop = np.asarray(loop, dtype=int).ravel()
        if loop.size == 0:
            continue
        pts = verts[loop]
        line = vedo.Line(pts, closed=False).c(cmap[i]).lw(2)
        if hasattr(line, "tube") and callable(getattr(line, "tube")):
            try:
                tube = line.tube(radius=tube_radius).c(cmap[i])
            except TypeError:
                tube = line.tube(tube_radius).c(cmap[i])
        else:
            tube = vedo.Tube(pts, r=tube_radius, cap=True).c(cmap[i])
        actors.append(tube)

        center = pts.mean(axis=0)
        s = vedo.Sphere(pos=center, r=sphere_radius, res=16).c(cmap[i]).alpha(0.5)
        actors.append(s)

    # 4) boundary indices
    if boundary_indices is not None and len(boundary_indices) > 0:
        bi = np.asarray(boundary_indices, int)
        bi = bi[(bi >= 0) & (bi < verts.shape[0])]
        if bi.size > 0:
            dots = vedo.Spheres(verts[bi], r=boundary_point_size, c="red", alpha=1.0)
            actors.append(dots)

    # 5) axes & show
    plt = vedo.Plotter(bg=background, title="Boundary loops (+ boundary points)")
    _axes_target = actors[0] if show_original_mesh else (vpart or vedo.Mesh([verts, faces]))
    try:
        axes_actor = vedo.Axes(_axes_target, axesType=4, xyGrid=True)
    except TypeError:
        try:
            axes_actor = vedo.Axes(_axes_target, xyGrid=True)
        except TypeError:
            axes_actor = vedo.Axes(_axes_target)
    plt.show(actors + [axes_actor], viewup="z").close()


def visualize_k_hop_plane(
    verts: np.ndarray,
    faces: np.ndarray,
    idx_pos: np.ndarray,
    idx_neg: np.ndarray,
    planes: list, # list of (n, d)
    center: np.ndarray,
    title: str = "k-hop neighbors and plane",
    display: bool = True
):
    """
    Visualize k-hop neighbors and the estimated plane.
    """
    if not display:
        return

    actors = []

    # 1) original mesh (semi-transparent)
    mesh = vedo.Mesh([verts, faces]).c("lightgray").alpha(0.2)
    actors.append(mesh)

    # 2) Positive and negative neighbor vertices
    if idx_pos.size > 0:
        pos_pts = vedo.Spheres(verts[idx_pos], r=0.005, c="lightblue", res=8).alpha(0.6)
        actors.append(pos_pts)
    if idx_neg.size > 0:
        neg_pts = vedo.Spheres(verts[idx_neg], r=0.005, c="salmon", res=8).alpha(0.6)
        actors.append(neg_pts)

    # 3) Estimated plane(s)
    plane_colors = ["green", "cyan", "magenta", "yellow"]
    for i, (n, d) in enumerate(planes):
        plane_pos = center - (np.dot(center, n) + d) * n
        plane_actor = vedo.Plane(pos=plane_pos, normal=n, s=(1.0, 1.0)).c(plane_colors[i % len(plane_colors)]).alpha(0.5)
        actors.append(plane_actor)

    # 4) Show plot
    plt = vedo.Plotter(bg="white", title=title)
    _axes_target = mesh
    try:
        axes_actor = vedo.Axes(_axes_target, axesType=4, xyGrid=True)
    except TypeError:
        try:
            axes_actor = vedo.Axes(_axes_target, xyGrid=True)
        except TypeError:
            axes_actor = vedo.Axes(_axes_target)
    plt.show(actors + [axes_actor], viewup="z").close()
