"""Tkinter-based parameter tuning GUI backed by YAML (used by d435_test.py)."""

from __future__ import annotations

import copy
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import yaml

# Nested defaults (must match params.yaml schema).
DEFAULT_PARAMS = {
    "canny": {
        "canny_low": 19,
        "canny_high": 69,
        "clahe_clip_limit": 6.2,
    },
    "hough": {
        "hough_threshold": 131,
        "hough_min_line_length": 142,
        "gap_ref_px": 28,
        "gap_min_px": 105,
        "gap_max_px": 206,
        "pair_min_gap_inches_x10": 10,
        "pair_max_gap_inches_x10": 35,
        "exclude_background_hough_lines": 1,
    },
    "depth": {
        "min_depth_cm": 4,
        "max_depth_cm": 200,
        "green_threshold_cm": 20,
        "depth_bin_mm": 20,
        "min_valid_percent": 5,
        "min_majority_percent": 18,
        "max_background_percent": 60,
    },
    "camera": {
        "visual_preset": 4,
        "emitter_enabled": 1,
        "laser_power": 180,
        "auto_exposure": 1,
        "exposure": 8500,
        "gain": 16,
        "spatial_magnitude": 1,
        "spatial_alpha_x100": 35,
        "spatial_delta": 30,
        "temporal_alpha_x100": 25,
        "temporal_delta": 30,
    },
}

# section, storage key in YAML, UI label, min, max, resolution, "int" | "float"
PARAM_DEFS: list[dict] = [
    {"section": "canny", "key": "canny_low", "label": "Canny Low", "min": 0, "max": 255, "res": 1, "kind": "int"},
    {"section": "canny", "key": "canny_high", "label": "Canny High", "min": 0, "max": 255, "res": 1, "kind": "int"},
    {
        "section": "canny",
        "key": "clahe_clip_limit",
        "label": "CLAHE clip limit",
        "min": 0.1,
        "max": 10.0,
        "res": 0.1,
        "kind": "float",
    },
    {"section": "hough", "key": "hough_threshold", "label": "Hough Thresh", "min": 1, "max": 300, "res": 1, "kind": "int"},
    {"section": "hough", "key": "hough_min_line_length", "label": "Min Line Len", "min": 1, "max": 400, "res": 1, "kind": "int"},
    {"section": "hough", "key": "gap_ref_px", "label": "Gap Ref Px", "min": 1, "max": 200, "res": 1, "kind": "int"},
    {"section": "hough", "key": "gap_min_px", "label": "Gap Min Px", "min": 1, "max": 200, "res": 1, "kind": "int"},
    {"section": "hough", "key": "gap_max_px", "label": "Gap Max Px", "min": 1, "max": 300, "res": 1, "kind": "int"},
    {
        "section": "hough",
        "key": "pair_min_gap_inches_x10",
        "label": "Pair Min (x0.1 in)",
        "min": 0,
        "max": 100,
        "res": 1,
        "kind": "int",
    },
    {
        "section": "hough",
        "key": "pair_max_gap_inches_x10",
        "label": "Pair Max (x0.1 in)",
        "min": 0,
        "max": 200,
        "res": 1,
        "kind": "int",
    },
    {
        "section": "hough",
        "key": "exclude_background_hough_lines",
        "label": "Exclude BG Hough",
        "min": 0,
        "max": 1,
        "res": 1,
        "kind": "int",
    },
    {"section": "depth", "key": "min_depth_cm", "label": "Min Depth cm", "min": 1, "max": 500, "res": 1, "kind": "int"},
    {"section": "depth", "key": "max_depth_cm", "label": "Max Depth cm", "min": 1, "max": 500, "res": 1, "kind": "int"},
    {"section": "depth", "key": "green_threshold_cm", "label": "Green cm", "min": 0, "max": 500, "res": 1, "kind": "int"},
    {"section": "depth", "key": "depth_bin_mm", "label": "Depth Bin mm", "min": 1, "max": 100, "res": 1, "kind": "int"},
    {"section": "depth", "key": "min_valid_percent", "label": "Min Valid %", "min": 0, "max": 100, "res": 1, "kind": "int"},
    {"section": "depth", "key": "min_majority_percent", "label": "Min Majority %", "min": 0, "max": 100, "res": 1, "kind": "int"},
    {"section": "depth", "key": "max_background_percent", "label": "Max Background %", "min": 0, "max": 100, "res": 1, "kind": "int"},
    {"section": "camera", "key": "visual_preset", "label": "Visual Preset", "min": 0, "max": 5, "res": 1, "kind": "int"},
    {"section": "camera", "key": "emitter_enabled", "label": "Emitter", "min": 0, "max": 1, "res": 1, "kind": "int"},
    {"section": "camera", "key": "laser_power", "label": "Laser Power", "min": 0, "max": 360, "res": 1, "kind": "int"},
    {"section": "camera", "key": "auto_exposure", "label": "Auto Exposure", "min": 0, "max": 1, "res": 1, "kind": "int"},
    {"section": "camera", "key": "exposure", "label": "Exposure", "min": 1, "max": 20000, "res": 1, "kind": "int"},
    {"section": "camera", "key": "gain", "label": "Gain", "min": 1, "max": 128, "res": 1, "kind": "int"},
    {"section": "camera", "key": "spatial_magnitude", "label": "Spatial Mag", "min": 1, "max": 5, "res": 1, "kind": "int"},
    {"section": "camera", "key": "spatial_alpha_x100", "label": "Spatial A x100", "min": 1, "max": 100, "res": 1, "kind": "int"},
    {"section": "camera", "key": "spatial_delta", "label": "Spatial Delta", "min": 1, "max": 100, "res": 1, "kind": "int"},
    {"section": "camera", "key": "temporal_alpha_x100", "label": "Temporal A x100", "min": 1, "max": 100, "res": 1, "kind": "int"},
    {"section": "camera", "key": "temporal_delta", "label": "Temporal Delta", "min": 1, "max": 100, "res": 1, "kind": "int"},
]

SECTION_TITLES = {
    "canny": "Canny",
    "hough": "Hough / line pairs",
    "depth": "Depth",
    "camera": "Camera",
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    for key, val in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def _clip_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))


def _clip_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


class ParamGUI:
    def __init__(self, yaml_path: str | Path) -> None:
        self.yaml_path = Path(yaml_path)
        self._params = copy.deepcopy(DEFAULT_PARAMS)
        self._load_yaml_into_params()

        self.root = tk.Tk()
        self.root.title("D435 parameter tuning")
        self.root.geometry("520x640")

        self._vars: dict[str, tk.Variable] = {}
        self._scales: dict[str, tk.Scale] = {}
        self._value_labels: dict[str, tk.Label] = {}
        self._entries: dict[str, tk.Entry] = {}
        self._row_specs: dict[str, dict] = {f"{d['section']}.{d['key']}": d for d in PARAM_DEFS}

        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Save to YAML", command=self._save_yaml).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="Load from YAML", command=self._load_from_file).pack(side=tk.LEFT)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        sections_order = ["canny", "hough", "depth", "camera"]
        for section in sections_order:
            frame = ttk.Frame(notebook, padding=6)
            notebook.add(frame, text=SECTION_TITLES[section])
            row = 0
            for spec in PARAM_DEFS:
                if spec["section"] != section:
                    continue
                self._build_row(frame, spec, row)
                row += 1

        self._sync_all_value_labels()

    def _get_storage_value(self, section: str, key: str) -> float | int:
        return self._params[section][key]

    def _set_storage_value(self, section: str, key: str, value: float | int) -> None:
        self._params[section][key] = value

    def _load_yaml_into_params(self) -> None:
        self._params = copy.deepcopy(DEFAULT_PARAMS)
        if not self.yaml_path.is_file():
            return
        with open(self.yaml_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            _deep_merge(self._params, loaded)

    def _build_row(self, parent: ttk.Frame, spec: dict, grid_row: int) -> None:
        section = spec["section"]
        key = spec["key"]
        sid = f"{section}.{key}"
        label = ttk.Label(parent, text=spec["label"], width=18)
        label.grid(row=grid_row, column=0, sticky=tk.W, pady=2)

        val_lbl = ttk.Label(parent, text="", width=12)
        val_lbl.grid(row=grid_row, column=1, sticky=tk.W, padx=4)
        self._value_labels[sid] = val_lbl

        kind = spec["kind"]
        raw = self._get_storage_value(section, key)
        if kind == "int":
            var: tk.Variable = tk.DoubleVar(value=float(int(raw)))
        else:
            var = tk.DoubleVar(value=float(raw))
        self._vars[sid] = var

        scale = tk.Scale(
            parent,
            from_=spec["min"],
            to=spec["max"],
            orient=tk.HORIZONTAL,
            resolution=spec["res"],
            showvalue=0,
            length=220,
            variable=var,
            command=lambda _v, s=sid: self._on_scale_move(s),
        )
        scale.grid(row=grid_row, column=2, sticky=tk.EW, pady=2)
        self._scales[sid] = scale

        ent = ttk.Entry(parent, width=10)
        ent.grid(row=grid_row, column=3, padx=(6, 0), pady=2)
        ent.bind("<Return>", lambda e, s=sid: self._on_entry_commit(s))
        self._entries[sid] = ent

        parent.columnconfigure(2, weight=1)

        self._push_var_to_widgets(sid)

    def _format_display(self, spec: dict, value: float) -> str:
        if spec["kind"] == "float":
            return f"{value:.2f}"
        return str(int(round(value)))

    def _on_scale_move(self, sid: str) -> None:
        spec = self._row_specs[sid]
        var = self._vars[sid]
        raw = float(var.get())
        if spec["kind"] == "int":
            v = _clip_int(raw, int(spec["min"]), int(spec["max"]))
            var.set(float(v))
        else:
            v = _clip_float(raw, float(spec["min"]), float(spec["max"]))
            # Snap to resolution
            res = float(spec["res"])
            v = round(v / res) * res
            v = _clip_float(v, float(spec["min"]), float(spec["max"]))
            var.set(v)
        section, key = spec["section"], spec["key"]
        self._set_storage_value(section, key, int(var.get()) if spec["kind"] == "int" else var.get())
        self._value_labels[sid].config(text=self._format_display(spec, float(var.get())))
        ent = self._entries[sid]
        ent.delete(0, tk.END)
        ent.insert(0, self._format_display(spec, float(var.get())))

    def _on_entry_commit(self, sid: str) -> None:
        spec = self._row_specs[sid]
        ent = self._entries[sid]
        text = ent.get().strip()
        try:
            if spec["kind"] == "int":
                v = int(round(float(text)))
                v = _clip_int(float(v), int(spec["min"]), int(spec["max"]))
                self._vars[sid].set(float(v))
            else:
                v = float(text)
                v = _clip_float(v, float(spec["min"]), float(spec["max"]))
                res = float(spec["res"])
                v = round(v / res) * res
                v = _clip_float(v, float(spec["min"]), float(spec["max"]))
                self._vars[sid].set(v)
        except ValueError:
            self._push_var_to_widgets(sid)
            return
        self._on_scale_move(sid)

    def _push_var_to_widgets(self, sid: str) -> None:
        spec = self._row_specs[sid]
        var = self._vars[sid]
        val = float(var.get())
        self._value_labels[sid].config(text=self._format_display(spec, val))
        ent = self._entries[sid]
        ent.delete(0, tk.END)
        ent.insert(0, self._format_display(spec, val))
        section, key = spec["section"], spec["key"]
        self._set_storage_value(section, key, int(val) if spec["kind"] == "int" else val)

    def _sync_all_value_labels(self) -> None:
        for sid in self._row_specs:
            self._push_var_to_widgets(sid)

    def _gather_nested_params_from_widgets(self) -> dict:
        nested = copy.deepcopy(DEFAULT_PARAMS)
        for spec in PARAM_DEFS:
            sid = f"{spec['section']}.{spec['key']}"
            var = self._vars[sid]
            val = float(var.get())
            if spec["kind"] == "int":
                nested[spec["section"]][spec["key"]] = int(round(val))
            else:
                nested[spec["section"]][spec["key"]] = val
        return nested

    def _save_yaml(self) -> None:
        data = self._gather_nested_params_from_widgets()
        self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    def _load_from_file(self) -> None:
        self._load_yaml_into_params()
        for spec in PARAM_DEFS:
            sid = f"{spec['section']}.{spec['key']}"
            raw = self._get_storage_value(spec["section"], spec["key"])
            if spec["kind"] == "int":
                self._vars[sid].set(float(int(raw)))
            else:
                self._vars[sid].set(float(raw))
            self._push_var_to_widgets(sid)

    def update(self) -> None:
        self.root.update()

    def get_all_params(self) -> dict:
        """Same keys as the merged dict from the former get_*_runtime_params() helpers."""
        p = self._gather_nested_params_from_widgets()
        c = p["canny"]
        h = p["hough"]
        d = p["depth"]
        cam = p["camera"]

        canny_low = int(max(0, min(254, int(c["canny_low"]))))
        canny_high = int(max(canny_low + 1, min(255, int(c["canny_high"]))))
        clahe_clip_limit = max(0.1, float(c["clahe_clip_limit"]))

        hough_threshold = max(1, int(h["hough_threshold"]))
        hough_min_line_length = max(1, int(h["hough_min_line_length"]))
        gap_ref_px = max(1, int(h["gap_ref_px"]))
        gap_min_px = max(1, int(h["gap_min_px"]))
        gap_max_px = max(gap_min_px, int(h["gap_max_px"]))
        pair_min_tenth_in = int(h["pair_min_gap_inches_x10"])
        pair_max_tenth_in = int(h["pair_max_gap_inches_x10"])
        pair_min_gap_m = max(0.0, pair_min_tenth_in / 10.0 / 39.3701)
        pair_max_gap_m = max(pair_min_gap_m, pair_max_tenth_in / 10.0 / 39.3701)

        min_depth_cm = int(d["min_depth_cm"])
        max_depth_cm = int(d["max_depth_cm"])
        green_cm = int(d["green_threshold_cm"])
        min_depth_m = max(0.01, min_depth_cm / 100.0)
        max_depth_m = max(min_depth_m + 0.01, max_depth_cm / 100.0)
        green_threshold_m = min(max(green_cm / 100.0, min_depth_m), max_depth_m)
        depth_bin_mm = max(1, int(d["depth_bin_mm"]))
        min_valid_percent = int(d["min_valid_percent"])
        min_majority_percent = int(d["min_majority_percent"])
        max_background_percent = int(d["max_background_percent"])
        min_valid_ratio = min(1.0, max(0.0, min_valid_percent / 100.0))
        min_majority_ratio = min(1.0, max(0.0, min_majority_percent / 100.0))
        max_background_ratio = min(1.0, max(0.0, max_background_percent / 100.0))

        return {
            "canny_low": canny_low,
            "canny_high": canny_high,
            "clahe_clip_limit": clahe_clip_limit,
            "hough_threshold": hough_threshold,
            "hough_min_line_length": hough_min_line_length,
            "gap_ref_px": gap_ref_px,
            "gap_min_px": gap_min_px,
            "gap_max_px": gap_max_px,
            "pair_min_gap_m": pair_min_gap_m,
            "pair_max_gap_m": pair_max_gap_m,
            "exclude_background_hough_lines": bool(
                int(h["exclude_background_hough_lines"])
            ),
            "min_depth_m": float(min_depth_m),
            "max_depth_m": float(max_depth_m),
            "green_threshold_m": float(green_threshold_m),
            "min_depth_mm": int(round(min_depth_m * 1000.0)),
            "max_depth_mm": int(round(max_depth_m * 1000.0)),
            "depth_bin_mm": int(depth_bin_mm),
            "min_valid_ratio": float(min_valid_ratio),
            "min_majority_ratio": float(min_majority_ratio),
            "max_background_ratio": float(max_background_ratio),
            "visual_preset": int(cam["visual_preset"]),
            "emitter_enabled": int(cam["emitter_enabled"]),
            "laser_power": int(cam["laser_power"]),
            "auto_exposure": int(cam["auto_exposure"]),
            "exposure": max(1, int(cam["exposure"])),
            "gain": max(1, int(cam["gain"])),
            "spatial_magnitude": max(1, int(cam["spatial_magnitude"])),
            "spatial_alpha": max(0.01, int(cam["spatial_alpha_x100"]) / 100.0),
            "spatial_delta": max(1, int(cam["spatial_delta"])),
            "temporal_alpha": max(0.01, int(cam["temporal_alpha_x100"]) / 100.0),
            "temporal_delta": max(1, int(cam["temporal_delta"])),
        }
