# -*- coding: utf-8 -*-

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MplPath
from PIL import Image
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, cKDTree
from tkinter import colorchooser, messagebox, ttk
from multiprocessing import Pool, cpu_count

# -------------------------------------------------
#  Suppression du FutureWarning lié à interpolate
# -------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------
#  Import du gestionnaire de base (db.py)
# -------------------------------------------------
#sys.path.append(str(Path(r"R:\2 - Surveillance\surveillance_app\database").resolve()))
from database.db import DatabaseManager  # classe déjà implémentée


class AppConfig:
    def __init__(self, path="config.json"):
        self.path = Path(path)
        if self.path.exists():
            self.load()
        else:
            self.set_defaults()
            self.save()

    def set_defaults(self):
        self.DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
        self.IMG_PATH = Path(r"R:\2 - Surveillance\surveillance_app\.Donnees\Plan systemes\Fond de plan.png")
        self.M1_IMG_PATH = Path(r"R:\2 - Surveillance\surveillance_app\.Donnees\Plan systemes\Fond M1.png")

        self.REAL_BOUNDS = [0.0, 254.39, 0.0, 168.88]
        self.M1_BOUNDS = [78.38, 158.95, 64.83, 158.40]

        self.DEFAULT_PRECISION = 0.1

        self.INTERPOLATION_METHOD = "IDW"
        self.TRIANGULATION_METHOD = "IDW"
        self.IDW_POWER = 5.0

        self.INTERPOLATION_RADIUS = 50.0
        self.MAX_NEIGHBORS = 8

        self.COLORMAP_SINGLE = "viridis"
        self.COLORMAP_COMPARE = "seismic"

        self.AVAILABLE_COLORMAPS = ["viridis", "seismic", "nipy_spectral"]

        self.GRID_MAX_POINTS = 6_000_000
        self.IDW_NEIGHBORS = 8

        self.SYMMETRIC_COLORBAR = False
        
        self.CURVE_POINT_FREQUENCY = "1D"

    def load(self):
        data = json.loads(self.path.read_text())
        self.set_defaults()
        for k, v in data.items():
            setattr(self, k, v)
        self.DB_PATH = Path(self.DB_PATH)
        self.IMG_PATH = Path(self.IMG_PATH)

    def save(self):
        data = {}
        for k, v in self.__dict__.items():
            if k == "path":
                continue
            data[k] = str(v) if isinstance(v, Path) else v
        self.path.write_text(json.dumps(data, indent=4))

def compute_matrix_task(args):

    (
        sys_id,
        date0,
        date,
        precision,
        config_dict,
        fill_before,
        fill_between,
        fill_after,
        active_layers,
        matrix_unit,
    ) = args

    db = DatabaseManager()
    try:
        repo = DataRepository(db)
        controller = MappingController(repo)

        processor = TimeSeriesProcessor(
            fill_before=fill_before,
            fill_between=fill_between,
            fill_after=fill_after,
        )

        triangulator = TriangulationEngine(config_dict["REAL_BOUNDS"])

        sensors, diff_dict, units = controller.compute(
            sys_id,
            date0,
            date,
            processor,
        )

        # filtrage calques
        if active_layers:
            filtered = []
            for s in sensors:
                layer = s.get("layer") if s.get("layer") else "Sans calque"
                if (sys_id, layer) in active_layers:
                    filtered.append(s)

            allowed_ids = {s["id"] for s in filtered}

            sensors = filtered
            diff_dict = {
                sid: val for sid, val in diff_dict.items()
                if sid in allowed_ids
            }

        if not sensors or not diff_dict:
            return sys_id, date, None

        # filtrage unité: une matrice = une unité (types mélangés autorisés)
        target_unit = matrix_unit
        grouped_sensors = []
        grouped_diff = {}
        for s in sensors:
            sid = s["id"]
            if sid not in diff_dict:
                continue
            sensor_unit = repo.get_sensor_unit(sid)
            if sensor_unit == target_unit:
                grouped_sensors.append(s)
                grouped_diff[sid] = diff_dict[sid]

        if not grouped_sensors or not grouped_diff:
            return sys_id, date, None

        result = triangulator.compute(
            grouped_sensors,
            grouped_diff,
            precision,
            config_dict,
        )

        return sys_id, date, result
    finally:
        try:
            db.close()
        except Exception:
            pass

# -------------------------------------------------
#  Fonctions de traitement temporel (utilisées dans le script)
# -------------------------------------------------
def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolation linéaire en fonction du temps (index = DatetimeIndex)."""
    if df.empty:
        return df
    df['value'] = df['value'].interpolate(method='time')
    return df


def subtract_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Soustrait la première valeur non nulle (baseline)."""
    if df.empty:
        return df
    first_valid = df['value'].first_valid_index()
    if first_valid is None:
        return df
    baseline = df.at[first_valid, 'value']
    df['value'] = df['value'] - baseline
    return df


def daily_average(df: pd.DataFrame) -> pd.DataFrame:
    """Moyenne journalière (UTC). Retourne un DataFrame avec un DatetimeIndex."""
    if df.empty:
        return df
    return df.resample('D').mean()


def analyze_sensor_type(sensors):
    types = [s.get("type") for s in sensors if s.get("type") not in (None, "", " ")]
    if not types:
        return None
    unique = set(types)
    return unique.pop() if len(unique) == 1 else None


UNIT_CONVERSION = {
    "nm": 1e-9,
    "µm": 1e-6,
    "um": 1e-6,
    "mm": 1e-3,
    "cm": 1e-2,
    "dm": 1e-1,
    "m": 1.0,
    "km": 1e3,
}


def normalize_units(values, units):
    converted = []
    if len(values) != len(units):
        return None, None
    for v, u in zip(values, units):
        if u not in UNIT_CONVERSION:
            return None, None
        converted.append(v * UNIT_CONVERSION[u])
    return converted, "m"


def analyze_units(units):
    units = [u for u in units if u not in (None, "", " ")]
    if not units:
        return ""
    unique = set(units)
    if len(unique) == 1:
        return unique.pop()

    prefixes = {
        "mm": ("m", 1e-3),
        "cm": ("m", 1e-2),
        "dm": ("m", 1e-1),
        "m": ("m", 1),
        "km": ("m", 1e3),
        "µm": ("m", 1e-6),
        "nm": ("m", 1e-9),
    }
    base_units = {prefixes[u][0] if u in prefixes else u for u in unique}
    return base_units.pop() if len(base_units) == 1 else "unités mixtes"


def choose_best_si_unit(values):
    values = np.asarray(values)
    if values.size == 0:
        return 1.0, "m"
    max_val = np.nanmax(np.abs(values))
    if max_val < 1e-6:
        return 1e9, "nm"
    if max_val < 1e-3:
        return 1e6, "µm"
    if max_val < 1:
        return 1e3, "mm"
    if max_val < 100:
        return 1e2, "cm"
    return 1.0, "m"


def normalize_diff_with_units(diff_dict, unit_by_sensor):
    """
    Normalise les valeurs si possible.

    - Si toutes les unités sont SI → conversion en m
    - Si unités absentes → on laisse tel quel
    - Si unités non SI identiques → on laisse tel quel
    - Si unités incompatibles → erreur
    """

    if not diff_dict:
        return {}, ""

    normalized = {}
    non_si_units = set()
    missing_units = False

    for sid, value in diff_dict.items():
        unit = unit_by_sensor.get(sid)

        if unit in (None, "", " "):
            missing_units = True
            normalized[sid] = value
            continue

        if unit in UNIT_CONVERSION:
            normalized[sid] = value * UNIT_CONVERSION[unit]
        else:
            non_si_units.add(unit)
            normalized[sid] = value

    # Cas SI + non SI incompatible
    if non_si_units and any(unit_by_sensor.get(sid) in UNIT_CONVERSION for sid in diff_dict.keys()):
        return None, None

    if missing_units:
        return normalized, ""

    if non_si_units:
        if len(non_si_units) != 1:
            return None, None
        return normalized, non_si_units.pop()

    return normalized, "m"


def load_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image de fond introuvable : {path}")
    img = Image.open(path).convert("RGBA")
    return img, img.width, img.height


class DataRepository:
    def __init__(self, db_manager):
        self.db = db_manager

    def get_systems(self):
        rows = self.db.get_systems()
        return [{"id": r["id"], "name": r["name"]} for r in rows]

    def get_sensors(self, system_id):
        rows = self.db.get_sensors_by_system(system_id)
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "x": r["x"],
                "y": r["y"],
                "layer": r["layer"],
                "type": r["type"],
                "unit": r["unit"],
            }
            for r in rows
        ]

    def get_measurements(self, sensor_id):
        return self.db.get_measurements_by_sensor(sensor_id)

    def get_sensor_unit(self, sensor_id):
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT unit FROM sensors WHERE id = ?", (sensor_id,))
        row = cursor.fetchone()
        if not row:
            return None

        unit = row["unit"]
        if unit in ("C", "°C", "degC", "celsius"):
            return "°C"
        return unit

    def get_dates_of_system(self, system_id):
        dates = set()
        sensors = self.get_sensors(system_id)
        for sensor in sensors:
            rows = self.get_measurements(sensor["id"])
            dates.update({pd.to_datetime(row[0]).date() for row in rows})
        return dates

    def get_common_dates(self, system_ids):
        common = None
        for sys_id in system_ids:
            sys_dates = self.get_dates_of_system(sys_id)
            if not sys_dates:
                continue
            common = sys_dates if common is None else common.intersection(sys_dates)
        return sorted(common) if common else []


class TimeSeriesProcessor:
    def __init__(self, fill_before=False, fill_between=True, fill_after=False):
        self.fill_before = fill_before
        self.fill_between = fill_between
        self.fill_after = fill_after

    def prepare_series(self, rows):
        df = pd.DataFrame(rows, columns=["datetime", "value", "unit"])
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["datetime", "value"]).set_index("datetime").sort_index()
        return df.resample("D")["value"].mean()

    def _apply_fill(self, series):
        if series.empty:
            return series
        if self.fill_between:
            series = series.interpolate("linear")
        if self.fill_before:
            first = series.first_valid_index()
            if first is not None:
                series.loc[series.index < first] = series.loc[first]
        if self.fill_after:
            last = series.last_valid_index()
            if last is not None:
                series.loc[series.index > last] = series.loc[last]
        return series


class TriangulationEngine:
    def __init__(self, bounds):
        self.bounds = bounds

    @staticmethod
    def _coerce_scalar(value):
        """Retourne une valeur scalaire si la config contient une liste/tuple imbriqué."""
        if isinstance(value, (list, tuple)) and not value:
            return None
        while isinstance(value, (list, tuple)) and value:
            value = value[0]
        return value

    def compute(self, sensors, diff_dict, step, settings):
        x_min, x_max, y_min, y_max = settings["REAL_BOUNDS"]

        xs, ys, vals, used_ids = [], [], [], set()
        for s in sensors:
            sid = s["id"]
            if sid not in diff_dict:
                continue
            x, y, v = s["x"], s["y"], diff_dict[sid]
            if (
                x is None
                or y is None
                or v is None
                or np.isnan(x)
                or np.isnan(y)
                or np.isnan(v)
                or np.isinf(x)
                or np.isinf(y)
                or np.isinf(v)
            ):
                continue
            xs.append(x)
            ys.append(y)
            vals.append(v)
            used_ids.add(sid)

        if len(xs) < 3:
            return None

        try:
            step = float(self._coerce_scalar(step))
        except (TypeError, ValueError):
            raise ValueError("La précision (step) est invalide")
        if step <= 0:
            raise ValueError("La précision (step) doit être > 0")

        points = np.column_stack([xs, ys])
        values = np.array(vals)

        nx = max(2, int((x_max - x_min) / step))
        ny = max(2, int((y_max - y_min) / step))

        try:
            grid_max_points = int(float(self._coerce_scalar(settings.get("GRID_MAX_POINTS", 6_000_000))))
        except (TypeError, ValueError):
            grid_max_points = 6_000_000
        if nx * ny > grid_max_points:
            raise ValueError(f"Grille trop grande ({nx*ny} points). Augmenter la précision.")

        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(xi, yi)

        method = str(self._coerce_scalar(settings.get("TRIANGULATION_METHOD", "IDW"))).upper()

        if method == "DELAUNAY":
            Z = griddata(points, values, (X, Y), method="linear")
        elif method == "IDW":
            try:
                power = float(self._coerce_scalar(settings.get("IDW_POWER", 2.0)))
            except (TypeError, ValueError):
                power = 2.0
            try:
                neighbors = int(float(self._coerce_scalar(settings.get("IDW_NEIGHBORS", 8))))
            except (TypeError, ValueError):
                neighbors = 8
            neighbors = max(1, min(neighbors, len(points)))

            grid_points = np.column_stack([X.ravel(), Y.ravel()])
            tree = cKDTree(points)
            dist, idx = tree.query(grid_points, k=neighbors)

            if neighbors == 1:
                dist = dist[:, np.newaxis]
                idx = idx[:, np.newaxis]

            weights = 1.0 / (dist**power + 1e-12)
            weighted_vals = weights * values[idx]
            Z = np.sum(weighted_vals, axis=1) / np.sum(weights, axis=1)
            Z = Z.reshape(X.shape)
        else:
            raise ValueError("Méthode interpolation inconnue")

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_path = MplPath(hull_points)

        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        mask = hull_path.contains_points(grid_points).reshape(X.shape)
        Z[~mask] = np.nan

        return {"X": X, "Y": Y, "Z": Z, "used_sensor_ids": used_ids}


class MappingController:
    def __init__(self, repository):
        self.repo = repository

    def compute(self, system_id, ref_date, cmp_date, processor):
        sensors = self.repo.get_sensors(system_id)

        diff_dict = {}
        units = []

        ref_ts = pd.Timestamp(ref_date)
        cmp_ts = pd.Timestamp(cmp_date)

        for sensor in sensors:
            rows = self.repo.get_measurements(sensor["id"])
            if not rows:
                continue

            unit = rows[0][2]
            if unit:
                units.append(unit)

            series = processor.prepare_series(rows)
            if series.empty:
                continue

            full_range = pd.date_range(ref_ts, cmp_ts, freq="D")
            series = processor._apply_fill(series.reindex(full_range))

            if ref_ts not in series.index or cmp_ts not in series.index:
                continue

            val_ref = series.loc[ref_ts]
            val_cmp = series.loc[cmp_ts]

            if pd.isna(val_ref) or pd.isna(val_cmp):
                continue

            diff_dict[sensor["id"]] = val_cmp - val_ref

        return sensors, diff_dict, units


class MultiSystemMappingApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self._triangulation_cache = {}
        
        self._update_job = None
        self._update_delay = 200  # millisecondes

        self._triangulation_cache = {}
        
        self.title("Mapping des systèmes de surveillance")
        self.geometry("1200x800")

        self.config_app = AppConfig()
        self._m1_active = False

        try:
            self.bg_image, self.img_w, self.img_h = load_image(self.config_app.IMG_PATH)
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self.destroy()
            return
        
        try:
            self.m1_image, self.m1_w, self.m1_h = load_image(self.config_app.M1_IMG_PATH)
        except Exception as e:
            self.m1_image = None
            print("Image M1 non chargée:", e)

        self.db = DatabaseManager()
        self.repository = DataRepository(self.db)
        self.controller = MappingController(self.repository)
        self.triangulator = TriangulationEngine(self.config_app.REAL_BOUNDS)

        self.systems = self.repository.get_systems()
        self.system_vars = {}
        self.system_colors = {}
        self.precision = tk.DoubleVar(value=self.config_app.DEFAULT_PRECISION)
        self.precision.trace_add("write", lambda *_: self._on_precision_change())

        self.mode_var = tk.StringVar(value="single")

        self.fill_before_var = tk.BooleanVar(value=False)
        self.fill_between_var = tk.BooleanVar(value=True)
        self.fill_after_var = tk.BooleanVar(value=False)

        self.ref_date_var = tk.StringVar()
        self.cmp_date_var = tk.StringVar()

        self.layer_vars = {}
        self.layer_frame = None

        self._build_sidebar()
        self._build_canvas()
        self._draw_background()
        self.update_idletasks()
        self.canvas.draw()

        self._is_zoomed_m1 = False

    def notify(self, title, message="", level="warn", duration=5000):
        styles = {
            "info":  ("#e8f4ff", "#004a99", "ℹ "),
            "warn":  ("#fff4e5", "#9c5700", "⚠ "),
            "error": ("#ffeaea", "#990000", "✖ "),
            "ok":    ("#e8ffe8", "#006600", "✔ "),
        }

        bg, fg, icon = styles.get(level, styles["warn"])

        text = f"{icon}{title}"
        if message:
            text += f" : {message}"

        self.status_bar.config(
            bg=bg,
            fg=fg,
            wraplength=230
        )

        self.status_var.set(text)

        if duration:

            if hasattr(self, "_status_job"):
                self.after_cancel(self._status_job)

            self._status_job = self.after(
                duration,
                lambda: self.status_var.set("Prêt")
            )

    def _build_canvas(self):
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        gs = GridSpec(1, 2, width_ratios=[20, 1], figure=self.fig)
        self._cbar_spec = gs[1]

        self.ax = self.fig.add_subplot(gs[0])
        self.cbar_ax = self.fig.add_subplot(gs[1])

        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for spine in self.ax.spines.values():
            spine.set_visible(True)

        self.cbar_ax.axis("off")

        self._interp_image = None
        self._cbar = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas.get_tk_widget().config(cursor="crosshair")
        self.canvas.mpl_connect("button_press_event", self._on_map_click)

        self._hover_annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            bbox=dict(
                boxstyle="square,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.8
            )
        )

        self._hover_annotation.set_visible(False)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)



    def _on_map_click(self, event):

        if event.inaxes != self.ax:
            return

        # vérifier si on clique sur un capteur
        for scatter in getattr(self, "_sensor_artists", []):

            cont, ind = scatter.contains(event)

            if cont:
                index = ind["ind"][0]
                sensor = scatter._sensors[index]

                x = sensor["x"]
                y = sensor["y"]

                self._plot_point_timeseries(x, y)
                return

        # sinon position du clic
        x = event.xdata
        y = event.ydata

        if x is None or y is None:
            return

        self._plot_point_timeseries(x, y)

    def _interpolate_value(self, triangulation, x, y):

        X = triangulation["X"]
        Y = triangulation["Y"]
        Z = triangulation["Z"]

        dist = (X - x)**2 + (Y - y)**2
        idx = dist.argmin()

        iy, ix = np.unravel_index(idx, X.shape)

        return Z[iy, ix]

    def _get_neighbor_distances(self, x, y, n=4):
        sensors = []

        for sys_data in self.systems:

            sid = sys_data["id"]

            if self.system_vars[sid].get() == 0:
                continue

            sensors_sys = self.repository.get_sensors(sid)

            for s in sensors_sys:

                if s["x"] is None or s["y"] is None:
                    continue

                dist = np.sqrt((s["x"]-x)**2 + (s["y"]-y)**2)

                sensors.append(dist)

        if not sensors:
            return None

        sensors.sort()

        return sensors[:n]
    
    def _compute_reliability(self, x, y):

        dists = self._get_neighbor_distances(x, y)

        if not dists:
            return None

        avg_dist = np.mean(dists)

        if avg_dist < 5:
            return "élevée", "green"
        elif avg_dist < 15:
            return "moyenne", "orange"
        else:
            return "faible", "red"

    def _show_neighbor_info(self, x, y, neighbors):

        text = f"Point ({x:.2f} , {y:.2f})\n\nCapteurs voisins :\n\n"

        for dist, s in neighbors:

            text += f"{s['name']}  →  {dist:.2f} m\n"

        win = tk.Toplevel(self)
        win.title("Voisins interpolation")

        txt = tk.Text(win, width=35, height=10)
        txt.pack(padx=10, pady=10)

        txt.insert("1.0", text)
        txt.config(state="disabled")

    def _plot_cursor_lines(self, ax):

        if not hasattr(self, "available_dates") or not self.available_dates:
            return

        try:
            ref_date = self.available_dates[self.ref_index.get()]
            cmp_date = self.available_dates[self.cmp_index.get()]
        except Exception:
            return

        ax.axvline(ref_date, color="green", linestyle="--", linewidth=1.5)
        ax.axvline(cmp_date, color="green", linestyle="--", linewidth=1.5)


    def _on_hover(self, event):

        if event.inaxes != self.ax:
            return

        visible = False

        for scatter in getattr(self, "_sensor_artists", []):

            cont, ind = scatter.contains(event)

            if cont:

                index = ind["ind"][0]

                x, y = scatter.get_offsets()[index]

                sensor = scatter._sensors[index]
                name = sensor["name"]

                self._hover_annotation.xy = (x, y)
                self._hover_annotation.set_text(name)

                self._hover_annotation.set_visible(True)

                visible = True

                break

        if not visible:
            self._hover_annotation.set_visible(False)

        self.canvas.draw_idle()

    def _auto_update(self, *args):
        if self._update_job is not None:
            self.after_cancel(self._update_job)

        self._update_job = self.after(self._update_delay, self._run_update)

    def _run_update(self):

        selected = [sys["id"] for sys in self.systems if self.system_vars[sys["id"]].get() == 1]

        if self.mode_var.get() == "single":
            expected = 1
        elif self.mode_var.get() == "compare":
            expected = 2
        else:  # cumul
            expected = 2  # minimum

        if len(selected) < expected:
            return

        if not hasattr(self, "available_dates") or not self.available_dates:
            return

        self._fill_matrix_with_triangulation()
        self._refresh_plot()

    def _on_system_toggle(self, system_id):

        self._reset_progress_bar()

        if self.mode_var.get() == "single":

            for sid, var in self.system_vars.items():
                if sid != system_id:
                    var.set(0)

        self._refresh_plot()

        self._update_available_dates()

        self.layer_vars.clear()

        self._update_layer_options_from_selection()

        self.layer_frame.update_idletasks()

        self._auto_update()

    def _draw_background(self):

        # supprimer ancienne image
        if hasattr(self, "_background_image") and self._background_image:
            try:
                self._background_image.remove()
            except:
                pass
            self._background_image = None

        # choisir image
        if self._m1_active and self.m1_image is not None:
            img = np.flipud(self.m1_image)
        else:
            img = np.flipud(self.bg_image)

        # afficher image
        self._background_image = self.ax.imshow(
            img,
            extent=self.config_app.REAL_BOUNDS,
            origin='lower',
            zorder=-10,
            alpha=1,
        )

        # 🔴 RESTAURATION DU ZOOM LOGIQUE (clé du fix)
        if getattr(self, "_is_zoomed_m1", False):
            x_min, x_max, y_min, y_max = self.config_app.M1_BOUNDS
        else:
            x_min, x_max, y_min, y_max = self.config_app.REAL_BOUNDS

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        self.canvas.draw_idle()

    def _clear_triangulation_layer(self):
        if self._interp_image is not None:
            try:
                self._interp_image.remove()
            except Exception:
                pass
            self._interp_image = None

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        if hasattr(self, "cbar_ax") and self.cbar_ax is not None:
            # Matplotlib peut détacher l'axe lors de remove() de la colorbar.
            # Dans ce cas, clear() plante car cbar_ax.figure est None.
            if self.cbar_ax.figure is not None:
                self.cbar_ax.clear()
                self.cbar_ax.axis("off")
            else:
                self.cbar_ax = self.fig.add_subplot(self._cbar_spec)
                self.cbar_ax.axis("off")

    def _on_precision_change(self):

        self._reset_progress_bar()

        if not hasattr(self, "available_dates") or not self.available_dates:
            return

        selected = [
            sys["id"] for sys in self.systems
            if self.system_vars[sys["id"]].get() == 1
        ]

        if self.mode_var.get() == "single" and len(selected) != 1:
            return

        if self.mode_var.get() == "compare" and len(selected) != 2:
            return

        self._fill_matrix_with_triangulation()
        self._refresh_plot()

    def _fill_matrix_with_triangulation(self):

        mode = self.mode_var.get()
        selected = [sys["id"] for sys in self.systems if self.system_vars[sys["id"]].get() == 1]

        if not selected:
            return

        if not hasattr(self, "available_dates") or not self.available_dates:
            return

        ref_date = self.available_dates[self.ref_index.get()]
        cmp_date = self.available_dates[self.cmp_index.get()]

        processor = TimeSeriesProcessor(
            fill_before=self.fill_before_var.get(),
            fill_between=self.fill_between_var.get(),
            fill_after=self.fill_after_var.get(),
        )

        # ===============================
        # MODE SINGLE
        # ===============================
        if mode == "single":

            if len(selected) != 1:
                return

            sys_id = selected[0]

            matrices = self._build_grouped_matrices(sys_id, ref_date, cmp_date, processor)

            if not matrices:
                self.notify("Triangulation", "Aucune matrice valide")
                return

            (unit, result), = matrices.items()

            self._current_unit = unit
            self._active_matrix_unit = unit
            self._active_selected_systems = [sys_id]
            self._active_compare = False

            self._used_sensor_ids = result["used_sensor_ids"]

            # 🔴 TYPE CORRECT
            all_sensors = self.repository.get_sensors(sys_id)
            used_sensors = [s for s in all_sensors if s["id"] in self._used_sensor_ids]
            self._current_type = self._compute_sensor_type(used_sensors)

            self._display_matrix(result["Z"])
            return

        # ===============================
        # MODE COMPARE
        # ===============================
        if mode == "compare":

            if len(selected) != 2:
                return

            sysA, sysB = selected

            matA = self._build_grouped_matrices(sysA, ref_date, cmp_date, processor)
            matB = self._build_grouped_matrices(sysB, ref_date, cmp_date, processor)

            common_units = [u for u in matA if u in matB]

            if not common_units:
                self.notify("Unités incompatibles")
                return

            unit = common_units[0]

            ZA = matA[unit]["Z"]
            ZB = matB[unit]["Z"]

            self._current_unit = unit
            self._active_matrix_unit = unit
            self._active_selected_systems = [sysA, sysB]
            self._active_compare = True

            self._used_sensor_ids = (
                matA[unit]["used_sensor_ids"] |
                matB[unit]["used_sensor_ids"]
            )

            # 🔴 TYPE CORRECT
            all_sensors = (
                self.repository.get_sensors(sysA) +
                self.repository.get_sensors(sysB)
            )
            used_sensors = [s for s in all_sensors if s["id"] in self._used_sensor_ids]
            self._current_type = self._compute_sensor_type(used_sensors)

            self._display_matrix(ZA - ZB)
            return

        # ===============================
        # MODE CUMUL
        # ===============================
        if mode == "cumul":

            if len(selected) < 2:
                self.notify("Cumul", "Sélectionnez au moins 2 systèmes")
                return

            all_sensors = []
            all_diff = {}
            unit_by_sensor = {}

            for sys_id in selected:

                sensors, diff_dict, _ = self.controller.compute(
                    sys_id, ref_date, cmp_date, processor
                )

                sensors, diff_dict = self._filter_sensors_by_layers(
                    sensors, diff_dict, sys_id
                )

                for s in sensors:

                    sid = s["id"]

                    if sid not in diff_dict:
                        continue

                    val = diff_dict[sid]

                    if val is None or np.isnan(val) or np.isinf(val):
                        continue

                    unit = self.repository.get_sensor_unit(sid)

                    if unit is None:
                        continue

                    all_sensors.append(s)
                    all_diff[sid] = val
                    unit_by_sensor[sid] = unit

            if len(all_sensors) < 3:
                self.notify("Cumul", "Pas assez de capteurs valides (<3)")
                return

            norm_diff, unit = normalize_diff_with_units(all_diff, unit_by_sensor)

            if norm_diff is None:
                self.notify("Cumul", "Unités incompatibles entre systèmes")
                return

            result = self.triangulator.compute(
                all_sensors,
                norm_diff,
                self.precision.get(),
                self.config_app.__dict__,
            )

            if result is None:
                self.notify("Cumul", "Triangulation impossible")
                return

            self._current_unit = unit
            self._active_matrix_unit = unit
            self._active_selected_systems = selected
            self._active_compare = False

            self._used_sensor_ids = result["used_sensor_ids"]

            # 🔴 TYPE CORRECT
            used_sensors = [s for s in all_sensors if s["id"] in self._used_sensor_ids]
            self._current_type = self._compute_sensor_type(used_sensors)

            self._display_matrix(result["Z"])

    def _display_matrix(self, Z):
        self._clear_triangulation_layer()
        unit = getattr(self, "_current_unit", "")

        if unit == "m":
            factor, display_unit = choose_best_si_unit(Z.flatten())
            Z = Z * factor
            unit = display_unit

        vmin = vmax = None
        if self.config_app.SYMMETRIC_COLORBAR:
            if not np.all(np.isnan(Z)):
                vmax = np.nanmax(np.abs(Z))
                vmin = -vmax

        self._interp_image = self.ax.imshow(
            Z,
            extent=self.config_app.REAL_BOUNDS,
            origin="lower",
            cmap=self.cmap_var.get(),
            vmin=vmin,
            vmax=vmax,
            zorder=0,
            alpha=0.8,
        )

        self.cbar_ax.axis("on")
        self._cbar = self.fig.colorbar(self._interp_image, cax=self.cbar_ax)

        # 🔴 FIX TYPE
        sensor_type = getattr(self, "_current_type", None)
        if not sensor_type:
            sensor_type = "valeur"

        unit_label = f" ({unit})" if unit not in (None, "", " ") else ""

        self._cbar.set_label(f"Δ {sensor_type}{unit_label}")

        self.canvas.draw_idle()

    def _filter_sensors_by_layers(self, sensors, diff_dict, system_id):

        if not self.layer_vars:
            return sensors, diff_dict

        active_layers = {
            key for key, var in self.layer_vars.items() if var.get()
        }

        filtered = []

        for s in sensors:

            layer = self._normalize_layer(s.get("layer"))

            if (system_id, layer) in active_layers:
                filtered.append(s)

        allowed_ids = {s["id"] for s in filtered}

        diff_dict = {
            sid: val for sid, val in diff_dict.items()
            if sid in allowed_ids
        }

        return filtered, diff_dict
    
    def _normalize_layer(self, layer):
        return "Sans calque" if layer in (None, "", " ") else layer

    def _build_grouped_matrices(self, system_id, ref_date, cmp_date, processor):
        sensors, diff_dict, _units = self.controller.compute(system_id, ref_date, cmp_date, processor)
        sensors, diff_dict = self._filter_sensors_by_layers(sensors, diff_dict, system_id)

        grouped = {}
        for sensor in sensors:
            sid = sensor["id"]
            if sid not in diff_dict:
                continue

            sensor_unit = self.repository.get_sensor_unit(sid)
            key = sensor_unit
            if key not in grouped:
                grouped[key] = {"sensors": [], "diff": {}, "types": set()}

            grouped[key]["sensors"].append(sensor)
            grouped[key]["diff"][sid] = diff_dict[sid]
            sensor_type = sensor.get("type") if sensor.get("type") not in (None, "", " ") else "Sans type"
            grouped[key]["types"].add(sensor_type)

        matrices = {}
        for unit, payload in grouped.items():
            result = self.triangulator.compute(
                payload["sensors"],
                payload["diff"],
                self.precision.get(),
                self.config_app.__dict__,
            )
            if result is not None:
                result["sensor_types"] = sorted(payload["types"])
                matrices[unit] = result

        return matrices

    def _refresh_plot(self):

        selected = [
            sys["id"] for sys in self.systems
            if self.system_vars[sys["id"]].get() == 1
        ]

        if hasattr(self, "_sensor_artists"):
            for art in self._sensor_artists:
                try:
                    art.remove()
                except:
                    pass

        self._sensor_artists = []

        used_ids = getattr(self, "_used_sensor_ids", None)
        show_used_only = self.show_used_only_var.get()

        for sys_id in selected:

            sensors = self.repository.get_sensors(sys_id)

            filtered = []

            for s in sensors:

                if s["x"] is None or s["y"] is None:
                    continue

                # 🔴 OPTION activée
                if show_used_only:
                    if used_ids is None or s["id"] not in used_ids:
                        continue

                filtered.append(s)

            if not filtered:
                continue

            xs = [s["x"] for s in filtered]
            ys = [s["y"] for s in filtered]

            scatter = self.ax.scatter(
                xs,
                ys,
                s=40,
                color=self.system_colors[sys_id],
                zorder=10,
                picker=5
            )

            scatter._sensors = filtered
            self._sensor_artists.append(scatter)

        self.canvas.draw_idle()

    def _on_fill_option_change(self):

        self._reset_progress_bar()

        selected = [
            sys["id"] for sys in self.systems
            if self.system_vars[sys["id"]].get() == 1
        ]

        expected = 1 if self.mode_var.get() == "single" else 2

        if len(selected) != expected:
            return

        if not hasattr(self, "available_dates") or not self.available_dates:
            return

        self._fill_matrix_with_triangulation()
        self._refresh_plot()

    def _toggle_zoom_m1(self):

        self._is_zoomed_m1 = not self._is_zoomed_m1

        if self._is_zoomed_m1:
            self.zoom_button.config(text="Reset Zoom")
        else:
            self.zoom_button.config(text="Zoom M1")

        # 🔴 IMPORTANT : redraw complet
        self._draw_background()

    def _choose_color(self, system_id):
        cur_color = self.system_colors.get(system_id, "#ff0000")
        new_color = colorchooser.askcolor(color=cur_color, title="Choisir une couleur")
        if new_color[1]:
            self.system_colors[system_id] = new_color[1]
            self._refresh_plot()

    def _on_mode_change(self):

        self._reset_progress_bar()

        self._clear_triangulation_layer()

        for var in self.system_vars.values():
            var.set(0)

        for w in self.layer_frame.winfo_children():
            w.destroy()

        self.layer_vars.clear()

        self.available_dates = []

        if hasattr(self, "_dates_initialized"):
            del self._dates_initialized

        self._reset_view()

        self.cmap_var.set(
            self.config_app.COLORMAP_SINGLE
            if self.mode_var.get() == "single"
            else self.config_app.COLORMAP_COMPARE
        )

        self.layer_frame.update_idletasks()

        self._refresh_plot()

    def _update_colormap(self):
        cmap = self.cmap_var.get()

        if self.mode_var.get() == "single":
            self.config_app.COLORMAP_SINGLE = cmap
        else:
            self.config_app.COLORMAP_COMPARE = cmap

        self._auto_update()

    def _update_layer_options_from_selection(self):
        selected = [sys["id"] for sys in self.systems if self.system_vars[sys["id"]].get() == 1]
        if not selected:
            for w in self.layer_frame.winfo_children():
                w.destroy()
            self.layer_vars.clear()
            return
        self._update_layer_options(selected)

    def _update_layer_options(self, system_ids):

        if isinstance(system_ids, int):
            system_ids = [system_ids]

        # nettoyage du panneau
        for w in self.layer_frame.winfo_children():
            w.destroy()

        self.layer_vars.clear()

        # -----------------------------
        # Bouton TOUS
        # -----------------------------
        self.all_layers_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            self.layer_frame,
            text="Tous",
            variable=self.all_layers_var,
            command=self._toggle_all_layers
        ).pack(anchor="w", pady=(0, 5))

        ttk.Separator(self.layer_frame, orient="horizontal").pack(fill="x", pady=3)

        # -----------------------------
        # Calques par système
        # -----------------------------
        for sys_id in system_ids:

            sensors = self.repository.get_sensors(sys_id)

            explicit_layers = sorted({s["layer"] for s in sensors if s.get("layer") not in (None, "", " ")})
            has_empty_layer = any(s.get("layer") in (None, "", " ") for s in sensors)
            layers = list(explicit_layers)
            if has_empty_layer or not layers:
                layers.append("Sans calque")

            system_name = self._system_name_from_id(sys_id)

            if not layers:
                ttk.Label(
                    self.layer_frame,
                    text=f"({system_name}) Aucun calque",
                    foreground="gray"
                ).pack(anchor="w", padx=5)
                continue

            for layer in layers:

                var = tk.BooleanVar(value=True)

                self.layer_vars[(sys_id, layer)] = var

                ttk.Checkbutton(
                    self.layer_frame,
                    text=f"({system_name}) {layer}",
                    variable=var,
                    command=self._auto_update
                ).pack(anchor="w")


    def _toggle_all_layers(self):
        state = self.all_layers_var.get()

        for var in self.layer_vars.values():
            var.set(state)

        self._auto_update()

    def _update_available_dates(self):

        selected = [sys["id"] for sys in self.systems if self.system_vars[sys["id"]].get() == 1]

        if not selected:
            self.available_dates = []
            return

        if self.mode_var.get() == "single":

            if len(selected) != 1:
                self.available_dates = []
                return

            dates = self.repository.get_common_dates([selected[0]])

        elif self.mode_var.get() in ("compare", "cumul"):

            # 🔴 IMPORTANT : intersection des dates
            if len(selected) < 2:
                self.available_dates = []
                return

            dates = self.repository.get_common_dates(selected)

        else:
            self.available_dates = []
            return

        if not dates:
            self.available_dates = []
            self.notify("Dates", "Aucune date commune entre les systèmes")
            return

        self.available_dates = sorted(dates)

        max_index = len(self.available_dates) - 1

        self.ref_scale.config(from_=0, to=max_index)
        self.cmp_scale.config(from_=0, to=max_index)

        if not hasattr(self, "_dates_initialized"):
            self.ref_index.set(0)
            self.cmp_index.set(max_index)
            self._dates_initialized = True
        else:
            if self.ref_index.get() > max_index:
                self.ref_index.set(0)
            if self.cmp_index.get() > max_index:
                self.cmp_index.set(max_index)

        self._update_date_labels()

    def _on_date_slider_change(self, event=None):
        if not hasattr(self, "available_dates") or not self.available_dates:
            return

        ref_i = self.ref_index.get()
        cmp_i = self.cmp_index.get()

        if ref_i > cmp_i:
            self.cmp_index.set(ref_i)

        self._update_date_labels()

    def _update_date_labels(self):
        if not self.available_dates:
            return
        self.ref_label.config(text=f"Référence : {self.available_dates[self.ref_index.get()]}")
        self.cmp_label.config(text=f"Comparaison : {self.available_dates[self.cmp_index.get()]}")

    def _apply_mapping(self):
        self._fill_matrix_with_triangulation()
        self._refresh_plot()

    def _reset_view(self):
        if self._interp_image is not None:
            try:
                self._interp_image.remove()
            except Exception:
                pass
            self._interp_image = None

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        if hasattr(self, "cbar_ax") and self.cbar_ax is not None:
            if self.cbar_ax.figure is not None:
                self.cbar_ax.clear()
                self.cbar_ax.axis("off")
            else:
                self.cbar_ax = self.fig.add_subplot(self._cbar_spec)
                self.cbar_ax.axis("off")

        if hasattr(self, "_used_sensor_ids"):
            del self._used_sensor_ids

        if hasattr(self, "_last_diff_dict"):
            del self._last_diff_dict

        for coll in list(self.ax.collections):
            coll.remove()
        for line in list(self.ax.lines):
            line.remove()

        self.ax.set_xlim(self.config_app.REAL_BOUNDS[0], self.config_app.REAL_BOUNDS[1])
        self.ax.set_ylim(self.config_app.REAL_BOUNDS[2], self.config_app.REAL_BOUNDS[3])

        self.canvas.draw_idle()

    def _system_name_from_id(self, system_id):
        for sys_data in self.systems:
            if sys_data["id"] == system_id:
                return sys_data["name"]
        return f"ID {system_id}"

    def _open_settings_window(self):

        win = tk.Toplevel(self)
        win.title("Paramètres")
        win.geometry("600x420")

        cfg = self.config_app

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0

        ttk.Label(frame, text="Base de données").grid(row=row, column=0, sticky="w")
        db_var = tk.StringVar(value=str(cfg.DB_PATH))
        ttk.Entry(frame, textvariable=db_var, width=50).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(frame, text="Image de fond").grid(row=row, column=0, sticky="w")
        img_var = tk.StringVar(value=str(cfg.IMG_PATH))
        ttk.Entry(frame, textvariable=img_var, width=50).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(frame, text="Coordonnées de l'image").grid(row=row, column=0, sticky="w")
        real_vars = []
        coord_frame = ttk.Frame(frame)
        coord_frame.grid(row=row, column=1, sticky="w")
        for val in cfg.REAL_BOUNDS:
            var = tk.DoubleVar(value=val)
            real_vars.append(var)
            ttk.Entry(coord_frame, textvariable=var, width=8).pack(side=tk.LEFT, padx=2)
        row += 1

        ttk.Label(frame, text="Coordonnées de M1").grid(row=row, column=0, sticky="w")
        m1_vars = []
        m1_frame = ttk.Frame(frame)
        m1_frame.grid(row=row, column=1, sticky="w")
        for val in cfg.M1_BOUNDS:
            var = tk.DoubleVar(value=val)
            m1_vars.append(var)
            ttk.Entry(m1_frame, textvariable=var, width=8).pack(side=tk.LEFT, padx=2)
        row += 1

        ttk.Label(frame, text="Précision").grid(row=row, column=0, sticky="w")
        prec_var = tk.DoubleVar(value=cfg.DEFAULT_PRECISION)
        ttk.Entry(frame, textvariable=prec_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(frame, text="Colormap 1 système").grid(row=row, column=0, sticky="w")
        cmap_single = tk.StringVar(value=cfg.COLORMAP_SINGLE)
        ttk.Combobox(frame, textvariable=cmap_single, values=cfg.AVAILABLE_COLORMAPS, state="readonly", width=20).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(frame, text="Colormap 2 systèmes").grid(row=row, column=0, sticky="w")
        cmap_compare = tk.StringVar(value=cfg.COLORMAP_COMPARE)
        ttk.Combobox(frame, textvariable=cmap_compare, values=cfg.AVAILABLE_COLORMAPS, state="readonly", width=20).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(frame, text="Loi triangulation").grid(row=row, column=0, sticky="w")
        tri_frame = ttk.Frame(frame)
        tri_frame.grid(row=row, column=1, sticky="w")

        tri_var = tk.StringVar(value=cfg.TRIANGULATION_METHOD)
        radius_var = tk.DoubleVar(value=cfg.INTERPOLATION_RADIUS)
        idw_power = tk.DoubleVar(value=cfg.IDW_POWER)

        rb_delaunay = ttk.Radiobutton(tri_frame, text="Linéaire (Delaunay)", variable=tri_var, value="DELAUNAY")
        rb_delaunay.grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(tri_frame, text="Rayon").grid(row=0, column=1)
        radius_entry = ttk.Entry(tri_frame, textvariable=radius_var, width=8)
        radius_entry.grid(row=0, column=2, padx=5)

        rb_idw = ttk.Radiobutton(tri_frame, text="Pondération distance (IDW)", variable=tri_var, value="IDW")
        rb_idw.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Label(tri_frame, text="Puissance").grid(row=1, column=1)
        idw_entry = ttk.Entry(tri_frame, textvariable=idw_power, width=8)
        idw_entry.grid(row=1, column=2, padx=5)

        def update_fields():
            if tri_var.get() == "DELAUNAY":
                radius_entry.state(["!disabled"])
                idw_entry.state(["disabled"])
            else:
                radius_entry.state(["disabled"])
                idw_entry.state(["!disabled"])

        tri_var.trace_add("write", lambda *args: update_fields())
        update_fields()
        row += 1

        ttk.Label(frame, text="Capteurs IDW").grid(row=row, column=0, sticky="w")
        neighbors_var = tk.IntVar(value=cfg.IDW_NEIGHBORS)
        ttk.Entry(frame, textvariable=neighbors_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(frame, text="Max points grille").grid(row=row, column=0, sticky="w")
        grid_var = tk.IntVar(value=cfg.GRID_MAX_POINTS)
        ttk.Entry(frame, textvariable=grid_var, width=10).grid(row=row, column=1, sticky="w")
        row += 2

        sym_var = tk.BooleanVar(value=cfg.SYMMETRIC_COLORBAR)
        ttk.Checkbutton(frame, text="Colorbar symétrique autour de 0", variable=sym_var).grid(row=row, column=0, columnspan=2, sticky="w")

        # Nombre nb points par courbe
        ttk.Label(frame, text="Points pour les courbes").grid(row=row, column=0, sticky="w")
        curve_options = {
            "1 par jour": "1D",
            "1 par semaine": "1W",
            "1 par mois": "1M",
            "1 par an": "1Y"
        }
        reverse_curve_options = {v: k for k, v in curve_options.items()}
        curve_freq_var = tk.StringVar(
            value=reverse_curve_options.get(cfg.CURVE_POINT_FREQUENCY, "1 par jour")
        )

        combo = ttk.Combobox(
            frame,
            textvariable=curve_freq_var,
            values=list(curve_options.keys()),
            state="readonly",
            width=20
        )

        combo.grid(row=row, column=1, sticky="w")

        row += 1

        def save():
            cfg.DB_PATH = Path(db_var.get())
            cfg.IMG_PATH = Path(img_var.get())
            cfg.REAL_BOUNDS = [v.get() for v in real_vars]
            cfg.M1_BOUNDS = [v.get() for v in m1_vars]
            cfg.DEFAULT_PRECISION = prec_var.get()
            cfg.COLORMAP_SINGLE = cmap_single.get()
            cfg.COLORMAP_COMPARE = cmap_compare.get()
            cfg.TRIANGULATION_METHOD = tri_var.get()
            cfg.INTERPOLATION_RADIUS = radius_var.get()
            cfg.IDW_POWER = idw_power.get()
            cfg.IDW_NEIGHBORS = neighbors_var.get()
            cfg.GRID_MAX_POINTS = grid_var.get()
            cfg.SYMMETRIC_COLORBAR = sym_var.get()
            cfg.CURVE_POINT_FREQUENCY = curve_options.get(curve_freq_var.get(), "1D")

            cfg.save()
            cfg.load()
            self._apply_mapping()
            messagebox.showinfo("Paramètres", "Configuration sauvegardée")
            win.destroy()

        ttk.Button(frame, text="Enregistrer", command=save).grid(row=row + 1, column=0, columnspan=2, pady=10)
        frame.columnconfigure(1, weight=1)

    def _build_sidebar(self):

        sidebar_container = ttk.Frame(self)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y)

        # -----------------------------
        # PARTIE SCROLLABLE
        # -----------------------------

        scroll_container = ttk.Frame(sidebar_container)
        scroll_container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(
            scroll_container,
            width=260,
            highlightthickness=0,
            bd=0
        )
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.sidebar = ttk.Frame(canvas)

        canvas.create_window((0, 0), window=self.sidebar, anchor="nw")

        self.sidebar.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.configure(yscrollcommand=scrollbar.set)

        # scroll souris seulement dans la zone scrollable
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        sidebar = self.sidebar

        # -----------------------------
        # CONTENU SCROLLABLE
        # -----------------------------

        mode_frame = ttk.LabelFrame(sidebar, text="Mode")
        mode_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(mode_frame, text="Variations d'1 système",
                        variable=self.mode_var, value="single",
                        command=self._on_mode_change).pack(anchor="w")
        
        ttk.Radiobutton(mode_frame, text="Cumuler plusieurs systèmes",
                        variable=self.mode_var, value="cumul",
                        command=self._on_mode_change).pack(anchor="w")

        ttk.Radiobutton(mode_frame, text="Comparaison de 2 systèmes",
                        variable=self.mode_var, value="compare",
                        command=self._on_mode_change).pack(anchor="w")

        self.zoom_button = ttk.Button(sidebar, text="Zoom M1", command=self._toggle_zoom_m1)
        self.zoom_button.pack(fill=tk.X, pady=4, padx=5)
        
        self.m1_button = ttk.Button(
            sidebar,
            text="Afficher poteaux M1",
            command=self._toggle_m1_background
        )
        self.m1_button.pack(fill=tk.X, pady=4, padx=5)

        frame = ttk.LabelFrame(sidebar, text="Systèmes à afficher")
        frame.pack(fill=tk.X, pady=3, padx=5)

        for sys_data in self.systems:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)

            var = tk.IntVar(value=0)
            self.system_vars[sys_data['id']] = var

            is_saphir = "SAPHIR" in sys_data['name'].upper()

            chk = ttk.Checkbutton(
                row,
                text=sys_data['name'],
                variable=var,
                command=lambda sid=sys_data['id']: self._on_system_toggle(sid),
            )

            if is_saphir:
                chk.state(["disabled"])

            chk.pack(side=tk.LEFT)

            ttk.Button(
                row,
                text="Couleur",
                command=lambda sid=sys_data['id']: self._choose_color(sid)
            ).pack(side=tk.RIGHT)

            default_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(self.system_colors) % 10]
            self.system_colors[sys_data['id']] = default_color

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=8)

        self.layer_frame = ttk.LabelFrame(sidebar, text="Calques utilisés")
        self.layer_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=8)

        date_frame = ttk.LabelFrame(sidebar, text="Sélection des dates")
        date_frame.pack(fill=tk.X, pady=4, padx=5)

        self.ref_index = tk.IntVar(value=0)
        self.cmp_index = tk.IntVar(value=0)

        self.ref_label = ttk.Label(date_frame, text="Référence : -")
        self.ref_label.pack()

        self.ref_scale = tk.Scale(
            date_frame,
            orient="horizontal",
            variable=self.ref_index,
            showvalue=False,
            command=self._on_date_slider_change
        )
        self.ref_scale.bind("<ButtonRelease-1>", self._on_date_slider_release)
        self.ref_scale.pack(fill=tk.X)

        self.cmp_label = ttk.Label(date_frame, text="Comparaison : -")
        self.cmp_label.pack()

        self.cmp_scale = tk.Scale(
            date_frame,
            orient="horizontal",
            variable=self.cmp_index,
            showvalue=False,
            command=self._on_date_slider_change
        )
        self.cmp_scale.bind("<ButtonRelease-1>", self._on_date_slider_release)
        self.cmp_scale.pack(fill=tk.X)

        prec_frame = ttk.Frame(sidebar)
        prec_frame.pack(fill=tk.X, pady=4)

        ttk.Label(prec_frame, text="Précision mapping :").pack(side=tk.LEFT)
        ttk.Entry(prec_frame, textvariable=self.precision,
                width=6, justify='center').pack(side=tk.RIGHT)

        fill_frame = ttk.LabelFrame(sidebar, text="Remplissage des données")
        fill_frame.pack(fill=tk.X, pady=6, padx=5)

        ttk.Checkbutton(fill_frame,
                        text="Remplir avant la 1ère valeur",
                        variable=self.fill_before_var,
                        command=self._on_fill_option_change).pack(anchor="w")

        ttk.Checkbutton(fill_frame,
                        text="Interpolation linéaire entre valeurs",
                        variable=self.fill_between_var,
                        command=self._on_fill_option_change).pack(anchor="w")

        ttk.Checkbutton(fill_frame,
                        text="Remplir après la dernière valeur",
                        variable=self.fill_after_var,
                        command=self._on_fill_option_change).pack(anchor="w")
        
        # -----------------------------
        # Option affichage points
        # -----------------------------

        self.show_used_only_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            sidebar,
            text="Afficher seulement capteurs utilisés",
            variable=self.show_used_only_var,
            command=self._refresh_plot
        ).pack(anchor="w", padx=5, pady=4)

        # -----------------------------
        # Colormap
        # -----------------------------

        cmap_frame = ttk.Frame(sidebar)
        cmap_frame.pack(fill=tk.X, pady=4)

        ttk.Label(cmap_frame, text="Colormap").pack(side=tk.LEFT)

        self.cmap_var = tk.StringVar(value=self.config_app.COLORMAP_SINGLE)

        cmap_combo = ttk.Combobox(
            cmap_frame,
            textvariable=self.cmap_var,
            values=self.config_app.AVAILABLE_COLORMAPS,
            state="readonly",
            width=12
        )

        cmap_combo.pack(side=tk.RIGHT)

        cmap_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._update_colormap()
        )

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=8)


        ttk.Button(
            sidebar,
            text="Calcul pour courbes",
            command=self._compute_curve_cache
        ).pack(fill=tk.X, pady=4, padx=5)

        self.progress = ttk.Progressbar(
            sidebar,
            orient="horizontal",
            length=200,
            mode="determinate"
        )
        self.progress.pack(fill=tk.X, padx=5, pady=3)

        ttk.Button(sidebar, text="Paramètres",
                command=self._open_settings_window).pack(fill=tk.X, pady=6, padx=5)

        # -----------------------------
        # BARRE DE MESSAGE FIXE
        # -----------------------------

        ttk.Separator(sidebar_container, orient="horizontal").pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Prêt")

        self.status_bar = tk.Label(
            sidebar_container,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            padx=6,
            pady=6
        )

        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _toggle_m1_background(self):

        self._m1_active = not self._m1_active

        if self._m1_active:
            self.m1_button.config(text="Afficher fond principal")
        else:
            self.m1_button.config(text="Afficher poteaux M1")

        # 🔴 redraw propre
        self._draw_background()

    def _reset_progress_bar(self):
        if hasattr(self, "progress"):
            self.progress["value"] = 0
            self.progress["maximum"] = 1

    def _on_date_slider_release(self, event=None):
        if not hasattr(self, "available_dates") or not self.available_dates:
            return

        ref_i = self.ref_index.get()
        cmp_i = self.cmp_index.get()

        if ref_i > cmp_i:
            self.cmp_index.set(ref_i)

        self._update_date_labels()
        self._auto_update()

    def _get_curve_dates(self):

        if not self.available_dates:
            return []

        dates = sorted(pd.to_datetime(self.available_dates))

        freq = self.config_app.CURVE_POINT_FREQUENCY

        # intervalle minimal entre deux dates retenues
        freq_days = {
            "1D": 1,
            "1W": 7,
            "1M": 30,
            "1Y": 365
        }.get(freq, 1)

        selected = []
        last_date = None

        for d in dates:

            if last_date is None:
                selected.append(d)
                last_date = d
                continue

            if (d - last_date).days >= freq_days:
                selected.append(d)
                last_date = d

        return [d.date() for d in selected]

    def _compute_curve_cache(self):

        selected = [
            sys["id"] for sys in self.systems
            if self.system_vars[sys["id"]].get() == 1
        ]

        if not selected:
            self.notify("Calcul", "Aucun système sélectionné", "warn")
            return

        if not self.available_dates:
            self.notify("Calcul", "Aucune date disponible", "warn")
            return

        if not hasattr(self, "_active_matrix_unit"):
            self.notify("Calcul", "Afficher une matrice avant de lancer le calcul des courbes.", "warn")
            return

        if self.layer_vars:
            active_layers = {k for k, v in self.layer_vars.items() if v.get()}
            if not active_layers:
                self.notify("Calcul", "Aucun calque sélectionné", "warn")
                return
        else:
            active_layers = set()

        curve_dates = self._get_curve_dates()

        if not curve_dates:
            self.notify("Calcul", "Aucune date utilisable", "warn")
            return

        self._triangulation_cache.clear()
        self._curve_units = {}

        tasks = []
        matrix_unit = self._active_matrix_unit

        for sys_id in selected:
            self._curve_units[sys_id] = matrix_unit

            for date in curve_dates:
                tasks.append(
                    (
                        sys_id,
                        curve_dates[0],
                        date,
                        self.precision.get(),
                        self.config_app.__dict__,
                        self.fill_before_var.get(),
                        self.fill_between_var.get(),
                        self.fill_after_var.get(),
                        active_layers,
                        matrix_unit,
                    )
                )

        total = len(tasks)

        self.progress["maximum"] = total
        self.progress["value"] = 0

        self.notify("Calcul", f"{total} matrices à calculer", "info", duration=None)

        count = 0

        with Pool(cpu_count()) as pool:
            for sys_id, date, result in pool.imap_unordered(compute_matrix_task, tasks):

                if sys_id not in self._triangulation_cache:
                    self._triangulation_cache[sys_id] = {}

                if result:
                    self._triangulation_cache[sys_id][date] = result

                count += 1
                self.progress["value"] = count

                self.update_idletasks()

        self.notify("Calcul terminé", f"{count} matrices calculées", "ok")

    def _get_value_from_matrix(self, triangulation, x, y):

        X = triangulation["X"]
        Y = triangulation["Y"]
        Z = triangulation["Z"]

        dist = (X - x)**2 + (Y - y)**2
        idx = dist.argmin()

        iy, ix = np.unravel_index(idx, X.shape)

        return Z[iy, ix]


    def _plot_point_timeseries(self, x, y):

        if not self._triangulation_cache:
            self.notify("Courbe", "Calculer les matrices avant affichage de la courbe", "warn")
            return

        selected = getattr(self, "_active_selected_systems", [])
        if not selected:
            self.notify("Courbe", "Aucun système actif pour la courbe.", "warn")
            return

        mode_compare = getattr(self, "_active_compare", False)

        if mode_compare:
            fig = plt.figure(figsize=(11, 7))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
            ax_main = fig.add_subplot(gs[0, :])
            interp_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        else:
            fig = plt.figure(figsize=(7, 6))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 2])
            ax_main = fig.add_subplot(gs[0])
            interp_axes = [fig.add_subplot(gs[1])]

        series_by_sys = {}
        plotted = False
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for idx, sys_id in enumerate(selected):
            matrices = self._triangulation_cache.get(sys_id, {})

            dates = []
            values = []

            for d, triangulation in matrices.items():

                if triangulation is None:
                    continue

                # 🔴 IMPORTANT : vérifier que le point est dans la zone valide
                v = self._get_value_from_matrix(triangulation, x, y)

                if np.isnan(v):
                    continue

                dates.append(pd.to_datetime(d))
                values.append(v)

            if not dates:
                continue

            dates, values = zip(*sorted(zip(dates, values)))
            dates = list(dates)
            values = list(values)

            series_by_sys[sys_id] = (dates, values)
            ax_main.plot(
                dates,
                values,
                marker="o",
                linewidth=1.8,
                color=colors[idx % len(colors)],
                label=self._system_name_from_id(sys_id),
            )
            plotted = True

        if mode_compare and len(selected) == 2 and selected[0] in series_by_sys and selected[1] in series_by_sys:
            d1, v1 = series_by_sys[selected[0]]
            d2, v2 = series_by_sys[selected[1]]

            map1 = {d: v for d, v in zip(d1, v1)}
            map2 = {d: v for d, v in zip(d2, v2)}
            common_dates = sorted(set(map1.keys()).intersection(map2.keys()))

            if common_dates:
                diff_values = [map1[d] - map2[d] for d in common_dates]
                ax_main.plot(
                    common_dates,
                    diff_values,
                    marker="o",
                    linewidth=1.8,
                    color=colors[2],
                    label="Différence (S1 - S2)",
                )
                plotted = True

        if not plotted:
            self.notify(
                "Courbe",
                "Aucune matrice disponible pour ce point (hors zone interpolée)",
                "warn"
            )
            plt.close(fig)
            return

        try:
            ref_date = self.available_dates[self.ref_index.get()]
            cmp_date = self.available_dates[self.cmp_index.get()]

            ax_main.axvline(ref_date, color="green", linewidth=0.8)
            ax_main.axvline(cmp_date, color="green", linewidth=0.8)
        except Exception:
            pass

        unit = getattr(self, "_current_unit", "")
        unit_label = f" ({unit})" if unit not in (None, "", " ") else ""
        sensor_type = getattr(self, "_current_type", None)
        if sensor_type:
            ax_main.set_ylabel(f"Δ {sensor_type}{unit_label}")
        else:
            ax_main.set_ylabel(f"Δ valeur{unit_label}")

        ax_main.set_title(f"Point ({x:.2f} , {y:.2f})")
        ax_main.set_xlabel("Date")
        ax_main.grid(True)
        ax_main.legend(loc="upper left")

        if mode_compare and len(selected) >= 2:
            label_1 = self._system_name_from_id(selected[0])
            label_2 = self._system_name_from_id(selected[1])
            neighbors_1 = self._get_interpolation_neighbors_for_system(x, y, selected[0])
            neighbors_2 = self._get_interpolation_neighbors_for_system(x, y, selected[1])

            max_dist_1 = max((d for d, _ in neighbors_1), default=1.0)
            max_dist_2 = max((d for d, _ in neighbors_2), default=1.0)
            common_radius = max(max_dist_1, max_dist_2, 1.0) * 1.15

            self._draw_interpolation_scheme(
                interp_axes[0],
                x,
                y,
                neighbors_1,
                title=label_1,
                radius=common_radius,
            )
            self._draw_interpolation_scheme(
                interp_axes[1],
                x,
                y,
                neighbors_2,
                title=label_2,
                radius=common_radius,
            )
        else:
            neighbors = self._get_interpolation_neighbors(x, y)
            label = self._system_name_from_id(selected[0]) if selected else "Interpolation"
            radius = max((d for d, _ in neighbors), default=1.0) * 1.15
            self._draw_interpolation_scheme(
                interp_axes[0],
                x,
                y,
                neighbors,
                title=label,
                radius=radius,
            )

        plt.tight_layout()
        plt.show()

    def _draw_interpolation_scheme(self, ax, x, y, neighbors, title="Interpolation", radius=None):

        ax.set_title(title)

        if not neighbors:
            ax.axis("off")
            return

        power = self.config_app.IDW_POWER
        dists = np.array([d for d, _ in neighbors])

        weights = 1 / (dists**power + 1e-12)
        weights = weights / np.sum(weights)

        if radius is None:
            radius = max(np.max(dists), 1.0) * 1.15

        # point analysé
        ax.scatter([x], [y], color="red", s=80, zorder=5)
        ax.text(x, y, "Point", ha="center", va="bottom")

        for (dist, sensor), w in zip(neighbors, weights):
            sx = sensor["x"]
            sy = sensor["y"]

            ax.scatter(sx, sy, color="blue")
            ax.plot([x, sx], [y, sy], color="orange", linewidth=1)

            ax.text(
                sx,
                sy,
                f"{sensor['name']}\n{dist:.1f} m\n{w*100:.1f}%",
                fontsize=8,
                ha="center",
                va="bottom",
            )

        ax.set_xlim(x - radius, x + radius)
        ax.set_ylim(y - radius, y + radius)
        ax.set_aspect("equal", adjustable="box")

        # pas de cadre / pas d'axes
        ax.set_frame_on(False)
        ax.axis("off")

    def _find_nearest_sensors(self, x, y, n=5):

        sensors = []

        # utiliser les capteurs réellement affichés
        for scatter in getattr(self, "_sensor_artists", []):

            for s in scatter._sensors:

                if s["x"] is None or s["y"] is None:
                    continue

                dist = np.sqrt((s["x"] - x)**2 + (s["y"] - y)**2)

                sensors.append((dist, s))

        if not sensors:
            return []

        sensors.sort(key=lambda v: v[0])

        return sensors[:n]

    def _get_neighbors_with_weights(self, x, y):

        neighbors = self._get_interpolation_neighbors(x, y)

        if not neighbors:
            return []

        power = self.config_app.IDW_POWER

        dists = np.array([d for d, _ in neighbors])

        weights = 1.0 / (dists**power + 1e-12)
        weights = weights / np.sum(weights)

        result = []

        for (dist, sensor), w in zip(neighbors, weights):
            result.append({
                "name": sensor["name"],
                "distance": dist,
                "weight": w
            })

        return result

    def _get_interpolation_neighbors(self, x, y):

        neighbors = []

        for scatter in getattr(self, "_sensor_artists", []):

            for s in scatter._sensors:

                if s["x"] is None or s["y"] is None:
                    continue

                dist = np.sqrt((s["x"] - x)**2 + (s["y"] - y)**2)

                neighbors.append((dist, s))

        neighbors.sort(key=lambda v: v[0])

        n = self.config_app.IDW_NEIGHBORS

        return neighbors[:n]

    def _get_interpolation_neighbors_for_system(self, x, y, system_id):
        sensors = self.repository.get_sensors(system_id)

        target_unit = getattr(self, "_active_matrix_unit", None)

        filtered = []
        for s in sensors:
            if s["x"] is None or s["y"] is None:
                continue

            sensor_unit = self.repository.get_sensor_unit(s["id"])

            if target_unit is not None and sensor_unit != target_unit:
                continue

            filtered.append(s)

        neighbors = []
        for s in filtered:
            dist = np.sqrt((s["x"] - x)**2 + (s["y"] - y)**2)
            neighbors.append((dist, s))

        neighbors.sort(key=lambda v: v[0])
        n = self.config_app.IDW_NEIGHBORS
        return neighbors[:n]

    def _compute_sensor_type(self, sensors):

        types = [
            s.get("type")
            for s in sensors
            if s.get("type") not in (None, "", " ")
        ]

        if not types:
            return None

        unique = set(types)

        if len(unique) == 1:
            return unique.pop()

        return "types mixtes"

    def _compute_idw_weights(self, neighbors):

        power = self.config_app.IDW_POWER

        dists = np.array([d for d, _ in neighbors])

        weights = 1.0 / (dists**power + 1e-12)

        weights = weights / np.sum(weights)

        return weights

    def _show_interpolation_neighbors(self, x, y):

        neighbors = self._get_interpolation_neighbors(x, y)

        if not neighbors:
            return

        weights = self._compute_idw_weights(neighbors)

        # afficher sur la carte
        for (dist, s), w in zip(neighbors, weights):

            self.ax.plot(
                [x, s["x"]],
                [y, s["y"]],
                color="orange",
                linewidth=1,
                zorder=15
            )

            self.ax.text(
                s["x"],
                s["y"],
                f"{s['name']}\n{dist:.1f} m\n{w*100:.1f} %",
                fontsize=8,
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round",
                    fc="white",
                    ec="none",
                    alpha=0.7
                ),
                zorder=20
            )

        self.canvas.draw_idle()

    def on_close(self):
        try:
            self.db.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = MultiSystemMappingApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
