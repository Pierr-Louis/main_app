#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Interface Tkinter pour afficher les courbes de surveillance depuis SQLite."""

from __future__ import annotations

import os
import sqlite3
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

WINDOW_TITLE = "Courbes de surveillance"
CONVERSION_CORDE_FISSURE = 0.1 / 150.0
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"
WINDOWS_DB_PATH = Path(r"R:\2 - Surveillance\surveillance_app\data\surveillance.db")


class DatabaseLoader:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else self._resolve_db_path()

    def _resolve_db_path(self) -> Path:
        if DEFAULT_DB_PATH.exists():
            return DEFAULT_DB_PATH
        return WINDOWS_DB_PATH

    def load(self) -> tuple[pd.DataFrame, dict[str, dict[str, object]], list[str]]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Base SQLite introuvable : {self.db_path}")

        with sqlite3.connect(self.db_path) as conn:
            systems = self._load_systems(conn)
            metadata = self._load_sensor_metadata(conn)
            measurements = self._load_measurements(conn)

        if measurements.empty:
            raise RuntimeError("Aucune mesure n'a été trouvée dans la base.")
        measurements = self._apply_relative_baseline(measurements)
        return measurements, metadata, systems

    def _load_systems(self, conn: sqlite3.Connection) -> list[str]:
        df = pd.read_sql_query("SELECT name FROM systems ORDER BY name", conn)
        return df["name"].dropna().astype(str).tolist()

    def _load_sensor_metadata(self, conn: sqlite3.Connection) -> dict[str, dict[str, object]]:
        query = """
            SELECT
                sys.name AS systeme,
                s.name AS capteur,
                s.type AS type,
                COALESCE(s.unit, '') AS unite,
                s.seuil_bas AS bas,
                s.seuil_haut AS haut,
                s.layer AS couche,
                s.x,
                s.y
            FROM sensors s
            JOIN systems sys ON sys.id = s.system_id
            ORDER BY sys.name, s.name
        """
        df = pd.read_sql_query(query, conn)
        info: dict[str, dict[str, object]] = {}
        for _, row in df.iterrows():
            systeme = str(row["systeme"]).strip()
            capteur = str(row["capteur"]).strip()
            info[self.sensor_key(systeme, capteur)] = {
                "systeme": systeme,
                "capteur": capteur,
                "type": str(row.get("type") or "").strip(),
                "unite": str(row.get("unite") or "").strip(),
                "bas": self._as_float(row.get("bas")),
                "haut": self._as_float(row.get("haut")),
                "couche": str(row.get("couche") or "").strip(),
                "x": self._as_float(row.get("x")),
                "y": self._as_float(row.get("y")),
            }
        return info

    def _load_measurements(self, conn: sqlite3.Connection) -> pd.DataFrame:
        query = """
            SELECT
                m.datetime AS Date,
                sys.name AS Systeme,
                s.name AS Capteur,
                m.value AS Valeur,
                COALESCE(m.unit, s.unit, '') AS Unite
            FROM measurements m
            JOIN sensors s ON s.id = m.sensor_id
            JOIN systems sys ON sys.id = s.system_id
            WHERE m.value IS NOT NULL
            ORDER BY sys.name, s.name, m.datetime
        """
        df = pd.read_sql_query(query, conn)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Systeme"] = df["Systeme"].astype(str).str.strip()
        df["Capteur"] = df["Capteur"].astype(str).str.strip()
        df["Valeur"] = pd.to_numeric(df["Valeur"], errors="coerce")
        df["Unite"] = df["Unite"].fillna("").astype(str).str.strip()
        df = df.dropna(subset=["Date", "Systeme", "Capteur", "Valeur"])
        df = df.sort_values(["Systeme", "Capteur", "Date"]).reset_index(drop=True)
        df["Jour"] = df["Date"].dt.normalize()
        df["JourOrdinal"] = df["Jour"].map(pd.Timestamp.toordinal)
        df["SensorKey"] = df.apply(lambda row: self.sensor_key(row["Systeme"], row["Capteur"]), axis=1)
        return df[["Date", "Jour", "JourOrdinal", "Systeme", "Capteur", "SensorKey", "Valeur", "Unite"]]

    def _apply_relative_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["SensorKey", "Date"]).copy()
        first_values = df.groupby("SensorKey")["Valeur"].transform("first")
        df["Valeur"] = df["Valeur"] - first_values
        return df

    @staticmethod
    def sensor_key(systeme: str, capteur: str) -> str:
        return f"{systeme}::{capteur}"

    @staticmethod
    def _as_float(value: object) -> float | None:
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def appliquer_lissage(df: pd.DataFrame, colonne: str, window: int = 7) -> pd.DataFrame:
    df_l = df.sort_values("Date").copy()
    df_l[colonne] = df_l[colonne].rolling(window=window, center=True, min_periods=1).mean()
    return df_l


def calculer_variation_annuelle(serie: pd.DataFrame, colonne: str) -> pd.DataFrame:
    """Calcule la variation vs ~1 an avant avec rapprochement temporel tolérant."""
    if serie.empty:
        return serie.copy()

    base = serie[["Date", colonne]].sort_values("Date").copy()
    ref = base.rename(columns={"Date": "DateRef", colonne: "ValeurRef"}).copy()
    base["TargetDate"] = base["Date"] - pd.Timedelta(days=365)

    merged = pd.merge_asof(
        base.sort_values("TargetDate"),
        ref.sort_values("DateRef"),
        left_on="TargetDate",
        right_on="DateRef",
        direction="nearest",
        tolerance=pd.Timedelta(days=2),
    )
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged["variation"] = merged[colonne] - merged["ValeurRef"]
    return merged[["Date", colonne, "variation"]]


def pente_locale_sur_fenetre(dates: pd.Series, values: pd.Series, window_days: int) -> pd.Series:
    """Pente locale basée sur une fenêtre temporelle glissante exprimée en jours."""
    if len(dates) < 2:
        return pd.Series(np.nan, index=dates.index)

    times = pd.to_datetime(dates)
    values_num = pd.to_numeric(values, errors="coerce")
    slopes = []

    for idx, current_date in enumerate(times):
        start_date = current_date - pd.Timedelta(days=window_days)
        mask = (times >= start_date) & (times <= current_date)
        sub_times = times[mask]
        sub_values = values_num[mask]
        valid = sub_values.notna()
        sub_times = sub_times[valid]
        sub_values = sub_values[valid]

        if len(sub_values) < 2:
            slopes.append(np.nan)
            continue

        x = (sub_times - sub_times.iloc[0]).dt.total_seconds().to_numpy() / 86400.0
        y = sub_values.to_numpy(dtype=float)
        if np.allclose(x, x[0]):
            slopes.append(np.nan)
            continue
        slopes.append(float(np.polyfit(x, y, 1)[0]))

    return pd.Series(slopes, index=dates.index)


class App(tk.Tk):
    def __init__(self, data: pd.DataFrame, capteur_info: dict[str, dict[str, object]], systemes: list[str]):
        super().__init__()
        self.title(WINDOW_TITLE)
        self.geometry("1350x980")
        self.data = data
        self.capteur_info = capteur_info
        self.systemes = systemes
        self.moyenne_journaliere = tk.BooleanVar(value=False)

        self.selected_systeme = tk.StringVar(value=self.systemes[0] if self.systemes else "")
        self.lissage_actif = tk.BooleanVar(value=False)
        self.afficher_variation = tk.BooleanVar(value=False)
        self.demarrer_a_zero = tk.BooleanVar(value=False)
        self.fenetre_mapping = {"30 jours": 30, "90 jours": 90, "365 jours": 365}
        self.fenetre_pente = tk.StringVar(value="30 jours")
        self.day_start = tk.IntVar(value=0)
        self.day_end = tk.IntVar(value=0)

        self.capteurs: list[str] = []
        self.sensor_vars: dict[str, tk.BooleanVar] = {}
        self.sensor_menu_text = tk.StringVar(value="Sélectionner les capteurs")
        self.sensor_window: tk.Toplevel | None = None
        self.pending_sensor_vars: dict[str, tk.BooleanVar] = {}
        self.fig = plt.figure(figsize=(13, 8))
        self.canvas: FigureCanvasTkAgg | None = None
        self._saved_views: list[tuple[float, float]] | None = None
        self._skip_restore_view = False
        self._dragging = False
        self._press_event = None

        self.columnconfigure(0, weight=1)
        self._creer_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._mettre_a_jour_liste_capteurs()

    def _on_close(self):
        self.destroy()
        sys.exit(0)

    def _creer_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

        ttk.Label(top, text="Système :", font=("Arial", 11)).pack(side=tk.LEFT)
        self.cb_systeme = ttk.Combobox(top, textvariable=self.selected_systeme, values=self.systemes, state="readonly", width=28)
        self.cb_systeme.pack(side=tk.LEFT, padx=5)
        self.cb_systeme.bind("<<ComboboxSelected>>", lambda e: self._on_systeme_change())

        ttk.Checkbutton(top, text="Moyenne par jour", variable=self.moyenne_journaliere, command=self._rafraichir).pack(side=tk.LEFT, padx=12)
        ttk.Checkbutton(top, text="Lisser la courbe (7 points)", variable=self.lissage_actif, command=self._rafraichir).pack(side=tk.LEFT, padx=12)
        ttk.Checkbutton(top, text="Afficher la variation (vs -1 an)", variable=self.afficher_variation, command=self._rafraichir).pack(side=tk.LEFT, padx=12)
        ttk.Checkbutton(top, text="Démarrer à 0 (fenêtre)", variable=self.demarrer_a_zero, command=self._rafraichir).pack(side=tk.LEFT, padx=12)

        ttk.Label(top, text="Fenêtre pente :").pack(side=tk.LEFT, padx=(12, 4))
        self.cb_fenetre = ttk.Combobox(top, textvariable=self.fenetre_pente, values=list(self.fenetre_mapping.keys()), state="readonly", width=10)
        self.cb_fenetre.pack(side=tk.LEFT)
        self.cb_fenetre.bind("<<ComboboxSelected>>", lambda e: self._rafraichir())

        ttk.Button(top, text="Exporter la sélection", command=self._exporter_selection).pack(side=tk.LEFT, padx=12)
        ttk.Button(top, text="Exporter tout le système", command=self._exporter_tous_les_graphes).pack(side=tk.LEFT, padx=8)

        selection_frame = ttk.LabelFrame(self, text="Capteurs affichés")
        selection_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=6)
        selection_frame.columnconfigure(1, weight=1)

        ttk.Label(selection_frame, text="Capteurs :").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Button(selection_frame, text="Choisir les capteurs", command=self._ouvrir_fenetre_capteurs).grid(row=0, column=1, padx=8, pady=8, sticky="w")
        ttk.Label(selection_frame, textvariable=self.sensor_menu_text).grid(row=0, column=2, padx=8, pady=8, sticky="ew")
        selection_frame.columnconfigure(2, weight=1)

        self.analysis_frame = ttk.LabelFrame(self, text="Analyse automatique")
        self.analysis_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=6)
        self.analysis_frame.columnconfigure(0, weight=1)
        self.analysis_text = tk.Text(self.analysis_frame, height=5, wrap="word")
        self.analysis_text.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self.analysis_text.configure(state="disabled")

        self.cur_frame = ttk.LabelFrame(self, text="Période à afficher (jours)")
        self.cur_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=8)

        ttk.Label(self.cur_frame, text="Début :", width=8).grid(row=0, column=0, padx=5, pady=5)
        self.cur_debut = ttk.Scale(self.cur_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.day_start, command=self._on_curseur_change)
        self.cur_debut.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.lbl_debut = ttk.Label(self.cur_frame, width=14, text="-")
        self.lbl_debut.grid(row=0, column=2, padx=5)

        ttk.Label(self.cur_frame, text="Fin :", width=8).grid(row=1, column=0, padx=5, pady=5)
        self.cur_fin = ttk.Scale(self.cur_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.day_end, command=self._on_curseur_change)
        self.cur_fin.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.lbl_fin = ttk.Label(self.cur_frame, width=14, text="-")
        self.lbl_fin.grid(row=1, column=2, padx=5)
        self.cur_frame.columnconfigure(1, weight=1)

        frame_graph = ttk.Frame(self)
        frame_graph.grid(row=4, column=0, sticky="nsew")
        frame_graph.columnconfigure(0, weight=1)
        frame_graph.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(frame_graph)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        self.canvas.mpl_connect("scroll_event", self._zoom_molette)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        frame_bottom = ttk.Frame(self)
        frame_bottom.grid(row=5, column=0, sticky="ew")
        ttk.Button(frame_bottom, text="Reset zoom", command=self._reset_zoom).pack(pady=5)

    def _reset_zoom(self):
        self._saved_views = None
        self._skip_restore_view = True
        self._mettre_a_jour_curseurs(reset_range=True)
        self._rafraichir()

    def _on_mouse_press(self, event):
        if event.button == 1 and event.inaxes:
            self._dragging = True
            self._press_event = event

    def _on_mouse_release(self, event):
        if event.button == 1:
            self._dragging = False
            self._press_event = None

    def _on_mouse_move(self, event):
        if not self._dragging or event.inaxes is None or self._press_event is None:
            return

        ax = event.inaxes
        dx = event.x - self._press_event.x
        dy = event.y - self._press_event.y

        width = ax.get_xlim()[1] - ax.get_xlim()[0]
        height = ax.get_ylim()[1] - ax.get_ylim()[0]
        dx_data = -dx * width / ax.bbox.width
        dy_data = -dy * height / ax.bbox.height

        for a in self.fig.axes:
            xl = a.get_xlim()
            a.set_xlim(xl[0] + dx_data, xl[1] + dx_data)

        yl = ax.get_ylim()
        ax.set_ylim(yl[0] + dy_data, yl[1] + dy_data)

        self._press_event = event
        self._sync_day_sliders_from_axes()
        if self.canvas:
            self.canvas.draw_idle()

    def _zoom_molette(self, event):
        ax = event.inaxes
        if ax is None:
            return

        scale = 1 / 1.2 if event.button == "up" else 1.2
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        new_width = (xlim[1] - xlim[0]) * scale
        new_height = (ylim[1] - ylim[0]) * scale

        relx = 0.5 if xlim[1] == xlim[0] else (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = 0.5 if ylim[1] == ylim[0] else (ydata - ylim[0]) / (ylim[1] - ylim[0])

        new_xlim = [xdata - new_width * relx, xdata + new_width * (1 - relx)]
        for a in self.fig.axes:
            a.set_xlim(new_xlim)
        ax.set_ylim([ydata - new_height * rely, ydata + new_height * (1 - rely)])

        self._sync_day_sliders_from_axes()
        if self.canvas:
            self.canvas.draw_idle()

    def _filtered_measurements(self, capteurs: list[str] | None = None) -> pd.DataFrame:
        df = self.data[self.data["Systeme"] == self.selected_systeme.get()].copy()
        if capteurs:
            df = df[df["Capteur"].isin(capteurs)]
        return df

    def _sensor_key(self, capteur: str) -> str:
        return DatabaseLoader.sensor_key(self.selected_systeme.get(), capteur)

    def _selected_capteurs(self) -> list[str]:
        return [capteur for capteur, var in self.sensor_vars.items() if var.get()]

    def _update_sensor_menu_text(self):
        selected = self._selected_capteurs()
        if not selected:
            self.sensor_menu_text.set("Aucun capteur sélectionné")
        elif len(selected) == 1:
            self.sensor_menu_text.set(selected[0])
        elif len(selected) <= 3:
            self.sensor_menu_text.set(", ".join(selected))
        else:
            self.sensor_menu_text.set(f"{len(selected)} capteurs sélectionnés")

    def _ouvrir_fenetre_capteurs(self):
        if self.sensor_window and self.sensor_window.winfo_exists():
            self.sensor_window.lift()
            self.sensor_window.focus_force()
            return

        self.pending_sensor_vars = {
            capteur: tk.BooleanVar(value=var.get()) for capteur, var in self.sensor_vars.items()
        }
        self.sensor_window = tk.Toplevel(self)
        self.sensor_window.title("Sélection des capteurs")
        self.sensor_window.transient(self)
        self.sensor_window.resizable(True, True)
        self.sensor_window.columnconfigure(0, weight=1)
        self.sensor_window.rowconfigure(0, weight=1)
        self.sensor_window.protocol("WM_DELETE_WINDOW", self._fermer_fenetre_capteurs)

        self.sensor_window_container = ttk.Frame(self.sensor_window)
        self.sensor_window_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sensor_window_container.columnconfigure(0, weight=1)

        actions = ttk.Frame(self.sensor_window)
        actions.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        ttk.Button(actions, text="Tout cocher", command=self._cocher_tous_capteurs).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Tout décocher", command=self._decocher_tous_capteurs).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Appliquer", command=self._appliquer_selection_capteurs).pack(side=tk.RIGHT, padx=4)

        self._rebuild_sensor_window_contents()

    def _rebuild_sensor_window_contents(self):
        if not self.sensor_window or not self.sensor_window.winfo_exists():
            return
        for child in self.sensor_window_container.winfo_children():
            child.destroy()

        columns = 4
        for col in range(columns):
            self.sensor_window_container.columnconfigure(col, weight=1)

        for index, capteur in enumerate(self.capteurs):
            ttk.Checkbutton(
                self.sensor_window_container,
                text=capteur,
                variable=self.pending_sensor_vars[capteur],
            ).grid(row=index // columns, column=index % columns, sticky="w", padx=6, pady=3)

    def _cocher_tous_capteurs(self):
        for var in self.pending_sensor_vars.values():
            var.set(True)

    def _decocher_tous_capteurs(self):
        for var in self.pending_sensor_vars.values():
            var.set(False)

    def _appliquer_selection_capteurs(self):
        for capteur, pending_var in self.pending_sensor_vars.items():
            if capteur in self.sensor_vars:
                self.sensor_vars[capteur].set(pending_var.get())
        self._fermer_fenetre_capteurs()
        self._on_capteurs_change()

    def _fermer_fenetre_capteurs(self):
        if self.sensor_window and self.sensor_window.winfo_exists():
            self.sensor_window.destroy()
        self.sensor_window = None
        self.pending_sensor_vars = {}

    def _mettre_a_jour_liste_capteurs(self):
        df = self._filtered_measurements()
        self.capteurs = sorted(df["Capteur"].dropna().astype(str).unique().tolist())
        self.sensor_vars = {}

        if not self.capteurs:
            self.sensor_menu_text.set("Aucun capteur disponible")
            if self.sensor_window and self.sensor_window.winfo_exists():
                self.sensor_window.destroy()
                self.sensor_window = None
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Aucun capteur pour ce système", ha="center", va="center")
            if self.canvas:
                self.canvas.draw_idle()
            return

        for index, capteur in enumerate(self.capteurs):
            var = tk.BooleanVar(value=index < min(3, len(self.capteurs)))
            self.sensor_vars[capteur] = var

        self._update_sensor_menu_text()
        self._rebuild_sensor_window_contents()
        self._mettre_a_jour_curseurs(reset_range=True)
        self.afficher_graphique(self._selected_capteurs())

    def _mettre_a_jour_curseurs(self, reset_range: bool = False):
        selected = self._selected_capteurs() or self.capteurs
        df = self._filtered_measurements(selected)
        if df.empty:
            return

        day_min = int(df["JourOrdinal"].min())
        day_max = int(df["JourOrdinal"].max())
        self.cur_debut.config(from_=day_min, to=day_max)
        self.cur_fin.config(from_=day_min, to=day_max)

        current_start = int(float(self.day_start.get() or day_min))
        current_end = int(float(self.day_end.get() or day_max))
        if reset_range:
            current_start, current_end = day_min, day_max
        else:
            current_start = min(max(current_start, day_min), day_max)
            current_end = min(max(current_end, day_min), day_max)
            if current_end < current_start:
                current_end = current_start

        self.day_start.set(current_start)
        self.day_end.set(current_end)
        self._update_day_labels(current_start, current_end)

    def _update_day_labels(self, start_day: int, end_day: int):
        self.lbl_debut.config(text=pd.Timestamp.fromordinal(int(start_day)).strftime("%Y-%m-%d"))
        self.lbl_fin.config(text=pd.Timestamp.fromordinal(int(end_day)).strftime("%Y-%m-%d"))

    def _sync_day_sliders_from_axes(self):
        if not self.fig.axes:
            return
        x0, x1 = self.fig.axes[0].get_xlim()
        dt0 = pd.Timestamp(mdates.num2date(min(x0, x1)).replace(tzinfo=None)).normalize()
        dt1 = pd.Timestamp(mdates.num2date(max(x0, x1)).replace(tzinfo=None)).normalize()
        start_day = dt0.toordinal()
        end_day = dt1.toordinal()
        self.day_start.set(start_day)
        self.day_end.set(end_day)
        self._update_day_labels(start_day, end_day)

    def _on_systeme_change(self):
        self._saved_views = None
        self._skip_restore_view = True
        self._mettre_a_jour_liste_capteurs()

    def _on_capteurs_change(self):
        self._update_sensor_menu_text()
        self._saved_views = None
        self._skip_restore_view = True
        self._mettre_a_jour_curseurs(reset_range=True)
        self._rafraichir()

    def _on_curseur_change(self, event=None):
        start_day = int(float(self.day_start.get()))
        end_day = int(float(self.day_end.get()))
        if end_day < start_day:
            end_day = start_day
            self.day_end.set(end_day)
        self._update_day_labels(start_day, end_day)
        self._skip_restore_view = True
        self._rafraichir()

    def _save_current_view(self):
        if not self.fig.axes:
            self._saved_views = None
            return
        self._saved_views = [ax.get_xlim() for ax in self.fig.axes]

    def _restore_view(self):
        if not self._saved_views or len(self._saved_views) != len(self.fig.axes):
            return
        for ax, xlim in zip(self.fig.axes, self._saved_views):
            ax.set_xlim(xlim)

    def _rafraichir(self):
        selected = self._selected_capteurs()
        if not selected:
            self._set_analysis_text("Aucun capteur sélectionné.")
            return
        if self._skip_restore_view:
            self.afficher_graphique(selected)
            self._skip_restore_view = False
        else:
            self._save_current_view()
            self.afficher_graphique(selected)
            self._restore_view()
        self._update_analysis(selected)
        if self.canvas:
            self.canvas.draw_idle()

    def _prepare_sensor_dataframe(self, capteur: str, apply_window: bool = False) -> pd.DataFrame:
        serie = self._filtered_measurements([capteur])[["Date", "Valeur"]].rename(columns={"Valeur": capteur}).sort_values("Date")
        if self.moyenne_journaliere.get():
            serie = serie.assign(Date=serie["Date"].dt.floor("D")).groupby("Date", as_index=False).mean(numeric_only=True)
        if serie.empty:
            return serie
        if self.lissage_actif.get():
            serie = appliquer_lissage(serie, capteur, window=7)

        merged = calculer_variation_annuelle(serie, capteur)
        if apply_window:
            start_day = int(float(self.day_start.get()))
            end_day = int(float(self.day_end.get()))
            start_dt = pd.Timestamp.fromordinal(start_day)
            end_dt = pd.Timestamp.fromordinal(end_day) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            mask = (merged["Date"] >= start_dt) & (merged["Date"] <= end_dt)
            return merged.loc[mask].copy()
        return merged.copy()

    @staticmethod
    def _baseline_value_at_window_start(df: pd.DataFrame, value_col: str, start_dt: pd.Timestamp) -> float | None:
        if df.empty:
            return None
        before = df[df["Date"] <= start_dt]
        if not before.empty:
            return float(before.iloc[-1][value_col])
        return float(df.iloc[0][value_col])

    def _sensor_info(self, capteur: str) -> dict[str, object]:
        return self.capteur_info.get(self._sensor_key(capteur), {})

    def _build_titles(self, capteur: str, type_cap: str) -> tuple[str, str, str | None]:
        if type_cap == "Capteur de déplacement":
            return f"Ouverture de la fissure {capteur}", "Ouverture (mm)", None
        if type_cap == "Corde vibrante":
            return f"Allongement de la corde vibrante {capteur}", "Allongement (microstrains)", None
        if type_cap == "Corde vibrante sur fissure":
            return f"Allongement de la corde vibrante {capteur}", "Allongement (microstrains)", f"Ouverture de la fissure {capteur}"
        if type_cap == "Capteur de température":
            return f"Température {capteur}", "Température (°C)", None
        if type_cap == "Inclinomètre":
            return f"Inclinaison du capteur {capteur}", "Inclinaison (degrés)", None
        if type_cap == "Alimentation":
            return "Alimentation de référence", "Tension (V)", None
        return f"Évolution du capteur : {capteur}", "Valeur", None

    def _plot_slopes(
        self,
        ax: plt.Axes,
        dates: pd.Series,
        primary: pd.Series,
        variation: pd.Series | None,
        unite: str,
        periode_label: str,
        color: str = "green",
        variation_color: str = "red",
        label_prefix: str = "",
    ):
        window_days = self.fenetre_mapping[self.fenetre_pente.get()]
        if len(dates) < 2:
            ax.set_title("Pas assez de points pour calculer une pente", fontsize=10)
            return

        slope_primary = pente_locale_sur_fenetre(dates, primary, window_days)
        valid_primary = slope_primary.notna()
        base_label = f"{label_prefix}Pente ({periode_label})" if label_prefix else f"Pente ({periode_label})"
        ax.plot(dates[valid_primary], slope_primary[valid_primary], linestyle="-", color=color, label=base_label)

        if variation is not None:
            slope_variation = pente_locale_sur_fenetre(dates, variation, window_days)
            valid_variation = slope_variation.notna()
            var_label = f"{label_prefix}Pente variation ({periode_label})" if label_prefix else f"Pente variation ({periode_label})"
            ax.plot(dates[valid_variation], slope_variation[valid_variation], linestyle="--", color=variation_color, label=var_label)

        ax.set_ylabel(f"Pente ({unite}/jour)" if unite else "Pente (/jour)")
        ax.set_xlabel("Date")
        ax.grid(True)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    def _set_analysis_text(self, text: str):
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", tk.END)
        self.analysis_text.insert("1.0", text)
        self.analysis_text.configure(state="disabled")

    def _analyser_capteur(self, capteur: str) -> str:
        df = self._prepare_sensor_dataframe(capteur, apply_window=False)
        if df.empty or len(df) < 2:
            return f"- {capteur} : données insuffisantes pour une analyse."

        work = df[["Date", capteur]].dropna().copy()
        if len(work) < 24:
            return f"- {capteur} : données insuffisantes pour une analyse."
        work["Year"] = work["Date"].dt.year
        work["Month"] = work["Date"].dt.month
        monthly_ref = work.groupby("Month")[capteur].median()
        work["Deseason"] = work[capteur] - work["Month"].map(monthly_ref)

        per_year = (
            work.groupby("Year")
            .agg(
                n=("Deseason", "size"),
                start=("Deseason", "first"),
                end=("Deseason", "last"),
                std=("Deseason", "std"),
                amp=(capteur, lambda s: float(s.max() - s.min())),
            )
            .reset_index()
        )
        if per_year.empty:
            return f"- {capteur} : données insuffisantes pour une analyse."

        per_year["delta"] = per_year["end"] - per_year["start"]
        noise = float(work["Deseason"].std(ddof=0))
        if np.isnan(noise) or noise <= 0:
            noise = float(np.median(np.abs(np.diff(work["Deseason"].to_numpy(dtype=float))))) if len(work) > 2 else 0.0
        noise = max(noise, 1e-9)

        latest = per_year.iloc[-1]
        x = (work["Date"] - work["Date"].min()).dt.total_seconds().to_numpy() / 86400.0
        y = work["Deseason"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if not np.allclose(x, x[0]) else 0.0

        seasonal_risk = abs(float(latest["delta"])) > (2.5 * noise)
        trend_risk = abs(slope) > (noise / 120.0)
        amplitude_risk = float(latest["amp"]) > (4.0 * noise)
        to_watch = seasonal_risk or trend_risk or amplitude_risk

        if to_watch:
            status = "⚠️ à surveiller"
        else:
            status = "✅ stable (corrigé saisonnier)"

        return (
            f"- {capteur} : {status} | année {int(latest['Year'])} Δcorr={float(latest['delta']):.4g} | "
            f"amp={float(latest['amp']):.4g} | pente_corr≈{slope:.4g}/jour"
        )

    def _update_analysis(self, capteurs: list[str]):
        if not capteurs:
            self._set_analysis_text("Aucun capteur sélectionné.")
            return
        analyses = [self._analyser_capteur(capteur) for capteur in capteurs]
        watch_list = [line for line in analyses if "à surveiller" in line]
        header = "Capteurs à surveiller :\n"
        if watch_list:
            summary = "\n".join(watch_list)
        else:
            summary = "Aucun capteur critique détecté sur l'analyse annuelle corrigée des saisonnalités."
        self._set_analysis_text(f"{header}{summary}\n\nDétail par capteur :\n" + "\n".join(analyses))

    def _exporter_selection(self):
        capteurs = self._selected_capteurs()
        if not capteurs:
            messagebox.showwarning("Export", "Sélectionnez au moins un capteur.")
            return
        dossier = filedialog.askdirectory(title="Choisir le dossier d'export")
        if not dossier:
            return
        for capteur in capteurs:
            self._exporter_un_capteur(dossier, capteur)
        messagebox.showinfo("Export", f"Export terminé pour {len(capteurs)} capteur(s).")

    def _exporter_tous_les_graphes(self):
        dossier = filedialog.askdirectory(title="Choisir le dossier d'export")
        if not dossier:
            return
        for capteur in self.capteurs:
            self._exporter_un_capteur(dossier, capteur)
        messagebox.showinfo("Export", f"Export terminé pour le système '{self.selected_systeme.get()}'.")

    def _exporter_un_capteur(self, dossier: str, capteur: str):
        df_merged = self._prepare_sensor_dataframe(capteur, apply_window=True)
        if df_merged.empty:
            return

        info = self._sensor_info(capteur)
        type_cap = str(info.get("type", "")).strip()
        bas = info.get("bas")
        haut = info.get("haut")
        unite_cap = str(info.get("unite", "")).strip()
        titre_principal, y_label, titre_secondaire = self._build_titles(capteur, type_cap)
        periode_label = self.fenetre_pente.get()

        fig_princ = plt.figure(figsize=(12, 8), dpi=300)
        gs = fig_princ.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax_g = fig_princ.add_subplot(gs[0, 0])
        ax_p = fig_princ.add_subplot(gs[1, 0])

        ax_g.plot(df_merged["Date"], df_merged[capteur], linestyle="-", color="#1f77b4", label=capteur)
        if self.afficher_variation.get():
            ax_g.plot(df_merged["Date"], df_merged["variation"], linestyle="--", color="orange", label="Variation (vs -1 an)")
        if bas is not None:
            ax_g.axhline(float(bas), color="red", linestyle="-", linewidth=1.2, label="Seuil bas")
        if haut is not None:
            ax_g.axhline(float(haut), color="red", linestyle="-", linewidth=1.2, label="Seuil haut")
        handles, labels = ax_g.get_legend_handles_labels()
        dedup = {}
        for h, l in zip(handles, labels):
            dedup.setdefault(l, h)
        ax_g.legend(dedup.values(), dedup.keys())
        ax_g.set_title(titre_principal, fontsize=14)
        ax_g.set_ylabel(y_label)
        ax_g.grid(True)

        self._plot_slopes(ax_p, df_merged["Date"], df_merged[capteur], df_merged["variation"] if self.afficher_variation.get() else None, unite_cap, periode_label)
        slope_handles, slope_labels = ax_p.get_legend_handles_labels()
        if slope_handles:
            ax_p.legend(slope_handles, slope_labels)

        date_min = df_merged["Date"].min().strftime("%Y%m%d")
        date_max = df_merged["Date"].max().strftime("%Y%m%d")
        systeme_suffix = self.selected_systeme.get().replace(os.sep, "_")
        fname1 = f"{systeme_suffix} - {capteur} - {date_min} - {date_max} - principal.png"
        fig_princ.savefig(os.path.join(dossier, fname1), bbox_inches="tight")
        plt.close(fig_princ)

        if type_cap == "Corde vibrante sur fissure":
            fig_sec = plt.figure(figsize=(12, 8), dpi=300)
            gs2 = fig_sec.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
            ax_r = fig_sec.add_subplot(gs2[0, 0])
            ax_pr = fig_sec.add_subplot(gs2[1, 0])
            df_merged["variation_conv"] = df_merged["variation"] * CONVERSION_CORDE_FISSURE
            serie_conv = df_merged[capteur] * CONVERSION_CORDE_FISSURE
            ax_r.plot(df_merged["Date"], serie_conv, linestyle="-", color="purple", label="Ouverture (mm)")
            if self.afficher_variation.get():
                ax_r.plot(df_merged["Date"], df_merged["variation_conv"], linestyle="--", color="orange", label="Variation (vs -1 an)")
            ax_r.set_title(titre_secondaire or "", fontsize=12)
            ax_r.set_ylabel("Ouverture (mm)")
            ax_r.grid(True)
            ax_r.legend()
            self._plot_slopes(ax_pr, df_merged["Date"], serie_conv, df_merged["variation_conv"] if self.afficher_variation.get() else None, "mm", periode_label)
            sec_handles, sec_labels = ax_pr.get_legend_handles_labels()
            if sec_handles:
                ax_pr.legend(sec_handles, sec_labels)
            fname2 = f"{systeme_suffix} - {capteur} - {date_min} - {date_max} - secondaire.png"
            fig_sec.savefig(os.path.join(dossier, fname2), bbox_inches="tight")
            plt.close(fig_sec)

    def afficher_graphique(self, capteurs: list[str]):
        if not capteurs:
            return

        series_map = {capteur: self._prepare_sensor_dataframe(capteur) for capteur in capteurs}
        series_map = {capteur: df for capteur, df in series_map.items() if not df.empty}
        if not series_map:
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Pas de données pour la sélection", ha="center", va="center")
            if self.canvas:
                self.canvas.draw_idle()
            return

        types = {str(self._sensor_info(capteur).get("type", "")).strip() for capteur in series_map}
        single_corde = len(series_map) == 1 and types == {"Corde vibrante sur fissure"}

        self.fig.clf()
        if single_corde:
            gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1], wspace=0.25, hspace=0.3)
            ax_main = self.fig.add_subplot(gs[0, 0])
            ax_right = self.fig.add_subplot(gs[0, 1], sharex=ax_main)
            ax_slope_main = self.fig.add_subplot(gs[1, 0], sharex=ax_main)
            ax_slope_right = self.fig.add_subplot(gs[1, 1], sharex=ax_right)
        else:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
            ax_main = self.fig.add_subplot(gs[0, 0])
            ax_right = None
            ax_slope_main = self.fig.add_subplot(gs[1, 0], sharex=ax_main)
            ax_slope_right = None

        start_dt = pd.Timestamp.fromordinal(int(float(self.day_start.get())))
        end_dt = pd.Timestamp.fromordinal(int(float(self.day_end.get()))) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        for index, (capteur, df_merged) in enumerate(series_map.items()):
            color = colors[index % len(colors)]
            info = self._sensor_info(capteur)
            bas = info.get("bas")
            haut = info.get("haut")
            unite_cap = str(info.get("unite", "")).strip()
            type_cap = str(info.get("type", "")).strip()
            titre_principal, y_label, titre_droit = self._build_titles(capteur, type_cap)

            serie_main = df_merged[capteur].copy()
            if self.demarrer_a_zero.get():
                base_val = self._baseline_value_at_window_start(df_merged, capteur, start_dt)
                if base_val is not None:
                    serie_main = serie_main - base_val

            ax_main.plot(df_merged["Date"], serie_main, linestyle="-", color=color, label=capteur)
            if self.afficher_variation.get():
                ax_main.plot(df_merged["Date"], df_merged["variation"], linestyle="--", color=color, alpha=0.75, label=f"{capteur} - variation")

            if len(series_map) == 1:
                if bas is not None:
                    ax_main.axhline(float(bas), color="red", linestyle="-", linewidth=1.2, label="Seuil bas")
                if haut is not None:
                    ax_main.axhline(float(haut), color="red", linestyle="-", linewidth=1.2, label="Seuil haut")
                ax_main.set_title(titre_principal, fontsize=14)
                ax_main.set_ylabel(y_label)
            else:
                ax_main.set_title(f"Système {self.selected_systeme.get()} - {len(series_map)} capteurs", fontsize=14)
                ax_main.set_ylabel("Valeur")

            self._plot_slopes(
                ax_slope_main,
                df_merged["Date"],
                serie_main,
                df_merged["variation"] if self.afficher_variation.get() else None,
                unite_cap,
                self.fenetre_pente.get(),
                color=color,
                variation_color=color,
                label_prefix=f"{capteur} - " if len(series_map) > 1 else "",
            )

            if single_corde and ax_right is not None and ax_slope_right is not None:
                df_merged["variation_conv"] = df_merged["variation"] * CONVERSION_CORDE_FISSURE
                serie_conv = serie_main * CONVERSION_CORDE_FISSURE
                ax_right.plot(df_merged["Date"], serie_conv, linestyle="-", color="purple", label="Ouverture (mm)")
                if self.afficher_variation.get():
                    ax_right.plot(df_merged["Date"], df_merged["variation_conv"], linestyle="--", color="orange", label="Variation (vs -1 an)")
                ax_right.set_title(titre_droit or "", fontsize=12)
                ax_right.set_ylabel("Ouverture (mm)")
                ax_right.grid(True)
                ax_right.legend()
                self._plot_slopes(ax_slope_right, df_merged["Date"], serie_conv, df_merged["variation_conv"] if self.afficher_variation.get() else None, "mm", self.fenetre_pente.get())

        handles, labels = ax_main.get_legend_handles_labels()
        dedup = {}
        for h, l in zip(handles, labels):
            dedup.setdefault(l, h)
        ax_main.legend(dedup.values(), dedup.keys())
        ax_main.grid(True)

        slope_handles, slope_labels = ax_slope_main.get_legend_handles_labels()
        slope_dedup = {}
        for h, l in zip(slope_handles, slope_labels):
            slope_dedup.setdefault(l, h)
        if slope_dedup:
            ax_slope_main.legend(slope_dedup.values(), slope_dedup.keys())
        ax_slope_main.grid(True)

        if single_corde and ax_slope_right is not None:
            right_handles, right_labels = ax_slope_right.get_legend_handles_labels()
            right_dedup = {}
            for h, l in zip(right_handles, right_labels):
                right_dedup.setdefault(l, h)
            if right_dedup:
                ax_slope_right.legend(right_dedup.values(), right_dedup.keys())
            ax_slope_right.relim()
            ax_slope_right.autoscale_view(scalex=False, scaley=True)

        ax_main.set_xlim(start_dt, end_dt)
        ax_slope_main.set_xlim(start_dt, end_dt)
        if ax_right is not None:
            ax_right.set_xlim(start_dt, end_dt)
        if ax_slope_right is not None:
            ax_slope_right.set_xlim(start_dt, end_dt)

        ax_slope_main.relim()
        ax_slope_main.autoscale_view(scalex=False, scaley=True)

        self.fig.autofmt_xdate()
        if self.canvas:
            self.canvas.draw_idle()


def main():
    try:
        loader = DatabaseLoader()
        data, capteur_info, systemes = loader.load()
    except Exception as exc:
        tk.Tk().withdraw()
        messagebox.showerror("Erreur de chargement", str(exc))
        return

    app = App(data, capteur_info, systemes)
    app.mainloop()


if __name__ == "__main__":
    main()
