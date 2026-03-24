#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 🔥 FIX IMPORT PROJET
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.db import DatabaseManager


# =========================================================
# 🔹 DB → DataFrame
# =========================================================

def charger_donnees_systeme(db, system_name):

    query = """
    SELECT 
        m.datetime,
        s.name as sensor,
        m.value
    FROM measurements m
    JOIN sensors s ON m.sensor_id = s.id
    JOIN systems sys ON s.system_id = sys.id
    WHERE sys.name = ?
    """

    df = pd.read_sql(query, db.conn, params=(system_name,))
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df


def pivot_data(df):

    if df.empty:
        return pd.DataFrame()

    df_pivot = df.pivot_table(
        index="datetime",
        columns="sensor",
        values="value",
        aggfunc="mean"
    ).reset_index()

    df_pivot.rename(columns={"datetime": "Date"}, inplace=True)

    return df_pivot


# =========================================================
# 🔹 Lissage
# =========================================================

def appliquer_lissage(df, colonnes, window=7):
    df_l = df.copy()
    for col in colonnes:
        df_l[col] = df_l[col].rolling(window=window, center=True, min_periods=1).mean()
    return df_l


# =========================================================
# 🔹 UI COMPLET (reprise Affichage.py)
# =========================================================

class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Courbes capteurs (DB)")
        self.geometry("1200x950")

        self.db = DatabaseManager()

        self.systemes = ["SAPHIR", "TS15", "GEOMETRE"]
        self.selected_system = tk.StringVar(value=self.systemes[0])

        self.data = pd.DataFrame()
        self.capteurs = []

        self.lissage_actif = tk.BooleanVar(value=False)
        self.afficher_variation = tk.BooleanVar(value=False)

        self.fenetre_mapping = {"1 mois": 30, "3 mois": 90, "12 mois": 365}
        self.fenetre_pente = tk.StringVar(value="1 mois")

        self.annee_debut = tk.IntVar()
        self.annee_fin = tk.IntVar()

        self._creer_widgets()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._on_system_change()

    # -----------------------------------------------------
    def _on_close(self):
        self.destroy()
        sys.exit(0)

    # -----------------------------------------------------
    def _creer_widgets(self):

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(top, text="Système :").pack(side=tk.LEFT)
        cb_sys = ttk.Combobox(top, textvariable=self.selected_system,
                              values=self.systemes, state="readonly", width=12)
        cb_sys.pack(side=tk.LEFT)
        cb_sys.bind("<<ComboboxSelected>>", lambda e: self._on_system_change())

        ttk.Label(top, text="Capteur :").pack(side=tk.LEFT, padx=10)
        self.selected_capteur = tk.StringVar()

        self.cb_capteur = ttk.Combobox(top, textvariable=self.selected_capteur,
                                      state="readonly", width=40)
        self.cb_capteur.pack(side=tk.LEFT)
        self.cb_capteur.bind("<<ComboboxSelected>>", lambda e: self._on_capteur_change())

        ttk.Checkbutton(top, text="Lissage 7j",
                        variable=self.lissage_actif,
                        command=self._refresh).pack(side=tk.LEFT, padx=10)

        ttk.Checkbutton(top, text="Variation -1 an",
                        variable=self.afficher_variation,
                        command=self._refresh).pack(side=tk.LEFT)

        ttk.Combobox(top, textvariable=self.fenetre_pente,
                     values=list(self.fenetre_mapping.keys()),
                     state="readonly", width=8).pack(side=tk.LEFT, padx=10)

        ttk.Button(top, text="Exporter",
                   command=self._exporter).pack(side=tk.LEFT, padx=10)

        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    # -----------------------------------------------------
    def _on_system_change(self):

        df = charger_donnees_systeme(self.db, self.selected_system.get())

        if df.empty:
            messagebox.showwarning("Vide", "Aucune donnée")
            return

        self.data = pivot_data(df)
        self.capteurs = [c for c in self.data.columns if c != "Date"]

        self.cb_capteur["values"] = self.capteurs

        if self.capteurs:
            self.cb_capteur.current(0)
            self._on_capteur_change()

    # -----------------------------------------------------
    def _on_capteur_change(self):
        self._refresh()

    # -----------------------------------------------------
    def _refresh(self):

        cap = self.selected_capteur.get()

        if not cap:
            return

        df = self.data[["Date", cap]].dropna()

        if df.empty:
            return

        if self.lissage_actif.get():
            df = appliquer_lissage(df, [cap])

        # variation
        df_prev = df.copy()
        df_prev["Date"] += pd.DateOffset(years=1)
        df_prev.rename(columns={cap: "prev"}, inplace=True)

        df = df.merge(df_prev, on="Date", how="left")
        df["variation"] = df[cap] - df["prev"]

        # affichage
        self.fig.clf()

        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        ax1.plot(df["Date"], df[cap], label=cap)

        if self.afficher_variation.get():
            ax1.plot(df["Date"], df["variation"], label="variation")

        ax1.legend()
        ax1.grid(True)

        # pente
        window = self.fenetre_mapping[self.fenetre_pente.get()]

        def pente(w):
            if len(w) < 2:
                return np.nan
            x = np.arange(len(w))
            return np.polyfit(x, w, 1)[0]

        slope = df[cap].rolling(window, min_periods=2).apply(pente, raw=True)

        ax2.plot(df["Date"], slope, color="green")
        ax2.set_title("Pente")
        ax2.grid(True)

        self.canvas.draw()

    # -----------------------------------------------------
    def _exporter(self):
        messagebox.showinfo("Info", "Export à compléter si besoin")


# =========================================================
# MAIN
# =========================================================

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()