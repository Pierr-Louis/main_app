
import os
import numpy as np
import pandas as pd
from pathlib import Path
from systems.base_system import format_duration


from systems.base_system import BaseMonitoringSystem


def load_sensors_template(file_path):

    df = pd.read_excel(file_path, engine="openpyxl")

    df.columns = [str(c).strip().lower() for c in df.columns]

    # colonnes possibles du template
    sensor_col = "name"
    x_col = "x"
    y_col = "y"
    type_col = "type"
    unit_col = "unit"
    layer_col = "layer"

    df_points = pd.DataFrame({
        "sensor": df[sensor_col].astype(str).str.strip().str.upper(),
        "x": pd.to_numeric(df.get(x_col), errors="coerce"),
        "y": pd.to_numeric(df.get(y_col), errors="coerce"),
        "type": df.get(type_col),
        "unit": df.get(unit_col),
        "layer": df.get(layer_col)
    })

    return df_points

UNIT_FACTORS = {
    "GEOMETRE": 1000,
    "ITMSOL": 1,
    "TS15": 1,
    "SAPHIR": 1
}

def read_one_saphir(file_path):

    from pathlib import Path
    import pandas as pd

    try:
        with open(file_path, "r", encoding="latin1") as f:
            line = f.readline()
            sep = ";" if ";" in line else ","

        df = pd.read_csv(
            file_path,
            sep=sep,
            encoding="latin1",
            engine="c",
            usecols=[0, 1],
            header=0
        )

        if df.shape[1] < 2:
            return file_path, None

        df.columns = ["datetime", "value"]

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            format="mixed",   # pandas récent
            dayfirst=True,
            errors="coerce"
        )

        df["value"] = (
            df["value"]
            .astype(str)
            .str.replace(",", ".")
        )

        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df["sensor"] = Path(file_path).stem.upper()

        return file_path, df

    except:
        return file_path, None

def detect_anomalies(df):

    # calcul par capteur
    def mad_group(group):

        median = group["value"].median()
        mad = (group["value"] - median).abs().median()

        if mad == 0:
            group["anomaly"] = False
            return group

        z = 0.6745 * (group["value"] - median) / mad

        group["anomaly"] = z.abs() > 5   # seuil robuste

        return group

    df = df.groupby("sensor", group_keys=False).apply(mad_group)

    return df

def adaptive_compression(df):

    df = df.sort_values(["sensor", "datetime"])

    result = []

    for sensor, group in df.groupby("sensor"):

        group = group.sort_values("datetime")

        threshold = group["value"].std() * 0.2

        group["diff"] = group["value"].diff().abs()

        mask = (group["diff"] > threshold)

        if "anomaly" in group.columns:
            mask |= group["anomaly"]

        important = group[mask]

        group["hour"] = group["datetime"].dt.floor("H")

        idx_min = group.groupby("hour")["value"].idxmin()
        idx_max = group.groupby("hour")["value"].idxmax()

        extrema = group.loc[idx_min.tolist() + idx_max.tolist()]

        merged = pd.concat([important, extrema], ignore_index=True)

        merged = merged.drop_duplicates(
            subset=["sensor", "datetime", "value"]
        )

        result.append(merged)

    df_final = pd.concat(result)

    return df_final.sort_values(["sensor", "datetime"])
    


# =========================================================
# GEOMETRE / ITMSOL
# =========================================================

class GeometreSystem(BaseMonitoringSystem):

    def __init__(self, system_name, points_path, measures_path):
        super().__init__(system_name)
        self.points_path = points_path
        self.measures_path = measures_path

    def load_points(self):
        return load_sensors_template(self.points_path)
        return self.points_df

    def load_measurements(self):

        df = pd.read_excel(self.measures_path, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]

        df.rename(columns={df.columns[0]: "sensor"}, inplace=True)

        df["sensor"] = df["sensor"].astype(str).str.strip().str.upper()

        for col in df.columns[1:]:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".")
                .replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
            )

            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def normalize_data(self):

        points = self.load_points()
        measures = self.load_measurements()

        df = measures.melt(id_vars=["sensor"], var_name="datetime", value_name="value")

        df = df.dropna(subset=["value"])

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        df = df.dropna(subset=["datetime"])

        # conversion m -> mm uniquement pour géomètre
        df["value"] = df["value"] * UNIT_FACTORS[self.system_name]

        df = df.merge(points, on="sensor", how="left")

        df["calibration"] = None

        return df[["datetime","sensor","x","y","type","layer","value","unit","calibration"]]

# =========================================================
# TS15
# =========================================================

class TS15System(BaseMonitoringSystem):

    def __init__(self, points_path, measures_path):
        super().__init__("TS15")
        self.points_path = points_path
        self.measures_path = measures_path
        self.points_df = None
        self.measures_df = None

    def load_points(self):
        self.points_df = load_sensors_template(self.points_path)
        return self.points_df

    def load_measurements(self):

        df = pd.read_excel(self.measures_path, engine="openpyxl")

        if "Date" not in df.columns:
            raise ValueError("Colonne 'Date' manquante dans le fichier TS15")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        for col in df.columns:
            if col != "Date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df_long = df.melt(
            id_vars=["Date"],
            var_name="sensor",
            value_name="value"
        )

        df_long.rename(columns={"Date": "datetime"}, inplace=True)

        df_long["sensor"] = df_long["sensor"].astype(str).str.strip().str.upper()

        df_long = df_long.dropna(subset=["datetime", "value"])

        self.measures_df = df_long

        return df_long

    def normalize_data(self):

        if self.points_df is None:
            self.load_points()

        if self.measures_df is None:
            self.load_measurements()

        df = self.measures_df.merge(self.points_df, on="sensor", how="left")

        df["value"] = df["value"] * UNIT_FACTORS[self.system_name]
        df["calibration"] = None

        return df[
            ["datetime","sensor","x","y","type","layer","value","unit","calibration"]
        ]


# =========================================================
# SAPHIR
# =========================================================

class SAPHIRSystem(BaseMonitoringSystem):

    def __init__(self, points_path, measures_dir):
        super().__init__("SAPHIR")

        self.points_path = points_path
        self.measures_dir = measures_dir

        self.points_df = None
        self.measures_df = None

    def load_points(self):

        self.points_df = load_sensors_template(self.points_path)
        
        if "calibration" not in self.points_df.columns:
            self.points_df["calibration"] = [{} for _ in range(len(self.points_df))]

        return self.points_df

    def load_measurements(self):

        import time
        from multiprocessing import Pool, cpu_count
        from pathlib import Path

        files = []

        # 🔍 scan fichiers
        for root, dirs, filenames in os.walk(self.measures_dir):
            if "csv nettoyés" in root.lower():
                for f in filenames:
                    if f.lower().endswith(".csv"):
                        files.append(os.path.join(root, f))

        total = len(files)

        print(f"Chargement SAPHIR : {total} fichiers...")

        if total == 0:
            self.measures_df = pd.DataFrame()
            return self.measures_df

        # ⚠️ réseau → limiter CPU
        n_workers = min(4, cpu_count() - 1)

        print(f"Utilisation de {n_workers} coeurs CPU")

        start_time = time.time()

        processed = 0
        valid = 0

        # ⚡ optimisation mémoire
        batch_size = 500
        dfs_batch = []
        all_data = []

        with Pool(n_workers) as pool:

            for file_path, result in pool.imap_unordered(
                read_one_saphir,
                files,
                chunksize=5   # ⚡ bon compromis
            ):

                processed += 1

                if result is not None:
                    dfs_batch.append(result)
                    valid += 1

                '''# ⚡ concat par batch (évite gros lag final)
                if len(dfs_batch) >= batch_size:
                    all_data.append(pd.concat(dfs_batch, ignore_index=True))
                    dfs_batch = []'''

                # 📊 progression fluide
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (total - processed) / rate if rate > 0 else 0
                percent = (processed / total) * 100

                filename = Path(file_path).name
                if len(filename) > 40:
                    filename = filename[:37] + "..."

                print(
                    f"\r[{percent:6.2f}%] {processed}/{total} | restant ~{format_duration(remaining)} | {filename}",
                    end=" " * 20,
                    flush=True,
                )

        print()

        # concat final léger
        if dfs_batch:
            all_data.append(pd.concat(dfs_batch, ignore_index=True))

        if not all_data:
            print("⚠️ Aucun fichier exploitable")
            self.measures_df = pd.DataFrame()
            return self.measures_df

        print("\nConcat final...")

        data = pd.concat(all_data, ignore_index=True, copy=False)

        # nettoyage minimal
        data = data.dropna(subset=["datetime", "value"])

        print(f"""
    Résumé SAPHIR :
    ✔ fichiers valides : {valid}
    ❌ fichiers ignorés : {total - valid}
    ✔ {len(data)} lignes chargées
    """)

        self.measures_df = data

        return data

    def normalize_data(self):

        if self.points_df is None:
            self.load_points()

        if self.measures_df is None:
            self.load_measurements()

        if self.measures_df is None or self.measures_df.empty:
            print("⚠️ Aucune donnée SAPHIR à normaliser")
            return pd.DataFrame()

        df = self.measures_df.merge(self.points_df, on="sensor", how="left")

        # 🔥 1. détecter anomalies
        df = detect_anomalies(df)

        # 🔥 2. compression intelligente
        df = adaptive_compression(df)

        df["value"] = df["value"] * UNIT_FACTORS[self.system_name]

        return df[["datetime","sensor","x","y","type","layer","value","unit","calibration"]]