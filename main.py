from pathlib import Path
import yaml
import multiprocessing
import os
import subprocess

from systems.monitoring_systems import (
    GeometreSystem,
    TS15System,
    SAPHIRSystem
)

from ui.matrice_app import MultiSystemMappingApp


IMPORT_DATA = True # recrée la base de donnée

BASE_DIR = Path(r"R:\2 - Surveillance\surveillance_app")
DATA_DIR = BASE_DIR / ".Donnees"

SENSORS_DIR = DATA_DIR / "Reperage capteurs"
MEASURES_DIR = DATA_DIR / "Mesures"


def load_config():

    config_file = DATA_DIR / "systems_config.yaml"

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_system(name, cfg):

    sensors = SENSORS_DIR / cfg["sensors"]

    if cfg["type"] in ["geometre", "itmsol"]:

        measures = MEASURES_DIR / cfg["measures"]
        return GeometreSystem(name, sensors, measures)

    if cfg["type"] == "ts15":

        measures = MEASURES_DIR / cfg["measures"]
        return TS15System(sensors, measures)

    if cfg["type"] == "saphir":

        folder = Path(cfg["measures_folder"])
        return SAPHIRSystem(sensors, folder)

def reset_database():
    db_path = BASE_DIR / "data" / "surveillance.db"
    db_script = BASE_DIR / "database" / "db.py"

    # Supprimer la base si elle existe
    if db_path.exists():
        print(f"Suppression de la base : {db_path}")
        os.remove(db_path)

    # Recréer la base en lançant db.py
    print("Recréation de la base via db.py...")
    subprocess.run(["python", str(db_script)], check=True)

    print("Base recréée avec succès\n")

def import_all_data():

    config = load_config()

    for system_name, cfg in config.items():

        print(f"\nImport {system_name}")

        system = create_system(system_name, cfg)

        system.import_to_database()
        system.close()

    print("\nImport terminé")


if __name__ == "__main__":

    multiprocessing.freeze_support()

    if IMPORT_DATA:
        reset_database()   # 🔥 AJOUT ICI
        import_all_data()

    app = MultiSystemMappingApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()