import sqlite3
from pathlib import Path
import pandas as pd


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        if self.conn is not None:
            return self.conn

        DB_PATH.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.row_factory = sqlite3.Row
        return self.conn

    # --------------------------------------------------
    # CREATION DES TABLES
    # --------------------------------------------------

    def create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS systems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS sensors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            system_id INTEGER,
            name TEXT,
            x REAL,
            y REAL,
            type TEXT,
            unit TEXT,
            layer TEXT,
            date_calibration TEXT,

            A0 REAL,
            A1 REAL,
            A2 REAL,
            A3 REAL,
            A4 REAL,
            A5 REAL,

            seuil_bas REAL,
            seuil_haut REAL,
            demi_plage REAL,
            seuil_theorique REAL,

            UNIQUE(system_id, name),
            FOREIGN KEY(system_id) REFERENCES systems(id) ON DELETE CASCADE
        );
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id INTEGER,
            datetime TEXT,
            value REAL,
            unit TEXT,
            UNIQUE(sensor_id, datetime),
            FOREIGN KEY(sensor_id) REFERENCES sensors(id) ON DELETE CASCADE
        );
        """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensors_system_name ON sensors(system_id, name);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurements_sensor_datetime ON measurements(sensor_id, datetime);"
        )

        self.conn.commit()

    # --------------------------------------------------
    # INSERTION SYSTEME
    # --------------------------------------------------

    def insert_system(self, name):
        cursor = self.conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO systems (name) VALUES (?)", (name,))
        self.conn.commit()

        cursor.execute("SELECT id FROM systems WHERE name = ?", (name,))
        return cursor.fetchone()["id"]

    # --------------------------------------------------
    # INSERTION CAPTEUR (AVEC CALIBRATION)
    # --------------------------------------------------

    def insert_sensor(self, system_id, name, x, y, sensor_type, unit=None, layer=None, calibration=None):
        cursor = self.conn.cursor()

        calibration = calibration or {}

        # Insertion ou ignore
        cursor.execute(
            """
            INSERT INTO sensors (
                system_id, name, x, y, type, unit, layer,
                date_calibration, A0, A1, A2, A3, A4, A5,
                seuil_bas, seuil_haut, demi_plage, seuil_theorique
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(system_id, name) DO UPDATE SET
                x = excluded.x,
                y = excluded.y,
                type = excluded.type,
                unit = excluded.unit,
                layer = excluded.layer
            """,
            (
                system_id,
                name,
                x,
                y,
                sensor_type,
                unit,
                layer,
                calibration.get("date_calibration"),
                calibration.get("A0"),
                calibration.get("A1"),
                calibration.get("A2"),
                calibration.get("A3"),
                calibration.get("A4"),
                calibration.get("A5"),
                calibration.get("seuil_bas"),
                calibration.get("seuil_haut"),
                calibration.get("demi_plage"),
                calibration.get("seuil_theorique"),
            ),
        )

        self.conn.commit()

        # Toujours récupérer l'id après (même si déjà existant)
        cursor.execute(
            """
            SELECT id FROM sensors
            WHERE system_id = ? AND name = ?
        """,
            (system_id, name),
        )

        result = cursor.fetchone()

        if result is None:
            raise Exception(f"Impossible de récupérer l'id du capteur {name}")

        return result["id"]

    # --------------------------------------------------
    # INSERTION MESURE
    # --------------------------------------------------

    def insert_measurements_batch(self, data_list):
        cursor = self.conn.cursor()

        cursor.executemany(
            """
            INSERT OR IGNORE INTO measurements (sensor_id, datetime, value, unit)
            VALUES (
                ?, ?, ?, 
                (SELECT unit FROM sensors WHERE id = ?)
            )
            """,
            [(sid, dt, val, sid) for sid, dt, val, _ in data_list],
        )

        self.conn.commit()

    # --------------------------------------------------
    # LECTURE
    # --------------------------------------------------

    def get_systems(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM systems")
        return cursor.fetchall()

    def get_sensors_by_system(self, system_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sensors WHERE system_id = ?", (system_id,))
        return cursor.fetchall()

    def get_measurements_by_sensor(self, sensor_id):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT datetime, value, unit
            FROM measurements
            WHERE sensor_id = ?
            ORDER BY datetime
        """,
            (sensor_id,),
        )
        return cursor.fetchall()

    # --------------------------------------------------
    def get_mapped_sensors(self, resolution=0.01, min_systems=2):
        """
        Regroupe les capteurs par position arrondie (mapping spatial).

        resolution: pas spatial (ex: 0.01)
        min_systems: nombre minimal de systèmes distincts dans un groupe
        """
        if resolution <= 0:
            raise ValueError("resolution doit être > 0")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            WITH mapped AS (
                SELECT
                    s.id AS sensor_id,
                    s.name AS sensor_name,
                    sys.name AS system_name,
                    ROUND(s.x / ?, 0) * ? AS x_map,
                    ROUND(s.y / ?, 0) * ? AS y_map
                FROM sensors s
                JOIN systems sys ON sys.id = s.system_id
                WHERE s.x IS NOT NULL AND s.y IS NOT NULL
            )
            SELECT
                x_map,
                y_map,
                COUNT(*) AS sensor_count,
                COUNT(DISTINCT system_name) AS system_count,
                GROUP_CONCAT(system_name || ':' || sensor_name, ' | ') AS sensors
            FROM mapped
            GROUP BY x_map, y_map
            HAVING COUNT(DISTINCT system_name) >= ?
            ORDER BY x_map, y_map
            """,
            (resolution, resolution, resolution, resolution, min_systems),
        )
        return cursor.fetchall()

    # --------------------------------------------------
    # GENERATION DES FICHIERS EXCEL MODELES
    # --------------------------------------------------

    def generate_excel_templates(self, folder="excel_templates"):
        """
        Génère un fichier Excel modèle pour chaque table
        afin d'assurer le bon format d'import des données.
        """

        folder_path = Path(folder)
        folder_path.mkdir(exist_ok=True)

        templates = {
            "systems": [
                "name"
            ],

            "sensors": [
                "system_name",
                "name",
                "x",
                "y",
                "type",
                "unit",
                "layer",
                "date_calibration",
                "A0",
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
                "seuil_bas",
                "seuil_haut",
                "demi_plage",
                "seuil_theorique"
            ],

            "measurements": [
                "system_name",
                "sensor_name",
                "datetime",
                "value",
                "unit"
            ]
        }

        for table, columns in templates.items():
            df = pd.DataFrame(columns=columns)
            file_path = folder_path / f"{table}_template.xlsx"
            df.to_excel(file_path, index=False)

        print(f"Templates Excel générés dans : {folder_path.resolve()}")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


def initialize_database():
    db = DatabaseManager()
    db.create_tables()
    #db.generate_excel_templates()
    db.close()


if __name__ == "__main__":
    initialize_database()
    print("Base de données initialisée.")
