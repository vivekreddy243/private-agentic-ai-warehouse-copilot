import sqlite3
from pathlib import Path

DB_PATH = Path("data/warehouse.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT UNIQUE,
    name TEXT NOT NULL,
    category TEXT,
    quantity INTEGER DEFAULT 0,
    reorder_threshold INTEGER DEFAULT 10,
    location TEXT,
    supplier TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS shipments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id TEXT UNIQUE,
    supplier TEXT,
    status TEXT,
    expected_date TEXT,
    notes TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    doc_type TEXT,
    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

sample_products = [
    ("SKU-101", "Wireless Mouse", "Electronics", 12, 10, "Rack-A1", "TechSupply"),
    ("SKU-102", "Mechanical Keyboard", "Electronics", 8, 10, "Rack-A2", "KeyMasters"),
    ("SKU-103", "USB-C Hub", "Accessories", 15, 12, "Rack-A3", "ConnectPro"),
    ("SKU-104", "Laptop Stand", "Accessories", 5, 8, "Rack-B1", "ErgoDesk"),
    ("SKU-105", "Noise-Cancelling Headphones", "Electronics", 4, 6, "Rack-B2", "SoundMax"),
    ("SKU-106", "Webcam", "Electronics", 10, 8, "Rack-B3", "VisionTech"),
    ("SKU-107", "Portable SSD", "Electronics", 6, 10, "Rack-C1", "DataStore"),
    ("SKU-108", "Barcode Scanner", "Warehouse Supplies", 3, 5, "Rack-C2", "ScanFlow"),
    ("SKU-109", "Thermal Printer", "Warehouse Supplies", 2, 4, "Rack-C3", "PrintCo"),
    ("SKU-110", "Packing Tape", "Warehouse Supplies", 25, 15, "Rack-D1", "PackMart")
]

sample_shipments = [
    ("SHIP-001", "TechSupply", "Delayed", "2026-04-10", "Wireless Mouse shipment delayed due to weather"),
    ("SHIP-002", "KeyMasters", "On Time", "2026-04-09", "Mechanical Keyboard shipment arriving on schedule"),
    ("SHIP-003", "SoundMax", "Delayed", "2026-04-11", "Headphones shipment delayed by supplier"),
    ("SHIP-004", "PackMart", "On Time", "2026-04-08", "Packing Tape shipment dispatched")
]

cur.executemany("""
INSERT OR IGNORE INTO products
(sku, name, category, quantity, reorder_threshold, location, supplier)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", sample_products)

cur.executemany("""
INSERT OR IGNORE INTO shipments
(shipment_id, supplier, status, expected_date, notes)
VALUES (?, ?, ?, ?, ?)
""", sample_shipments)

conn.commit()
conn.close()

print("Database initialized successfully at data/warehouse.db")