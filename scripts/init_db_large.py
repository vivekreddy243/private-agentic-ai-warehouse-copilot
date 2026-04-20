import sqlite3
import random
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path("data/warehouse.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)

PRODUCT_PREFIXES = [
    "Wireless", "Mechanical", "Portable", "Industrial", "Smart", "Heavy-Duty",
    "Thermal", "Barcode", "Compact", "Advanced", "Ergonomic", "Digital"
]
PRODUCT_TYPES = [
    "Mouse", "Keyboard", "Scanner", "Printer", "Monitor", "SSD", "Laptop Stand",
    "Headphones", "Webcam", "Dock", "Cable", "Router", "Battery Pack", "Label Roll",
    "Tape", "Tablet", "Sensor", "Reader", "Hub", "Adapter"
]
CATEGORIES = [
    "Electronics", "Accessories", "Warehouse Supplies", "Networking", "Office Devices"
]
SUPPLIERS = [
    "TechSupply", "KeyMasters", "ConnectPro", "ErgoDesk", "SoundMax",
    "VisionTech", "DataStore", "ScanFlow", "PrintCo", "PackMart",
    "WireHub", "OfficeCore", "LogiWorld", "PrimeParts", "ByteWorks"
]
LOCATIONS = [f"Rack-{row}{col}" for row in "ABCDE" for col in range(1, 41)]

DB_PATH.unlink(missing_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE products (
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
CREATE TABLE shipments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id TEXT UNIQUE,
    supplier TEXT,
    status TEXT,
    expected_date TEXT,
    notes TEXT
)
""")

cur.execute("""
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    doc_type TEXT,
    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

cur.execute("""
CREATE TABLE sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT,
    name TEXT,
    quantity_sold INTEGER,
    revenue REAL,
    sale_date TEXT
)
""")

cur.execute("""
CREATE TABLE outflow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT,
    name TEXT,
    quantity_out INTEGER,
    destination TEXT,
    outflow_date TEXT
)
""")

cur.execute("""
CREATE TABLE inflow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT,
    name TEXT,
    quantity_in INTEGER,
    source TEXT,
    inflow_date TEXT
)
""")

products = []
used_names = set()

for i in range(1, 1001):
    prefix = random.choice(PRODUCT_PREFIXES)
    ptype = random.choice(PRODUCT_TYPES)
    name = f"{prefix} {ptype}"
    if name in used_names:
        name = f"{name} {i}"
    used_names.add(name)

    sku = f"SKU-{1000 + i}"
    category = random.choice(CATEGORIES)
    quantity = random.randint(0, 120)
    reorder_threshold = random.randint(8, 35)
    location = random.choice(LOCATIONS)
    supplier = random.choice(SUPPLIERS)

    products.append((sku, name, category, quantity, reorder_threshold, location, supplier))

cur.executemany("""
INSERT INTO products (sku, name, category, quantity, reorder_threshold, location, supplier)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", products)

shipment_rows = []
statuses = ["On Time", "Delayed", "Dispatched", "Pending"]
for i in range(1, 401):
    shipment_id = f"SHIP-{2000 + i}"
    supplier = random.choice(SUPPLIERS)
    status = random.choices(statuses, weights=[45, 20, 20, 15])[0]
    expected_date = (datetime(2026, 4, 1) + timedelta(days=random.randint(0, 60))).strftime("%Y-%m-%d")
    notes = f"{supplier} shipment status: {status}"
    shipment_rows.append((shipment_id, supplier, status, expected_date, notes))

cur.executemany("""
INSERT INTO shipments (shipment_id, supplier, status, expected_date, notes)
VALUES (?, ?, ?, ?, ?)
""", shipment_rows)

sales_rows = []
outflow_rows = []
inflow_rows = []

product_sample = [(p[0], p[1], p[6]) for p in products]

for sku, name, supplier in product_sample:
    base_price = random.randint(20, 600)

    for month in [1, 2, 3, 4]:
        for _ in range(random.randint(1, 4)):
            day = random.randint(1, 28)
            qty_sold = random.randint(1, 20)
            revenue = qty_sold * base_price
            sales_rows.append((sku, name, qty_sold, float(revenue), f"2026-{month:02d}-{day:02d}"))

        for _ in range(random.randint(1, 3)):
            day = random.randint(1, 28)
            qty_out = random.randint(1, 18)
            destination = random.choice(["Retail Dispatch", "Warehouse Transfer", "Client Delivery", "Internal Use"])
            outflow_rows.append((sku, name, qty_out, destination, f"2026-{month:02d}-{day:02d}"))

        for _ in range(random.randint(1, 3)):
            day = random.randint(1, 28)
            qty_in = random.randint(5, 30)
            inflow_rows.append((sku, name, qty_in, supplier, f"2026-{month:02d}-{day:02d}"))

cur.executemany("""
INSERT INTO sales (sku, name, quantity_sold, revenue, sale_date)
VALUES (?, ?, ?, ?, ?)
""", sales_rows)

cur.executemany("""
INSERT INTO outflow (sku, name, quantity_out, destination, outflow_date)
VALUES (?, ?, ?, ?, ?)
""", outflow_rows)

cur.executemany("""
INSERT INTO inflow (sku, name, quantity_in, source, inflow_date)
VALUES (?, ?, ?, ?, ?)
""", inflow_rows)

conn.commit()
conn.close()

print("Large warehouse database created successfully with 1000 products.")