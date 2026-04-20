from typing import Optional, Tuple


def build_query(intent: str, entities: dict) -> Tuple[Optional[str], tuple, str]:
    month_filter = entities.get("time_filter")
    search_term = entities.get("search_term")
    supplier_name = entities.get("supplier_name")
    location_name = entities.get("location_name")
    shipment_id = entities.get("shipment_id")
    comparison_months = entities.get("comparison_months", [])

    if intent == "supplier_count":
        return "SELECT COUNT(DISTINCT supplier) AS supplier_count FROM products", (), "supplier_count"

    if intent == "supplier_list":
        return "SELECT DISTINCT supplier FROM products WHERE supplier IS NOT NULL AND TRIM(supplier) != '' ORDER BY supplier", (), "supplier_list"

    if intent == "product_count":
        return "SELECT COUNT(DISTINCT name) AS product_count FROM products", (), "product_count"

    if intent == "filtered_product_count":
        if not search_term:
            return None, (), "filtered_product_count"
        return (
            """
            SELECT sku, name, category, quantity, location, supplier
            FROM products
            WHERE LOWER(name) LIKE ?
               OR LOWER(category) LIKE ?
            ORDER BY name
            """.strip(),
            (f"%{search_term}%", f"%{search_term}%"),
            "filtered_product_count",
        )

    if intent == "filtered_product_list":
        if supplier_name:
            return (
                """
                SELECT sku, name, category, quantity, location, supplier
                FROM products
                WHERE LOWER(supplier) LIKE ?
                ORDER BY name
                """.strip(),
                (f"%{supplier_name.lower()}%",),
                "filtered_product_list_supplier",
            )
        if location_name:
            return (
                """
                SELECT sku, name, category, quantity, location, supplier
                FROM products
                WHERE LOWER(location) LIKE ?
                ORDER BY name
                """.strip(),
                (f"%{location_name.lower()}%",),
                "filtered_product_list_location",
            )
        if not search_term:
            return None, (), "filtered_product_list"
        return (
            """
            SELECT sku, name, category, quantity, location, supplier
            FROM products
            WHERE LOWER(name) LIKE ?
               OR LOWER(category) LIKE ?
            ORDER BY name
            """.strip(),
            (f"%{search_term}%", f"%{search_term}%"),
            "filtered_product_list",
        )

    if intent == "product_list":
        return (
            """
            SELECT DISTINCT name, category, location, supplier
            FROM products
            ORDER BY name
            """.strip(),
            (),
            "product_list",
        )

    if intent == "product_type_summary":
        return (
            """
            SELECT category, COUNT(*) AS product_count
            FROM products
            GROUP BY category
            ORDER BY product_count DESC
            """.strip(),
            (),
            "product_type_summary",
        )

    if intent == "location_count":
        return "SELECT COUNT(DISTINCT location) AS location_count FROM products", (), "location_count"

    if intent == "rack_type_count":
        return (
            """
            SELECT COUNT(DISTINCT
                CASE
                    WHEN INSTR(location, '-') > 0 THEN SUBSTR(location, 1, INSTR(location, '-') - 1)
                    ELSE location
                END
            ) AS rack_type_count
            FROM products
            WHERE location IS NOT NULL AND TRIM(location) != ''
            """.strip(),
            (),
            "rack_type_count",
        )

    if intent == "location_lookup":
        if not search_term:
            return None, (), "location_lookup"
        return (
            """
            SELECT sku, name, location, quantity, supplier
            FROM products
            WHERE LOWER(name) LIKE ?
            ORDER BY name
            """.strip(),
            (f"%{search_term}%",),
            "location_lookup",
        )

    if intent == "location_usage_summary":
        if location_name:
            return (
                """
                SELECT sku, name, category, quantity, location, supplier
                FROM products
                WHERE LOWER(location) LIKE ?
                ORDER BY name
                """.strip(),
                (f"%{location_name.lower()}%",),
                "location_usage_by_location",
            )
        return (
            """
            SELECT location, COUNT(*) AS product_count, SUM(quantity) AS total_quantity
            FROM products
            WHERE location IS NOT NULL AND TRIM(location) != ''
            GROUP BY location
            ORDER BY product_count DESC, total_quantity DESC
            """.strip(),
            (),
            "location_usage_summary",
        )

    if intent == "sales_total":
        sql = "SELECT name, SUM(quantity_sold) AS quantity_sold, SUM(revenue) AS revenue FROM sales"
        params = ()
        if month_filter:
            sql += " WHERE sale_date LIKE ?"
            params = (f"{month_filter}%",)
        sql += " GROUP BY name ORDER BY quantity_sold DESC"
        return sql, params, "sales_total"

    if intent == "sales_top_item":
        sql = "SELECT name, SUM(quantity_sold) AS quantity_sold, SUM(revenue) AS revenue FROM sales"
        params = ()
        if month_filter:
            sql += " WHERE sale_date LIKE ?"
            params = (f"{month_filter}%",)
        sql += " GROUP BY name ORDER BY quantity_sold DESC LIMIT 1"
        return sql, params, "sales_top_item"

    if intent == "sales_by_month_summary":
        if len(comparison_months) >= 2:
            placeholders = ",".join(["?"] * len(comparison_months))
            sql = (
                f"SELECT SUBSTR(sale_date, 1, 7) AS month, "
                f"SUM(quantity_sold) AS total_quantity_sold, "
                f"SUM(revenue) AS total_revenue "
                f"FROM sales "
                f"WHERE SUBSTR(sale_date, 1, 7) IN ({placeholders}) "
                f"GROUP BY month ORDER BY month"
            )
            return sql, tuple(comparison_months), "sales_by_month_comparison"
        return (
            """
            SELECT SUBSTR(sale_date, 1, 7) AS month,
                   SUM(quantity_sold) AS total_quantity_sold,
                   SUM(revenue) AS total_revenue
            FROM sales
            GROUP BY month
            ORDER BY total_quantity_sold DESC
            """.strip(),
            (),
            "sales_by_month_summary",
        )

    if intent == "inflow_total":
        sql = "SELECT name, SUM(quantity_in) AS quantity_in FROM inflow"
        params = ()
        if month_filter:
            sql += " WHERE inflow_date LIKE ?"
            params = (f"{month_filter}%",)
        sql += " GROUP BY name ORDER BY quantity_in DESC"
        return sql, params, "inflow_total"

    if intent == "outflow_total":
        sql = "SELECT name, SUM(quantity_out) AS quantity_out FROM outflow"
        params = ()
        if month_filter:
            sql += " WHERE outflow_date LIKE ?"
            params = (f"{month_filter}%",)
        sql += " GROUP BY name ORDER BY quantity_out DESC"
        return sql, params, "outflow_total"

    if intent == "low_stock":
        return (
            """
            SELECT sku, name, category, quantity, reorder_threshold, location, supplier,
                   (reorder_threshold - quantity) AS stock_gap
            FROM products
            WHERE quantity < reorder_threshold
            ORDER BY stock_gap DESC, quantity ASC
            """.strip(),
            (),
            "low_stock",
        )

    if intent == "delayed_shipments":
        return (
            """
            SELECT shipment_id, supplier, status, expected_date, notes
            FROM shipments
            WHERE LOWER(status) = 'delayed'
            """.strip(),
            (),
            "delayed_shipments",
        )

    if intent == "shipment_status_lookup":
        if not shipment_id:
            return None, (), "shipment_status_lookup"
        return (
            """
            SELECT shipment_id, supplier, status, expected_date, notes
            FROM shipments
            WHERE shipment_id = ?
            """.strip(),
            (shipment_id,),
            "shipment_status_lookup",
        )

    if intent == "stock_level_lookup":
        if not search_term:
            return None, (), "stock_level_lookup"
        return (
            """
            SELECT sku, name, quantity, reorder_threshold, location, supplier
            FROM products
            WHERE LOWER(name) LIKE ?
            ORDER BY quantity DESC, name
            """.strip(),
            (f"%{search_term}%",),
            "stock_level_lookup",
        )

    if intent == "supplier_risk":
        return (
            """
            SELECT supplier, COUNT(*) AS low_stock_items
            FROM products
            WHERE quantity < reorder_threshold
            GROUP BY supplier
            ORDER BY low_stock_items DESC, supplier ASC
            """.strip(),
            (),
            "supplier_risk",
        )

    if intent == "category_summary":
        return (
            """
            SELECT category, SUM(quantity) AS total_quantity
            FROM products
            GROUP BY category
            ORDER BY total_quantity DESC
            """.strip(),
            (),
            "category_summary",
        )

    if intent == "zero_stock":
        return (
            """
            SELECT sku, name, category, quantity, location, supplier
            FROM products
            WHERE quantity = 0
            ORDER BY name
            """.strip(),
            (),
            "zero_stock",
        )

    if intent == "restock_priority":
        return (
            """
            SELECT sku, name, category, quantity, reorder_threshold, location, supplier,
                   (reorder_threshold - quantity) AS stock_gap
            FROM products
            WHERE quantity < reorder_threshold
            ORDER BY stock_gap DESC, quantity ASC
            """.strip(),
            (),
            "restock_priority",
        )

    if intent == "unsold_items":
        month_value = month_filter or entities.get("fallback_month")
        if not month_value:
            return None, (), "unsold_items"
        return "SELECT DISTINCT name FROM sales WHERE sale_date LIKE ?", (f"{month_value}%",), "unsold_items_seed"

    return None, (), "unsupported"
