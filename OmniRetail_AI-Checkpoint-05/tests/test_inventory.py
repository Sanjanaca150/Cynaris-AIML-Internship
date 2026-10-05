from app.inventory import generate_inventory_alerts


def test_inventory_has_20_skus():
    df = generate_inventory_alerts()

    assert len(df) == 20


def test_inventory_statuses():
    df = generate_inventory_alerts()

    valid_statuses = {
        "HEALTHY",
        "REORDER",
        "CRITICAL",
    }

    assert set(df["status"]).issubset(valid_statuses)