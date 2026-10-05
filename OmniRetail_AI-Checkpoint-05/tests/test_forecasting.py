import pandas as pd

from app.forecasting import calculate_mape


def test_calculate_mape():
    actual = pd.Series([10, 20, 30])
    predicted = pd.Series([10, 18, 33])

    mape = calculate_mape(actual, predicted)

    assert mape >= 0
    assert mape < 20


def test_demand_data_exists():
    df = pd.read_csv("data/demand_data.csv")

    assert len(df) > 0
    assert df["sku_id"].nunique() == 20