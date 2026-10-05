from app.forecasting import train_forecast


if __name__ == "__main__":
    result = train_forecast("SKU-001", test_days=60)

    print()
    print("Prophet Demand Forecast")
    print("-----------------------")
    print("SKU:", result["sku_id"])
    print("Training rows:", result["train_rows"])
    print("Testing rows:", result["test_rows"])
    print("MAPE:", result["mape"], "%")
    print("Model saved:", result["model_path"])