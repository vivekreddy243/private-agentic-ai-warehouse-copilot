from app.agents.tool_helpers import evaluate_forecast_model


def main():
    result = evaluate_forecast_model()
    metrics = result["metrics"]
    details_df = result["details_df"]

    print("ML Forecast Evaluation")
    print("----------------------")
    print(f"MAE: {metrics['MAE']}")
    print(f"MSE: {metrics['MSE']}")
    print(f"R2 : {metrics['R2']}")
    print()

    if details_df is not None and not details_df.empty:
        print("Per-product evaluation:")
        print(details_df.to_string(index=False))
    else:
        print("No evaluation details available.")


if __name__ == "__main__":
    main()