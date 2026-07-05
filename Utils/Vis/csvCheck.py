import pandas as pd
df = pd.read_csv("run_history.csv")

print("Total rows:", len(df))
print("Rows with real training data:", df["losses/policy_loss"].notna().sum())
print("Rows with only system telemetry:", df["losses/policy_loss"].isna().sum())