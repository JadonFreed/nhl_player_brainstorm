import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore")

# --- SKATER MODEL (Per-Minute Talent via Regression) ---

def generate_skater_ratings(path="data/skaters.csv"):
    """
    Trains a regression model on rate stats (per 60) to predict overall talent,
    and calculates easy-to-interpret offensive and defensive metrics.
    """
    print("--- 1. Generating Skater Ratings (Per-Minute Talent) ---")
    
    df = pd.read_csv(path).dropna(subset=["gameScore"])
    
    # Aggregate data across all seasons/situations for player totals/averages
    agg_dict = {c: "sum" for c in df.columns if c not in ["playerId","name","team","position","season","situation"]}
    for pct in ["onIce_xGoalsPercentage","onIce_corsiPercentage","onIce_fenwickPercentage"]:
        if pct in df.columns:
            agg_dict[pct] = "mean"
            
    players = df.groupby(["playerId","name","position"], as_index=False).agg(agg_dict)
    players = players[players["icetime"] > 0].copy()

    # Calculate key per-60 counting stats (features for model)
    count_feats = [
        "I_F_goals","I_F_primaryAssists","I_F_secondaryAssists","I_F_points",
        "I_F_xGoals","I_F_highDangerGoals","I_F_takeaways", "I_F_giveaways",
        "I_F_shotsOnGoal"
    ]
    for feat in count_feats:
        if feat in players.columns:
            players[f"{feat}_per60"] = players[feat] / (players["icetime"] / 3600.0)
            
    # Define features for the RATE-ONLY model (Excludes deployment/icetime)
    rate_features = [f"{f}_per60" for f in count_feats]
    pct_features = ["onIce_xGoalsPercentage", "onIce_fenwickPercentage", "onIce_corsiPercentage"]
    features = [f for f in rate_features + pct_features if f in players.columns]
    
    # Add Defenseman dummy for role adjustment
    players["pos_D"] = (players["position"] == "D").astype(int)
    if "pos_D" not in features:
        features.append("pos_D")

    model_df = players.dropna(subset=features + ["gameScore"]).copy()

    # --- Train Regression Model (Predicts relative talent) ---
    X = model_df[features].astype(float)
    y = model_df["gameScore"].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Training on a subset for robust fitting, predicting on the full set
    X_train, _, y_train, _ = train_test_split(X_scaled, y.values, test_size=0.20, random_state=42)

    model = HistGradientBoostingRegressor(max_iter=100, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred_full = model.predict(scaler.transform(model_df[features]))
    
    # Scale Prediction to 50-100 range
    mm = MinMaxScaler((50,100))
    model_df["Overall_Talent_Rating"] = mm.fit_transform(y_pred_full.reshape(-1,1)).flatten()

    # --- Create Interpretable Metrics ---
    model_df["Offensive_Rate"] = (model_df["I_F_goals_per60"] * 0.6) + (model_df["I_F_primaryAssists_per60"] * 0.4)
    model_df["xGD_OnIce_Per_100"] = (model_df["onIce_xGoalsPercentage"] - 0.5) * 100 
    
    output_cols = ["name", "position", "Overall_Talent_Rating", "Offensive_Rate", "xGD_OnIce_Per_100", "I_F_takeaways_per60"]
    final_skaters = model_df.sort_values("Overall_Talent_Rating", ascending=False).head(20).copy()
    
    print("Skaters (Top 20):\n", final_skaters[output_cols].to_string(index=False, float_format="%.2f"))
    print("\n------------------------------------------------------------")
    return final_skaters

# --- GOALIE MODEL (Shrunk GSAx for Stable Efficiency) ---
def generate_goalie_ratings(path="data/goalies.csv"):
    """
    Calculates Bayesian-shrunk GSAx for stable, per-minute goalie performance.
    """
    print("--- 2. Generating Goalie Ratings (Stable Performance) ---")
    
    goalies = pd.read_csv(path)
    
    # Define constants for shrinkage (100 hours is a typical stabilization factor)
    MIN_ICETIME_SECONDS = 3600 * 10 
    PRIOR_TIME_SECONDS = 3600 * 100 
    PRIOR_GSAX_RATE = 0.0 # League average GSAx is zero
    
    goalies = goalies[goalies["situation"] == "all"].copy()
    goalies = goalies[goalies["icetime"] >= MIN_ICETIME_SECONDS].copy()

    goalies["icetime_hours"] = goalies["icetime"] / 3600.0

    # Calculate GSAx rate (unshrunk)
    goalies["GSAx"] = goalies["xGoals"] - goalies["goals"]
    goalies["GSAx_rate"] = goalies["GSAx"] / goalies["icetime_hours"]

    # Apply Bayesian Shrinkage to GSAx Rate
    goalies["GSAx_Per60_Stabilized"] = (
        (goalies["GSAx_rate"] * goalies["icetime_hours"]) + 
        (PRIOR_GSAX_RATE * (PRIOR_TIME_SECONDS / 3600.0))
    ) / (goalies["icetime_hours"] + (PRIOR_TIME_SECONDS / 3600.0))

    # --- Create Interpretable Metrics ---
    goalies["SV_Pct"] = 1 - (goalies["goals"] / goalies["unblocked_shot_attempts"])
    goalies["xSV_Pct"] = 1 - (goalies["xGoals"] / goalies["xOnGoal"])
    
    # Save Percentage Over Expected (SPOx)
    goalies["SPOx_Per_1000"] = (goalies["SV_Pct"] - goalies["xSV_Pct"]) * 1000 
    
    output_cols = ["name", "team", "GSAx_Per60_Stabilized", "SPOx_Per_1000", "SV_Pct"]
    final_goalies = goalies.sort_values("GSAx_Per60_Stabilized", ascending=False).head(10).copy()
    
    print("Goalies (Top 10):\n", final_goalies[output_cols].to_string(index=False, float_format="%.3f"))
    print("\n------------------------------------------------------------")
    return final_goalies

# --- Execute Rating System ---
if __name__ == '__main__':
    skaters_data = generate_skater_ratings()
    goalies_data = generate_goalie_ratings()
    print("Rating computation complete. Results displayed above.")