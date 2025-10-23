import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# --- 1. SKATER RATING MODEL (Per-Minute Talent via Regression) ---

def generate_skater_ratings(path="skaters.csv"):
    """
    Trains a stable Gradient Boosting (XGBoost-style) model on per-60 stats
    to generate an Overall Talent Rating (50-100) for ALL skaters.
    """
    print("Generating ratings for ALL skaters...")
    
    df = pd.read_csv(path).dropna(subset=["gameScore"])
    
    # 1. Aggregate and Clean Data
    agg_dict = {c: "sum" for c in df.columns if c not in ["playerId","name","team","position","season","situation"]}
    for pct in ["onIce_xGoalsPercentage","onIce_corsiPercentage","onIce_fenwickPercentage"]:
        if pct in df.columns:
            agg_dict[pct] = "mean"
            
    players = df.groupby(["playerId","name","team","position"], as_index=False).agg(agg_dict)
    players = players[players["icetime"] > 0].copy()

    # 2. Calculate Per-60 Rate Features
    count_feats = [
        "I_F_goals","I_F_primaryAssists","I_F_secondaryAssists","I_F_points",
        "I_F_xGoals","I_F_highDangerGoals","I_F_takeaways", "I_F_giveaways",
        "I_F_shotsOnGoal"
    ]
    for feat in count_feats:
        if feat in players.columns:
            players[f"{feat}_per60"] = players[feat] / (players["icetime"] / 3600.0)
            
    # 3. Define Features for the Model (RATE-ONLY)
    rate_features = [f"{f}_per60" for f in count_feats]
    pct_features = ["onIce_xGoalsPercentage", "onIce_fenwickPercentage", "onIce_corsiPercentage"]
    features = [f for f in rate_features + pct_features if f in players.columns]
    
    players["pos_D"] = (players["position"] == "D").astype(int)
    if "pos_D" not in features:
        features.append("pos_D")

    model_df = players.dropna(subset=features + ["gameScore"]).copy()

    # 4. Train/Predict with Gradient Boosting (XGBoost-style model)
    X = model_df[features].astype(float)
    y = model_df["gameScore"].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, _, y_train, _ = train_test_split(X_scaled, y.values, test_size=0.20, random_state=42)

    model = HistGradientBoostingRegressor(max_iter=100, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred_full = model.predict(scaler.transform(model_df[features]))
    
    # Scale Prediction to 50-100 (Overall Talent Rating)
    mm = MinMaxScaler((50,100))
    model_df["Overall_Talent_Rating"] = mm.fit_transform(y_pred_full.reshape(-1,1)).flatten()

    # 5. Create Interpretable Metrics
    model_df["Offensive_Rate"] = (model_df["I_F_goals_per60"] * 0.6) + (model_df["I_F_primaryAssists_per60"] * 0.4)
    model_df["xGD_OnIce_Per_100"] = (model_df["onIce_xGoalsPercentage"] - 0.5) * 100 
    
    # 6. Final Output
    output_cols = ["playerId", "name", "team", "position", 
                   "Overall_Talent_Rating", "Offensive_Rate", 
                   "xGD_OnIce_Per_100", "I_F_takeaways_per60",
                   "games_played", "icetime"]
    
    final_skaters = model_df[output_cols].sort_values("Overall_Talent_Rating", ascending=False).copy()
    
    final_skaters.to_csv("all_skaters_ratings_final.csv", index=False, float_format="%.2f")
    print(f"Skaters: Saved {len(final_skaters)} players to all_skaters_ratings_final.csv")
    return final_skaters.head(5).to_string(index=False, float_format="%.2f")

# --- 2. GOALIE MODEL (Shrunk GSAx for Stable Efficiency) ---
def generate_goalie_ratings(path="goalies.csv"):
    """
    Calculates Bayesian-shrunk GSAx for stable, per-minute goalie performance for ALL players.
    """
    print("Generating ratings for ALL goalies...")
    
    goalies = pd.read_csv(path)
    
    # Define constants for shrinkage
    MIN_ICETIME_SECONDS = 3600 * 10 
    PRIOR_TIME_SECONDS = 3600 * 100 
    PRIOR_GSAX_RATE = 0.0 
    
    goalies = goalies[goalies["situation"] == "all"].copy()
    goalies = goalies[goalies["icetime"] >= MIN_ICETIME_SECONDS].copy()

    goalies["icetime_hours"] = goalies["icetime"] / 3600.0

    # 1. Calculate GSAx rate (unshrunk)
    goalies["GSAx"] = goalies["xGoals"] - goalies["goals"]
    goalies["GSAx_rate"] = goalies["GSAx"] / goalies["icetime_hours"]

    # 2. Apply Bayesian Shrinkage to GSAx Rate
    goalies["GSAx_Per60_Stabilized"] = (
        (goalies["GSAx_rate"] * goalies["icetime_hours"]) + 
        (PRIOR_GSAX_RATE * (PRIOR_TIME_SECONDS / 3600.0))
    ) / (goalies["icetime_hours"] + (PRIOR_TIME_SECONDS / 3600.0))

    # 3. Create Interpretable Metrics
    goalies["SV_Pct"] = 1 - (goalies["goals"] / goalies["unblocked_shot_attempts"])
    goalies["xSV_Pct"] = 1 - (goalies["xGoals"] / goalies["xOnGoal"])
    
    goalies["SPOx_Per_1000"] = (goalies["SV_Pct"] - goalies["xSV_Pct"]) * 1000 
    
    # 4. Final Output
    output_cols = ["playerId", "name", "team", "GSAx_Per60_Stabilized", 
                   "SPOx_Per_1000", "SV_Pct", "games_played", "icetime"]
    
    final_goalies = goalies[output_cols].sort_values("GSAx_Per60_Stabilized", ascending=False).copy()
    
    final_goalies.to_csv("all_goalies_ratings_final.csv", index=False, float_format="%.3f")
    print(f"Goalies: Saved {len(final_goalies)} players to all_goalies_ratings_final.csv")
    return final_goalies.head(5).to_string(index=False, float_format="%.3f")

# --- Execute Rating System ---
if __name__ == '__main__':
    skater_head = generate_skater_ratings(path="data/skaters.csv")
    goalie_head = generate_goalie_ratings(path="data/goalies.csv")
    
    print("\n--- Summary of Generated Data ---")
    print("\n[all_skaters_ratings_final.csv] Sample:")
    print(skater_head)
    print("\n[all_goalies_ratings_final.csv] Sample:")
    print(goalie_head)
    print("\nProcessing complete. Check your directory for the two CSV files.")