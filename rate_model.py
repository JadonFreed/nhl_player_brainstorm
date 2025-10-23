import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# --- EA Sports NHL 26 Player Overall Ratings (OVR) for comparison ---
# This list represents a professional, consensus ranking of overall talent.
EA_RATINGS = [
    ("Connor McDavid", 97), ("Nathan MacKinnon", 96), ("Leon Draisaitl", 96),
    ("Nikita Kucherov", 96), ("Quinn Hughes", 95), ("Cale Makar", 95),
    ("Aleksander Barkov", 95), ("Sidney Crosby", 94), ("David Pastrnak", 94),
    ("Jack Eichel", 94), ("Auston Matthews", 94), ("Kirill Kaprizov", 94),
    ("Mikko Rantanen", 93), ("Matthew Tkachuk", 93), ("Sam Reinhart", 93),
    ("Jack Hughes", 93), ("Zach Werenski", 92), ("Roman Josi", 92),
    ("Victor Hedman", 92), ("Rasmus Dahlin", 92), ("Brayden Point", 92),
    ("Artemi Panarin", 92), ("William Nylander", 92), ("Mitch Marner", 92),
    ("Kyle Connor", 92), ("Brad Marchand", 91), ("Elias Pettersson", 91),
    ("Alex Ovechkin", 90), ("Filip Forsberg", 90), ("Mark Stone", 90)
]

def prepare_data(path="data/skaters.csv"):
    """Loads, aggregates, and computes per-60 stats, excluding cumulative features."""
    df = pd.read_csv(path).dropna(subset=["gameScore"])
    
    # Aggregate over player across seasons (sum counts, mean percentages)
    agg_sum = {c: "sum" for c in df.columns if c not in ["playerId","name","team","position","season","situation"]}
    for pct in ["onIce_xGoalsPercentage","onIce_corsiPercentage","onIce_fenwickPercentage"]:
        if pct in df.columns:
            agg_sum[pct] = "mean"
            
    players = df.groupby(["playerId","name","position"], as_index=False).agg(agg_sum)
    players = players[players["icetime"] > 0].copy()

    # Compute per60 for important counting stats
    count_feats = [
        "I_F_goals","I_F_primaryAssists","I_F_secondaryAssists","I_F_points",
        "I_F_xGoals","I_F_highDangerShots","I_F_highDangerGoals","I_F_takeaways",
        "I_F_giveaways","I_F_hits","I_F_shotsOnGoal","I_F_shifts","shotsBlockedByPlayer"
    ]
    for feat in count_feats:
        if feat in players.columns:
            players[f"{feat}_per60"] = players[feat] / (players["icetime"] / 3600.0)
            
    # Define features for the RATE-ONLY model (EXCLUDES icetime_minutes)
    reduced_features = [
        "I_F_xGoals_per60", "I_F_primaryAssists_per60", "I_F_secondaryAssists_per60",
        "I_F_points_per60", "onIce_xGoalsPercentage", "onIce_fenwickPercentage", "I_F_highDangerShots_per60",
        "shotsBlockedByPlayer_per60", "I_F_shotsOnGoal_per60", "I_F_takeaways_per60", "I_F_shifts_per60",
        "I_F_hits_per60", "onIce_corsiPercentage"
    ]
    
    features = [f for f in reduced_features if f in players.columns]
    
    # Add a simple "is defenseman" feature
    players["pos_D"] = (players["position"] == "D").astype(int)
    if "pos_D" not in features:
        features.append("pos_D")

    players = players.dropna(subset=features + ["gameScore"]).copy()
    return players, features

def train_and_compare(players, features):
    """Trains the model and performs comparison against EA ratings."""
    X = players[features].astype(float)
    y = players["gameScore"].astype(float)

    # Scale and split data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.20, random_state=42)

    # Train Gradient Boosting Model (as a robust predictor)
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0,1]

    # Full predictions -> scale to 50-100 rating
    y_pred_full = model.predict(scaler.transform(players[features]))
    mm = MinMaxScaler((50,100))
    ratings_rate = mm.fit_transform(y_pred_full.reshape(-1,1)).flatten()
    players["rating_rate_50_100"] = ratings_rate

    # --- Comparison against EA Ratings ---
    ea_df = pd.DataFrame(EA_RATINGS, columns=["name_ea", "rating_ea"])
    ea_df["name_key"] = ea_df["name_ea"].str.lower().str.replace(r"[^a-z ]","", regex=True).str.strip()
    
    players["name_key"] = players["name"].str.lower().str.replace(r"[^a-z ]","", regex=True).str.strip()
    merged = pd.merge(ea_df, players, on="name_key", how="inner")
    
    if merged.empty:
        print("\nFATAL: No players matched between your data and the EA Ratings list.")
        return

    # Compute Rank Correlation (Spearman)
    spearman_corr, _ = spearmanr(merged["rating_ea"], merged["rating_rate_50_100"])
    
    # Top-20 overlap
    ea_top20 = ea_df.sort_values("rating_ea", ascending=False).head(20)["name_key"].tolist()
    model_top20 = players.sort_values("rating_rate_50_100", ascending=False).head(20)["name_key"].tolist()
    overlap = set(ea_top20).intersection(model_top20)

    # --- Output Results ---
    print(f"\n--- Rate-Only Model Performance ---")
    print(f"Model R^2 (vs cumulative GameScore): {r2:.3f}")
    print(f"Players used for rating: {players.shape[0]}")
    
    print(f"\n--- External Validation (vs EA NHL 26 OVR) ---")
    print(f"Matched Players: {merged.shape[0]}")
    print(f"Spearman Rank Correlation: {spearman_corr:.3f} (Rank agreement)")
    print(f"Top-20 Overlap: {len(overlap)}/{len(ea_top20)} players")

    print("\nTop 20 Players by Your Custom Rating:")
    display_cols = ["name", "position", "rating_rate_50_100", "rating_ea"]
    final_table = merged.sort_values("rating_rate_50_100", ascending=False)
    
    # Add non-matched players back for a full list
    full_list = players.merge(merged[["playerId", "rating_ea"]], on="playerId", how="left")
    full_list = full_list[["name", "position", "rating_rate_50_100", "rating_ea"]].sort_values("rating_rate_50_100", ascending=False).head(20)

    print(full_list.to_string(index=False, float_format="%.2f"))

if __name__ == '__main__':
    players, features = prepare_data()
    train_and_compare(players, features)