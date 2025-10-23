import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# --- EA Sports NHL 26 Goalie Overall Ratings (OVR) for comparison ---
EA_GOALIE_RATINGS = [
    ("Connor Hellebuyck", 94), ("Andrei Vasilevskiy", 93), ("Igor Shesterkin", 92),
    ("Ilya Sorokin", 91), ("Sergei Bobrovsky", 90), ("Jake Oettinger", 90),
    ("Filip Gustavsson", 89), ("Thatcher Demko", 88), ("Juuse Saros", 88),
    ("Darcy Kuemper", 87), ("Dustin Wolf", 87), ("Frederik Andersen", 87),
    ("Jacob Markstrom", 87), ("Linus Ullmark", 87), ("Jordan Binnington", 86),
    ("Logan Thompson", 86), ("Mackenzie Blackwood", 86), ("Jeremy Swayman", 85)
]

def create_goalie_ratings(path="data/goalies.csv"):
    """
    Loads goalie data, computes performance rates, generates a 50-100 rating,
    and validates against external EA ratings.
    """
    try:
        goalies = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        return None

    # Filter to 'all' situations for total season performance
    goalies = goalies[goalies["situation"] == "all"].copy()
    
    # Filter for minimum ice time stability (3600 seconds/hr * 10 hrs ≈ 10 games)
    MIN_ICETIME_SECONDS = 3600 * 10 
    goalies = goalies[goalies["icetime"] >= MIN_ICETIME_SECONDS].copy()
    
    if goalies.empty:
        print("Not enough qualifying goalies (minimum 10 games of ice time equivalent).")
        return None

    goalies["icetime_hours"] = goalies["icetime"] / 3600.0

    # --- Core Metrics: Goals Saved Above Expected (GSAx) per 60 ---
    if "xGoals" in goalies.columns and "goals" in goalies.columns:
        # For goalies, MoneyPuck stats are usually AGINST (xGA, GA)
        goalies["GSAx"] = goalies["xGoals"] - goalies["goals"]
        goalies["GSAx_per60"] = goalies["GSAx"] / goalies["icetime_hours"]
    else:
        # Fallback if column names are slightly different (using OnIce_A for goalie data)
        if "OnIce_A_xGoals" in goalies.columns and "OnIce_A_goals" in goalies.columns:
             goalies["GSAx"] = goalies["OnIce_A_xGoals"] - goalies["OnIce_A_goals"]
        else:
            goalies["GSAx"] = 0
            
        goalies["GSAx_per60"] = goalies["GSAx"] / goalies["icetime_hours"]

    # --- Efficiency Metrics ---
    # Raw Save Percentage (SV%)
    goalies["SV_Pct"] = 1 - (goalies["goals"] / goalies["unblocked_shot_attempts"])
    
    # Expected Save Percentage (xSV%) - using xGoals/xOnGoal to see if a goalie saves shots better than league expectation
    if "xOnGoal" in goalies.columns:
        goalies["xSV_Pct"] = 1 - (goalies["xGoals"] / goalies["xOnGoal"])
    else:
        goalies["xSV_Pct"] = goalies["SV_Pct"] # Default to raw SV% if xOnGoal is missing

    # --- Performance Metric (Composite Score) ---
    # Heavily rewards GSAx, moderately rewards outperforming expectation, and ensures good raw SV% is present.
    goalies["rating_raw"] = (
        goalies["GSAx_per60"] * 5.0 +                     
        (goalies["SV_Pct"] - goalies["xSV_Pct"]) * 100 +  
        goalies["SV_Pct"] * 50                          
    )

    # --- Normalize to 50-100 Rating Scale ---
    mm = MinMaxScaler((50, 100))
    goalies["rating_raw"] = goalies["rating_raw"].fillna(goalies["rating_raw"].median()) 
    
    goalies["goalie_rating"] = mm.fit_transform(goalies[["rating_raw"]]).flatten()
    goalies["goalie_rank"] = goalies["goalie_rating"].rank(ascending=False, method="min").astype(int)

    # --- Validation against EA Ratings ---
    ea_df = pd.DataFrame(EA_GOALIE_RATINGS, columns=["name_ea", "rating_ea"])
    ea_df["name_key"] = ea_df["name_ea"].str.lower().str.replace(r"[^a-z ]","", regex=True).str.strip()
    
    goalies["name_key"] = goalies["name"].str.lower().str.replace(r"[^a-z ]","", regex=True).str.strip()
    merged = pd.merge(ea_df, goalies, on="name_key", how="inner")
    
    # Compute Rank Correlation (Spearman)
    spearman_corr, _ = spearmanr(merged["rating_ea"], merged["goalie_rating"])
    
    # --- Output Results ---
    print("\n--- Goalie Model Validation ---")
    print(f"Qualifying Goalies (Min {MIN_ICETIME_SECONDS/3600} Hrs TOI): {goalies.shape[0]}")
    print(f"Matched Players for Validation: {merged.shape[0]}")
    print(f"Spearman Rank Correlation (Model vs EA): {spearman_corr:.3f}")

    print("\nTop 10 Goalies by Custom Rating:")
    output_cols = ["name", "team", "GSAx_per60", "SV_Pct", "goalie_rating", "rating_ea", "goalie_rank"]
    
    # Display columns present in the dataframe
    final_output = merged[[col for col in output_cols if col in merged.columns]].sort_values("goalie_rating", ascending=False).head(10)
    
    # Merge back in top goalies not in the EA list for a complete view
    top_10_model_only = goalies[~goalies['playerId'].isin(merged['playerId'])].sort_values("goalie_rating", ascending=False).head(10)
    final_output = pd.concat([final_output, top_10_model_only[["name", "team", "GSAx_per60", "SV_Pct", "goalie_rating", "goalie_rank"]].assign(rating_ea=np.nan)], ignore_index=True).head(10)

    print(final_output.to_string(index=False, float_format="%.3f"))
    
    return goalies

create_goalie_ratings()