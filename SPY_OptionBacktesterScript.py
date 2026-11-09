from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ==========================================
# 1. FETCH ACTUAL HISTORICAL SPY & VIX DATA
# ==========================================
print("Fetching real historical data from Yahoo Finance...")
# Download SPY and VIX (Implied Volatility proxy) daily data
spy_data = yf.download("SPY", start="2022-01-01", end="2026-01-01")['Close']
vix_data = yf.download("^VIX", start="2022-01-01", end="2026-01-01")['Close']

# Clean and align the datasets
df_market = pd.DataFrame({'SPY': spy_data, 'VIX': vix_data}).dropna()
df_market.index = pd.to_datetime(df_market.index)

# ==========================================
# 2. OPTIONS ENGINE FUNCTIONS
# ==========================================
def bs_price(S, K, T, r, sigma):
    if T <= 0.001:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def get_condor_value(S, legs, T, r, sigma):
    return sum(bs_price(S, strike, T, r, sigma) * qty for strike, qty in legs)

# ==========================================
# 3. BACKTEST PARAMETERS & CONFIGURATIONS
# ==========================================
DAYS_TO_EXP = 15
RISK_FREE_RATE = 0.04

PROFIT_TARGET_PCT = 0.25   # Keep 25% of credit
STOP_LOSS_PCT = 0.50       # Exit if you lose 50% of credit
MANAGE_DAYS_DTE = 3        # Exit 3 days before expiration

# Backtest intervals (Look for entry opportunities every 15 trading days)
trade_entry_frequency = 15 
results = []

# ==========================================
# 4. HISTORICAL EXECUTION LOOP
# ==========================================
print(f"Running backtest across {len(df_market)} historical market days...")

for idx in range(0, len(df_market) - DAYS_TO_EXP, trade_entry_frequency):
    # Entry Day Data (Day 0)
    entry_date = df_market.index[idx]
    S0 = float(df_market['SPY'].iloc[idx])
    vix_0 = float(df_market['VIX'].iloc[idx]) / 100.0  # Convert VIX percent to decimal sigma

    offset = 1  # Define the offset multiplier for strike prices
    
    # Establish dynamic strikes relative to where SPY is trading right now
    # e.g., if SPY is at 400, Strikes are: Buy 405C, Sell 410C, Sell 420C, Buy 430C
    legs = [
        (S0 + 5*offset,   1),   # Long Leg
        (S0 + 10*offset, -1),   # Short Body
        (S0 + 20*offset, -1),   # Short Body
        (S0 + 30*offset,  1)    # Long Wing
    ]
    
    # Calculate initial entry premium collected using real VIX on that day
    entry_value = get_condor_value(S0, legs, DAYS_TO_EXP / 252, RISK_FREE_RATE, vix_0)
    initial_credit = -entry_value
    
    # Skip if the option model returns a garbage credit layout due to anomalies
    if initial_credit <= 0:
        continue
        
    trade_closed = False
    
    # Track the trade chronologically through actual historical business days
    for day in range(1, DAYS_TO_EXP + 1):
        current_idx = idx + day
        if current_idx >= len(df_market):
            break
            
        current_date = df_market.index[current_idx]
        current_spy = float(df_market['SPY'].iloc[current_idx])
        current_vix = float(df_market['VIX'].iloc[current_idx]) / 100.0
        
        time_left_years = (DAYS_TO_EXP - day) / 252
        current_dte = DAYS_TO_EXP - day
        
        # Re-value the structure using historical daily SPY and VIX values
        current_cost = get_condor_value(current_spy, legs, time_left_years, RISK_FREE_RATE, current_vix)
        current_pnl = initial_credit + current_cost
        
        # --- MANAGEMENT CHECKPOINTS ---
        if current_pnl >= (initial_credit * PROFIT_TARGET_PCT):
            results.append({'Entry': entry_date, 'PnL': current_pnl, 'Days': day, 'Reason': "Profit Target"})
            trade_closed = True
            break
            
        if current_pnl <= -(initial_credit * STOP_LOSS_PCT):
            results.append({'Entry': entry_date, 'PnL': current_pnl, 'Days': day, 'Reason': "Stop Loss"})
            trade_closed = True
            break
            
        if current_dte <= MANAGE_DAYS_DTE:
            results.append({'Entry': entry_date, 'PnL': current_pnl, 'Days': day, 'Reason': "Time Bailout"})
            trade_closed = True
            break
            
    if not trade_closed:
        results.append({'Entry': entry_date, 'PnL': current_pnl, 'Days': DAYS_TO_EXP, 'Reason': "Expiration"})

# ==========================================
# 5. METRICS ANALYSIS
# ==========================================
df_results = pd.DataFrame(results)

print("\n" + "="*50)
print("     ACTUAL HISTORICAL SPY BACKTEST RESULTS       ")
print("="*50)
print(f"Total Trades Executed:   {len(df_results)}")
print(f"Win Rate:               {(df_results['PnL'] > 0).mean() * 100:.1f}%")
print(f"Average Profit/Loss:    ${df_results['PnL'].mean():.2f}")
print(f"Average Days In Trade:  {df_results['Days'].mean():.1f} Days")
print("\n--- Breakdown of Real World Exits (%) ---")
print(df_results['Reason'].value_counts(normalize=True) * 100)
print("="*50)
