import requests
import ephem
import random
from selectolax.parser import HTMLParser
import yfinance as yf
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from scipy.signal import argrelextrema
from scipy.stats import norm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

def calculate_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def calculate_d2(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

def calculate_call_delta(d1):
    return norm.cdf(d1)

def calculate_put_delta(d1):
    return -norm.cdf(-d1)

def calculate_gamma(d1, S, K, T, r, sigma):
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_vega(d1, S, K, T, r):
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def calculate_theta(d1, d2, S, K, T, r, sigma, option_type):
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:  # put
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return theta

def calculate_rho(d2, K, T, r, option_type):
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:  # put
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return rho

def calculate_greeks(S, K, T, r, sigma, option_type):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)
    
    if option_type == 'call':
        delta = calculate_call_delta(d1)
    else:  # put
        delta = calculate_put_delta(d1)
    
    gamma = calculate_gamma(d1, S, K, T, r, sigma)
    vega = calculate_vega(d1, S, K, T, r)
    theta = calculate_theta(d1, d2, S, K, T, r, sigma, option_type)
    rho = calculate_rho(d2, K, T, r, option_type)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

class QuantAnalyzer:
    def __init__(self):
        pass

    def get_stock_data(self, ticker, period="1y"):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                raise ValueError(f"No data available for ticker {ticker}")
            return data
        except Exception as e:
            raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

    def calculate_returns(self, data):
        if len(data) < 2:
            raise ValueError("Insufficient data to calculate returns")
        data['Daily_Return'] = data['Close'].pct_change()
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
        return data

    def calculate_volatility(self, data, window=20):
        if len(data) < window:
            raise ValueError(f"Insufficient data to calculate {window}-day volatility")
        if 'Daily_Return' not in data.columns:
            data['Daily_Return'] = data['Close'].pct_change()
        volatility = data['Daily_Return'].rolling(window=window).std() * np.sqrt(252)
        return volatility.iloc[-1]

    def calculate_var(self, data, confidence_level=0.95):
        if 'Daily_Return' not in data.columns:
            data['Daily_Return'] = data['Close'].pct_change()
        returns = data['Daily_Return'].dropna()
        var = np.percentile(returns, 100 * (1 - confidence_level))
        return var

    def perform_monte_carlo_simulation(self, data, num_simulations=1000, time_horizon=30):
        last_price = data['Close'].iloc[-1]
        daily_returns = data['Close'].pct_change().dropna()
        
        simulation_df = pd.DataFrame()
        for x in range(num_simulations):
            prices = [last_price]
            for y in range(time_horizon):
                prices.append(prices[-1] * (1 + np.random.choice(daily_returns)))
            simulation_df[x] = prices
        
        return simulation_df

    def calculate_sharpe_ratio(self, data, risk_free_rate=0.02):
        if 'Daily_Return' not in data.columns:
            data['Daily_Return'] = data['Close'].pct_change()
        returns = data['Daily_Return'].dropna()
        sharpe_ratio = (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))
        return sharpe_ratio

    def perform_arima_forecast(self, data, order=(1,1,1), steps=30):
        if len(data) < 50:
            raise ValueError("Insufficient data to perform ARIMA forecast")
        
        data = data.asfreq('D')
        
        model = ARIMA(data['Close'], order=order)
        results = model.fit()
        forecast = results.forecast(steps=steps)
        return forecast

    def analyze(self, ticker):
        try:
            data = self.get_stock_data(ticker)
            data = self.calculate_returns(data)
            
            analysis = f"Quantitative Analysis for {ticker}:\n\n"
            
            try:
                volatility = self.calculate_volatility(data)
                analysis += f"Volatility (20-day): {volatility:.2%}\n"
                if volatility > 0.4:
                    analysis += "The stock shows high volatility, indicating significant price fluctuations.\n"
                elif volatility < 0.2:
                    analysis += "The stock shows low volatility, indicating relatively stable price movements.\n"
                else:
                    analysis += "The stock shows moderate volatility.\n"
            except Exception as e:
                analysis += f"Unable to calculate volatility: {str(e)}\n"
            
            try:
                sharpe_ratio = self.calculate_sharpe_ratio(data)
                analysis += f"\nSharpe Ratio: {sharpe_ratio:.2f}\n"
                if sharpe_ratio > 1:
                    analysis += "The stock has a good return relative to its risk.\n"
                elif sharpe_ratio < 0:
                    analysis += "The stock's return is lower than the risk-free rate.\n"
                else:
                    analysis += "The stock's return is positive but may not compensate adequately for its risk.\n"
            except Exception as e:
                analysis += f"Unable to calculate Sharpe Ratio: {str(e)}\n"
            
            try:
                var = self.calculate_var(data)
                analysis += f"\nValue at Risk (95% confidence): {var:.2%}\n"
                analysis += f"There is a 5% chance that the stock will lose more than {abs(var):.2%} in a single day.\n"
            except Exception as e:
                analysis += f"Unable to calculate Value at Risk: {str(e)}\n"
            
            try:
                monte_carlo = self.perform_monte_carlo_simulation(data)
                mc_mean = monte_carlo.iloc[-1].mean()
                mc_std = monte_carlo.iloc[-1].std()
                analysis += f"\nMonte Carlo Simulation (30 days):\n"
                analysis += f"Expected price: ${mc_mean:.2f}\n"
                analysis += f"95% Confidence Interval: ${mc_mean - 1.96*mc_std:.2f} to ${mc_mean + 1.96*mc_std:.2f}\n"
            except Exception as e:
                analysis += f"Unable to perform Monte Carlo simulation: {str(e)}\n"
            
            try:
                forecast = self.perform_arima_forecast(data)
                last_price = data['Close'].iloc[-1]
                forecast_change = (forecast[-1] - last_price) / last_price
                analysis += f"\nARIMA Forecast (30 days):\n"
                analysis += f"Forecasted price: ${forecast[-1]:.2f}\n"
                analysis += f"Forecasted change: {forecast_change:.2%}\n"
            except Exception as e:
                analysis += f"Unable to perform ARIMA forecast: {str(e)}\n"
            
            return analysis
        except Exception as e:
            return f"Error performing quantitative analysis: {str(e)}"

class FundamentalAnalyzer:
    def __init__(self):
        self.sec_url = "https://www.sec.gov/cgi-bin/browse-edgar"

    def get_10k_url(self, ticker):
        params = {
            'action': 'getcompany',
            'CIK': ticker,
            'type': '10-K',
            'dateb': '',
            'owner': 'exclude',
            'count': '1'
        }
        response = requests.get(self.sec_url, params=params)
        parser = HTMLParser(response.text)
        table = parser.css_first('table.tableFile2')
        if table:
            for row in table.css('tr')[1:]:
                cols = row.css('td')
                if cols and len(cols) > 3:
                    doc_link = cols[1].css_first('a')
                    if doc_link and '10-K' in doc_link.text():
                        return 'https://www.sec.gov' + doc_link.attributes['href']
        return None

    def scrape_10k(self, url):
        response = requests.get(url)
        parser = HTMLParser(response.text)
        text = parser.body.text()
        return text

    def get_financial_ratios(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            
            balance_sheet = stock.balance_sheet
            income_stmt = stock.financials
            cash_flow = stock.cashflow

            ratios = {}

            if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                ratios['Current Ratio'] = balance_sheet.loc['Total Current Assets'] / balance_sheet.loc['Total Current Liabilities']
            
            if 'Total Liab' in balance_sheet.index and 'Total Assets' in balance_sheet.index:
                ratios['Debt to Equity'] = balance_sheet.loc['Total Liab'] / (balance_sheet.loc['Total Assets'] - balance_sheet.loc['Total Liab'])
            
            if 'Net Income' in income_stmt.index and 'Total Stockholder Equity' in balance_sheet.index:
                ratios['Return on Equity'] = income_stmt.loc['Net Income'] / balance_sheet.loc['Total Stockholder Equity']
            
            if 'Net Income' in income_stmt.index and 'Total Assets' in balance_sheet.index:
                ratios['Return on Assets'] = income_stmt.loc['Net Income'] / balance_sheet.loc['Total Assets']
            
            if 'Net Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                ratios['Profit Margin'] = income_stmt.loc['Net Income'] / income_stmt.loc['Total Revenue']
            
            if 'marketCap' in stock.info and 'Net Income' in income_stmt.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                if net_income > 0:
                    ratios['P/E Ratio'] = stock.info['marketCap'] / net_income
                else:
                    ratios['P/E Ratio'] = np.nan  # Not applicable for negative earnings

            return pd.DataFrame(ratios)

        except Exception as e:
            print(f"Error in get_financial_ratios: {str(e)}")
            return pd.DataFrame()

    def analyze_financials(self, ticker):
        ratios = self.get_financial_ratios(ticker)
        if ratios.empty:
            return f"Unable to retrieve financial data for {ticker}. The stock might be delisted or data might be unavailable."

        analysis = f"Fundamental Analysis for {ticker}:\n\n"

        if 'Current Ratio' in ratios.columns:
            current_ratio = ratios['Current Ratio'].iloc[0]
            analysis += f"Current Ratio: {current_ratio:.2f}\n"
            if current_ratio > 2:
                analysis += "The company has a strong liquidity position.\n"
            elif current_ratio < 1:
                analysis += "The company may face liquidity issues.\n"
            else:
                analysis += "The company has an adequate liquidity position.\n"

        if 'Debt to Equity' in ratios.columns:
            debt_to_equity = ratios['Debt to Equity'].iloc[0]
            analysis += f"\nDebt to Equity Ratio: {debt_to_equity:.2f}\n"
            if debt_to_equity > 2:
                analysis += "The company has high leverage, which may increase financial risk.\n"
            elif debt_to_equity < 0.5:
                analysis += "The company has low leverage, indicating a conservative financial structure.\n"
            else:
                analysis += "The company has a balanced debt level.\n"

        if 'Return on Equity' in ratios.columns:
            roe = ratios['Return on Equity'].iloc[0]
            analysis += f"\nReturn on Equity: {roe:.2%}\n"
            if roe > 0.15:
                analysis += "The company shows strong profitability and efficient use of equity.\n"
            elif roe < 0.05:
                analysis += "The company's profitability is low compared to its equity.\n"
            else:
                analysis += "The company shows moderate profitability.\n"

        if 'P/E Ratio' in ratios.columns:
            pe_ratio = ratios['P/E Ratio'].iloc[0]
            if np.isnan(pe_ratio):
                analysis += "\nP/E Ratio: Not applicable (negative earnings)\n"
                analysis += "The company currently has negative earnings, making the P/E ratio not meaningful.\n"
            else:
                analysis += f"\nP/E Ratio: {pe_ratio:.2f}\n"
                if pe_ratio > 25:
                    analysis += "The stock may be overvalued or investors expect high growth.\n"
                elif pe_ratio < 10:
                    analysis += "The stock may be undervalued or facing growth challenges.\n"
                else:
                    analysis += "The stock is reasonably valued relative to earnings.\n"

        if not any(ratio in ratios.columns for ratio in ['Current Ratio', 'Debt to Equity', 'Return on Equity', 'P/E Ratio']):
            analysis += "Unable to calculate key financial ratios. The required financial data might be unavailable for this stock."

        return analysis


class EnhancedOptionChainAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Option Chain Analyzer")
        self.master.geometry("1200x800")

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both")

        self.option_frame = ttk.Frame(self.notebook)
        self.chart_frame = ttk.Frame(self.notebook)
        self.analysis_frame = ttk.Frame(self.notebook)
        self.fundamental_frame = ttk.Frame(self.notebook)
        self.quant_frame = ttk.Frame(self.notebook)
        self.summary_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.option_frame, text="Option Chain")
        self.notebook.add(self.chart_frame, text="Technical Chart")
        self.notebook.add(self.analysis_frame, text="Technical Analysis")
        self.notebook.add(self.fundamental_frame, text="Fundamental Analysis")
        self.notebook.add(self.quant_frame, text="Quant Analysis")
        self.notebook.add(self.summary_frame, text="Summary")

        self.recent_tickers = deque(maxlen=5)
        
        self.setup_option_frame()
        self.setup_chart_frame()
        self.setup_analysis_frame()
        self.setup_fundamental_frame()
        self.setup_quant_frame()
        self.setup_summary_frame()
        self.setup_income_estimator_frame()        

        self.fundamental_analyzer = FundamentalAnalyzer()
        self.quant_analyzer = QuantAnalyzer()

    def setup_option_frame(self):
        input_frame = ttk.Frame(self.option_frame)
        input_frame.pack(pady=10)
    
        tk.Label(input_frame, text="Enter Ticker:").grid(row=0, column=0, padx=5)
        self.ticker_entry = ttk.Entry(input_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5)
        tk.Label(input_frame, text="(Press Enter)").grid(row=0, column=2, padx=5)
    
        self.recent_tickers_var = tk.StringVar()
        self.recent_tickers_dropdown = ttk.Combobox(input_frame, textvariable=self.recent_tickers_var, state="readonly", width=15)
        self.recent_tickers_dropdown.grid(row=0, column=3, padx=5)
        self.recent_tickers_dropdown.set("Recent Tickers")
    
        tk.Label(input_frame, text="Select Expiration:").grid(row=1, column=0, padx=5, pady=5)
        self.expiry_var = tk.StringVar()
        self.expiry_dropdown = ttk.Combobox(input_frame, textvariable=self.expiry_var, state="readonly", width=15)
        self.expiry_dropdown.grid(row=1, column=1, padx=5, pady=5)
    
        self.analyze_button = ttk.Button(input_frame, text="Analyze", command=self.start_analysis)
        self.analyze_button.grid(row=1, column=2, padx=5, pady=5)
    
        columns = ("Type", "Strike", "Premium", "Last Price", "Bid", "Ask", "Volume", "Open Interest", "Implied Volatility", "Delta", "Gamma", "Theta", "Vega", "Rho")
        self.result_tree = ttk.Treeview(self.option_frame, columns=columns, show="headings")
        for col in columns:
            self.result_tree.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(self.result_tree, _col, False))
            self.result_tree.column(col, width=100)
        self.result_tree.pack(expand=True, fill="both")
    
        self.note_label = tk.Label(self.option_frame, text="Note: Click on column headers to sort.", fg="blue")
        self.note_label.pack(pady=5)
    
        self.ticker_entry.bind("<Return>", lambda event: self.fetch_expiry_dates())
        self.recent_tickers_dropdown.bind("<<ComboboxSelected>>", self.use_recent_ticker)

    def setup_chart_frame(self):
        self.chart_fig, self.chart_ax = plt.subplots(figsize=(10, 6))
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(expand=True, fill="both")

    def setup_analysis_frame(self):
        self.analysis_text = tk.Text(self.analysis_frame, wrap=tk.WORD, width=80, height=20)
        self.analysis_text.pack(padx=10, pady=10, expand=True, fill="both")

    def setup_fundamental_frame(self):
        self.fundamental_text = tk.Text(self.fundamental_frame, wrap=tk.WORD, width=80, height=20)
        self.fundamental_text.pack(padx=10, pady=10, expand=True, fill="both")
        
    def setup_summary_frame(self):
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD, width=80, height=20)
        self.summary_text.pack(padx=10, pady=10, expand=True, fill="both")

    def setup_quant_frame(self):
        self.quant_text = tk.Text(self.quant_frame, wrap=tk.WORD, width=80, height=20)
        self.quant_text.pack(padx=10, pady=10, expand=True, fill="both")

    def treeview_sort_column(self, tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        try:
            l.sort(key=lambda t: float(t[0].replace('$', '').replace('%', '')), reverse=reverse)
        except ValueError:
            l.sort(reverse=reverse)
        
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)
        
        tv.heading(col, command=lambda: self.treeview_sort_column(tv, col, not reverse))
    
    def setup_income_estimator_frame(self):
        self.income_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.income_frame, text="Income Estimator")
        
        ttk.Label(self.income_frame, text="Number of Contracts:").pack(pady=5)
        self.num_contracts_entry = ttk.Entry(self.income_frame)
        self.num_contracts_entry.pack(pady=5)
        self.num_contracts_entry.insert(0, "1") 
        
        ttk.Label(self.income_frame, text="Delta Range:").pack(pady=5)
        self.delta_slider = ttk.Scale(self.income_frame, from_=0.1, to=0.5, orient='horizontal', length=200, value=0.2)
        self.delta_slider.pack(pady=5)
        self.delta_value_label = ttk.Label(self.income_frame, text="0.20")
        self.delta_value_label.pack(pady=5)
        self.delta_slider.bind("<Motion>", self.update_delta_label)
        
        ttk.Button(self.income_frame, text="Estimate Income", command=self.calculate_income_estimate).pack(pady=10)
        
        self.income_result_text = tk.Text(self.income_frame, wrap=tk.WORD, width=80, height=20)
        self.income_result_text.pack(padx=10, pady=10, expand=True, fill="both")
        
    def update_delta_label(self, event):
        value = round(self.delta_slider.get(), 2)
        self.delta_value_label.config(text=f"{value:.2f}")    
        
    def estimate_income(self, option_chain, num_contracts, expiration_date, lower_delta=0.20, upper_delta=0.30):
        if 'Delta' not in option_chain.columns:
            current_price = option_chain['lastPrice'].iloc[0] + option_chain['strike'].iloc[0] 
            option_chain['Delta'] = option_chain.apply(lambda row: self.calculate_delta(row, current_price, expiration_date), axis=1)
        
        filtered_options = option_chain[
            ((option_chain['Type'] == 'Call') & (option_chain['Delta'] >= lower_delta) & (option_chain['Delta'] <= upper_delta)) |
            ((option_chain['Type'] == 'Put') & (option_chain['Delta'].abs() >= lower_delta) & (option_chain['Delta'].abs() <= upper_delta))
        ]
        
        if filtered_options.empty:
            return "No options found within the specified delta range."
        
        call_income = filtered_options[filtered_options['Type'] == 'Call']['lastPrice'].sum() * 100 * num_contracts
        put_income = filtered_options[filtered_options['Type'] == 'Put']['lastPrice'].sum() * 100 * num_contracts
        
        weekly_income = call_income + put_income
        monthly_income = weekly_income * 4  # assuming 4 weeks in a month
        
        result = f"Estimated Income (based on {num_contracts} contracts):\n"
        result += f"Weekly Income: ${weekly_income:.2f}\n"
        result += f"Monthly Income: ${monthly_income:.2f}\n"
        result += f"\nCall Options Income: ${call_income:.2f}\n"
        result += f"Put Options Income: ${put_income:.2f}\n"
        result += f"\nOptions used for calculation:\n"
        result += filtered_options[['Type', 'strike', 'lastPrice', 'Delta']].to_string(index=False)
        
        return result
    
    def calculate_income_estimate(self):
        ticker = self.ticker_entry.get().upper()
        expiry = self.expiry_var.get()
        delta_value = round(self.delta_slider.get(), 2)
        
        if not ticker or not expiry:
            messagebox.showerror("Error", "Please enter a ticker symbol and select an expiration date")
            return
        
        try:
            max_contracts = int(self.num_contracts_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of contracts")
            return
        
        try:
            stock = yf.Ticker(ticker)
            chain = stock.option_chain(expiry)
            
            all_options = pd.concat([chain.calls, chain.puts])
            all_options['Type'] = ['Call'] * len(chain.calls) + ['Put'] * len(chain.puts)
            
            income_estimate = self.estimate_income_scenarios(ticker, all_options, max_contracts, expiry, delta_value)
            
            self.income_result_text.delete(1.0, tk.END)
            self.income_result_text.insert(tk.END, income_estimate)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def calculate_delta(self, row, current_price, expiration_date, risk_free_rate=0.02):
        S = current_price
        K = row['strike']
        T = (pd.to_datetime(expiration_date) - pd.Timestamp.now()).days / 365
        r = risk_free_rate
        sigma = row['impliedVolatility']
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if row['Type'] == 'Call':
            return norm.cdf(d1)
        else:  
            return -norm.cdf(-d1)  
        
    def estimate_income_scenarios(self, ticker, option_chain, max_contracts, expiration_date, delta_value):
        lower_delta = max(0.1, delta_value - 0.05)
        upper_delta = min(0.5, delta_value + 0.05)
        
        if 'Delta' not in option_chain.columns:
            current_price = option_chain['lastPrice'].iloc[0] + option_chain['strike'].iloc[0]
            option_chain['Delta'] = option_chain.apply(lambda row: self.calculate_delta(row, current_price, expiration_date), axis=1)
        
        filtered_calls = option_chain[
            (option_chain['Type'] == 'Call') & 
            (option_chain['Delta'] >= lower_delta) & 
            (option_chain['Delta'] <= upper_delta)
        ]
        filtered_puts = option_chain[
            (option_chain['Type'] == 'Put') & 
            (option_chain['Delta'].abs() >= lower_delta) & 
            (option_chain['Delta'].abs() <= upper_delta)
        ]
        
        call_premium = filtered_calls['lastPrice'].sum() * 100
        put_premium = filtered_puts['lastPrice'].sum() * 100
        
        today = pd.Timestamp.now()
        expiry = pd.to_datetime(expiration_date)
        days_to_expiry = (expiry - today).days
        is_monthly = days_to_expiry > 25 
        
        frequency = "Monthly" if is_monthly else "Weekly"
        multiplier = 1 if is_monthly else 4  
        
        scenarios = {
            f"{max_contracts} Calls": call_premium * max_contracts,
            f"{max_contracts} Puts": put_premium * max_contracts,
            f"{max_contracts//2} Calls + {max_contracts//2} Puts": (call_premium * (max_contracts//2)) + (put_premium * (max_contracts//2))
        }
        
        result = f"Estimated {frequency} Income Scenarios for {ticker}:\n\n"
        for scenario, income in scenarios.items():
            result += f"{scenario}: ${income:.2f}\n"
            if not is_monthly:
                result += f"    Monthly (x4): ${income * multiplier:.2f}\n"
            result += "\n"
        
        if is_monthly:
            result += "Note: These are monthly options. The income shown is already on a monthly basis.\n\n"
        else:
            result += "Note: These are weekly options. Monthly estimate assumes 4 successful trades per month.\n\n"
        
        result += f"Delta Range: {lower_delta:.2f} - {upper_delta:.2f}\n\n"
        
        result += "Options used for calculation:\n"
        result += "Calls:\n"
        result += filtered_calls[['strike', 'lastPrice', 'Delta']].to_string(index=False) + "\n\n"
        result += "Puts:\n"
        result += filtered_puts[['strike', 'lastPrice', 'Delta']].to_string(index=False)
        
        return result

    def use_recent_ticker(self, event):
        selected_ticker = self.recent_tickers_var.get()
        if selected_ticker != "Recent Tickers":
            self.ticker_entry.delete(0, tk.END)
            self.ticker_entry.insert(0, selected_ticker)
            self.fetch_expiry_dates()

    def add_to_recent_tickers(self, ticker):
        if ticker in self.recent_tickers:
            self.recent_tickers.remove(ticker)
        self.recent_tickers.appendleft(ticker)
        self.update_recent_tickers_dropdown()

    def update_recent_tickers_dropdown(self):
        self.recent_tickers_dropdown['values'] = list(self.recent_tickers)

    def fetch_expiry_dates(self):
        ticker = self.ticker_entry.get().upper()
        if not ticker:
            messagebox.showerror("Error", "Please enter a ticker symbol")
            return

        try:
            stock = yf.Ticker(ticker)
            expiry_dates = stock.options

            if not expiry_dates:
                messagebox.showerror("Error", f"No options available for {ticker}")
                return

            self.expiry_dropdown['values'] = expiry_dates
            self.expiry_dropdown.set(expiry_dates[0])
            self.add_to_recent_tickers(ticker)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def calculate_option_greeks(self, row, current_price, risk_free_rate, days_to_expiry, option_type):
        S = current_price
        K = row['strike']
        T = days_to_expiry / 365  # Convert days to years
        r = risk_free_rate
        sigma = row['impliedVolatility']
        
        return calculate_greeks(S, K, T, r, sigma, option_type)

    def analyze_option_chain(self, ticker, expiry):
        try:
            stock = yf.Ticker(ticker)
            
            try:
                current_price = stock.info['regularMarketPrice']
            except KeyError:
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    raise ValueError("Unable to retrieve current stock price")

            chain = stock.option_chain(expiry)

            calls = chain.calls
            puts = chain.puts

            all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))

            current_price_index = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - current_price))

            call_strikes = all_strikes[current_price_index:current_price_index + 7]
            calls = calls[calls['strike'].isin(call_strikes)]

            put_strikes = all_strikes[max(0, current_price_index - 6):current_price_index + 1]
            puts = puts[puts['strike'].isin(put_strikes)]

            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            days_to_expiry = (expiry_date - datetime.now()).days

            risk_free_rate = 0.02  # 2% annual rate

            calls['greeks'] = calls.apply(lambda row: self.calculate_option_greeks(row, current_price, risk_free_rate, days_to_expiry, 'call'), axis=1)
            puts['greeks'] = puts.apply(lambda row: self.calculate_option_greeks(row, current_price, risk_free_rate, days_to_expiry, 'put'), axis=1)

            self.master.after(0, self.update_results, calls, puts, current_price)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}\nPlease try again or check your internet connection."
            self.master.after(0, messagebox.showerror, "Error", error_message)
        finally:
            self.master.after(0, lambda: self.analyze_button.config(state="normal"))

    def update_results(self, calls, puts, current_price):
        self.result_tree.delete(*self.result_tree.get_children())
        
        columns = ("Type", "Strike", "Premium", "Last Price", "Bid", "Ask", "Volume", "Open Interest", "Implied Volatility", "Delta", "Gamma", "Theta", "Vega", "Rho")
        self.result_tree.config(columns=columns)
        for col in columns:
            self.result_tree.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(self.result_tree, _col, False))
            self.result_tree.column(col, width=100)
        
        all_options = pd.concat([calls, puts])
        all_options['Type'] = ['Call'] * len(calls) + ['Put'] * len(puts)
        all_options_sorted = all_options.sort_values('strike', ascending=False)
        
        for _, row in all_options_sorted.iterrows():
            if row['Type'] == 'Call' and row['strike'] >= current_price:
                self.insert_option_row("Call", row, current_price)
        
        # Insert the current price row
        self.result_tree.insert("", "end", values=("Current Price", f"${current_price:.2f}", "", "", "", "", "", "", "", "", "", "", "", ""), tags=('header',))
        self.result_tree.tag_configure('header', background='lightgrey')
        
        for _, row in all_options_sorted.iterrows():
            if row['Type'] == 'Put' and row['strike'] <= current_price:
                self.insert_option_row("Put", row, current_price)
    
    def insert_option_row(self, option_type, row, current_price):
        strike = row['strike']
        last_price = row['lastPrice']
        premium = last_price * 100
        greeks = row['greeks']
    
        self.result_tree.insert("", "end", values=(
            option_type,
            f"${strike:.2f}",
            f"${premium:.2f}",
            f"${last_price:.2f}",
            f"${row['bid']:.2f}",
            f"${row['ask']:.2f}",
            row['volume'],
            row['openInterest'],
            f"{row['impliedVolatility']:.2%}",
            f"{greeks['delta']:.4f}",
            f"{greeks['gamma']:.4f}",
            f"{greeks['theta']:.4f}",
            f"{greeks['vega']:.4f}",
            f"{greeks['rho']:.4f}"
        ))

    def calculate_support_resistance(self, data, window=20):
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(data) - window):
            if all(data['Low'].iloc[i] <= data['Low'].iloc[i-j] for j in range(1, window+1)) and \
            all(data['Low'].iloc[i] <= data['Low'].iloc[i+j] for j in range(1, window+1)):
                support_levels.append((data.index[i], data['Low'].iloc[i]))
            
            if all(data['High'].iloc[i] >= data['High'].iloc[i-j] for j in range(1, window+1)) and \
            all(data['High'].iloc[i] >= data['High'].iloc[i+j] for j in range(1, window+1)):
                resistance_levels.append((data.index[i], data['High'].iloc[i]))
        
        return support_levels, resistance_levels

    def perform_fundamental_analysis(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            
            analysis = f"Fundamental Analysis for {ticker}:\n\n"
            
            current_price = info.get('currentPrice', None)
            if current_price:
                analysis += f"Price (per share): ${current_price:.2f}\n"
            else:
                analysis += "Price (per share): Not available\n"
            
            shares_outstanding = info.get('sharesOutstanding', None)
            if shares_outstanding:
                analysis += f"Shares Outstanding: {shares_outstanding:,}\n"
            else:
                analysis += "Shares Outstanding: Not available\n"
            
            market_cap = info.get('marketCap', None)
            if market_cap:
                analysis += f"Market Cap: ${market_cap:,.0f}\n"
            else:
                analysis += "Market Cap: Not available\n"
            
            if 'Cash' in balance_sheet.index:
                cash = balance_sheet.loc['Cash'].iloc[0]
                analysis += f"Cash: ${cash:,.0f}\n"
            else:
                analysis += "Cash: Not available\n"
            
            if 'Total Debt' in balance_sheet.index:
                debt = balance_sheet.loc['Total Debt'].iloc[0]
                analysis += f"Debt: ${debt:,.0f}\n"
            else:
                analysis += "Debt: Not available\n"
            
            ev = info.get('enterpriseValue', None)
            if ev:
                analysis += f"Enterprise Value: ${ev:,.0f}\n"
            else:
                analysis += "Enterprise Value: Not available\n"
            
            pe_ratio = info.get('trailingPE', None)
            if pe_ratio:
                analysis += f"P/E Ratio: {pe_ratio:.2f}\n"
            else:
                analysis += "P/E Ratio: Not available\n"
            
            profit_margin = info.get('profitMargins', None)
            if profit_margin:
                analysis += f"Profit Margin: {profit_margin:.2%}\n"
            else:
                analysis += "Profit Margin: Not available\n"
            
            if 'Total Revenue' in financials.index:
                revenue = financials.loc['Total Revenue'].iloc[0]
                analysis += f"Revenue (TTM): ${revenue:,.0f}\n"
            else:
                analysis += "Revenue: Not available\n"
            
            if 'Net Income' in financials.index:
                net_income = financials.loc['Net Income'].iloc[0]
                analysis += f"Net Income (TTM): ${net_income:,.0f}\n"
            else:
                analysis += "Net Income: Not available\n"
            
            if 'Net Income' in financials.index and 'Total Stockholder Equity' in balance_sheet.index:
                net_income = financials.loc['Net Income'].iloc[0]
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                roe = net_income / total_equity
                analysis += f"Return on Equity (ROE): {roe:.2%}\n"
            else:
                analysis += "Return on Equity (ROE): Not available\n"
            
            dividend_yield = info.get('dividendYield', None)
            if dividend_yield:
                analysis += f"Dividend Yield: {dividend_yield:.2%}\n"
            else:
                analysis += "Dividend Yield: Not available\n"
            
            return analysis
        except Exception as e:
            return f"Error in fundamental analysis: {str(e)}\n\nPlease check if the ticker symbol is correct and try again."

    def perform_quant_analysis(self, data, ticker, expiry):
        analysis = "Quantitative Analysis:\n"
        
        volatility = self.quant_analyzer.calculate_volatility(data)
        var = self.quant_analyzer.calculate_var(data)
        
        analysis += f"30-day Volatility: {volatility:.2%}\n"
        analysis += f"Value at Risk (95% confidence): {abs(var):.2%}\n"
        
        mc_results = self.quant_analyzer.perform_monte_carlo_simulation(data)
        mc_mean = mc_results.iloc[-1].mean()
        mc_std = mc_results.iloc[-1].std()
        analysis += f"Monte Carlo 30-day Forecast:\n"
        analysis += f"  Expected Price: ${mc_mean:.2f}\n"
        analysis += f"  95% Confidence Interval: ${mc_mean - 1.96*mc_std:.2f} to ${mc_mean + 1.96*mc_std:.2f}\n"
        
        sharpe_ratio = self.quant_analyzer.calculate_sharpe_ratio(data)
        analysis += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        
        stock = yf.Ticker(ticker)
        beta = stock.info.get('beta', None)
        if beta:
            analysis += f"Beta: {beta:.2f}\n"
        
        return analysis

    def update_quant_analysis(self, ticker):
        expiry = self.expiry_var.get()
        if not hasattr(self, 'data') or self.data.empty:
            self.prepare_data(ticker, expiry)
        analysis = self.perform_quant_analysis(self.data, ticker, expiry)
        self.master.after(0, self.update_quant_text, analysis)

    def update_quant_text(self, analysis):
        self.quant_text.delete(1.0, tk.END)
        self.quant_text.insert(tk.END, analysis)

    def prepare_data(self, ticker, expiry):
        end_date = datetime.strptime(expiry, "%Y-%m-%d")
        start_date = end_date - timedelta(days=180)  # 6 months of data
        stock = yf.Ticker(ticker)
        self.data = stock.history(start=start_date, end=end_date)
    
        self.data['Daily_Return'] = self.data['Close'].pct_change()
    
        self.data['SMA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
        self.data = self.calculate_rsi(self.data)
        self.data['Upper_BB'], self.data['Lower_BB'] = self.calculate_bollinger_bands(self.data)
    
    def start_analysis(self):
        ticker = self.ticker_entry.get().upper()
        expiry = self.expiry_var.get()
        
        if not ticker or not expiry:
            messagebox.showerror("Error", "Please enter a ticker symbol and select an expiration date")
            return
    
        self.analyze_button.config(state="disabled")
        
        self.prepare_data(ticker, expiry)
        
        threading.Thread(target=self.analyze_option_chain, args=(ticker, expiry), daemon=True).start()
        threading.Thread(target=self.update_technical_analysis, args=(ticker, expiry), daemon=True).start()
        threading.Thread(target=self.update_fundamental_analysis, args=(ticker,), daemon=True).start()
        threading.Thread(target=self.update_quant_analysis, args=(ticker,), daemon=True).start()
        threading.Thread(target=self.update_comprehensive_analysis, args=(ticker, expiry), daemon=True).start()
        threading.Thread(target=self.calculate_income_estimate, daemon=True).start()        

    def update_fundamental_analysis(self, ticker):
        analysis = self.perform_fundamental_analysis(ticker)
        self.master.after(0, self.update_fundamental_text, analysis)

    def update_fundamental_text(self, analysis):
        self.fundamental_text.delete(1.0, tk.END)
        self.fundamental_text.insert(tk.END, analysis)
        
    def update_comprehensive_analysis(self, ticker, expiry):
        summary = self.generate_comprehensive_analysis(ticker, expiry)
        self.master.after(0, self.update_summary_text, summary)

    def update_summary_text(self, summary):
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)

    def update_technical_analysis(self, ticker, expiration_date):
        try:
            end_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=180)  # 6 months of data
    
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
    
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data = self.calculate_rsi(data)
            data['Upper_BB'], data['Lower_BB'] = self.calculate_bollinger_bands(data)
    
            analysis = self.generate_analysis(data)
    
            self.master.after(0, self.update_analysis_text, analysis)
    
            self.update_chart(data, ticker, expiration_date)
    
        except Exception as e:
            error_message = f"Failed to update technical analysis: {str(e)}"
            self.master.after(0, messagebox.showerror, "Error", error_message)

    def calculate_rsi(self, data, period=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def generate_analysis(self, data):
        last_close = data['Close'].iloc[-1]
        last_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
        rsi = data['RSI'].iloc[-1]
        upper_bb = data['Upper_BB'].iloc[-1]
        lower_bb = data['Lower_BB'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]
    
        analysis = f"Technical Analysis for {self.ticker_entry.get().upper()}:\n\n"
    
        if last_close > sma20 > sma50:
            analysis += "Price Trend: Bullish. The stock is trading above both the 20-day and 50-day SMAs.\n"
        elif sma20 > sma50 and last_close > sma50:
            analysis += "Price Trend: Cautiously bullish. The stock is above the 50-day SMA but below the 20-day SMA.\n"
        elif last_close < sma20 < sma50:
            analysis += "Price Trend: Bearish. The stock is trading below both the 20-day and 50-day SMAs.\n"
        else:
            analysis += "Price Trend: Mixed. The stock is showing conflicting signals relative to its moving averages.\n"
    
        if last_volume > avg_volume * 1.5:
            analysis += "Volume: Significantly higher than average. This could indicate strong interest in the stock.\n"
        elif last_volume < avg_volume * 0.5:
            analysis += "Volume: Significantly lower than average. This could indicate lack of interest or uncertainty.\n"
        else:
            analysis += "Volume: Around average levels.\n"
    
        if rsi > 70:
            analysis += f"RSI: Overbought ({rsi:.2f} > 70). The stock might be due for a pullback.\n"
        elif rsi < 30:
            analysis += f"RSI: Oversold ({rsi:.2f} < 30). The stock might be due for a bounce.\n"
        else:
            analysis += f"RSI: Neutral ({rsi:.2f}). Neither overbought nor oversold.\n"
    
        if last_close > upper_bb:
            analysis += "Bollinger Bands: Price is above the upper band. This could indicate overbought conditions or strong upward momentum.\n"
        elif last_close < lower_bb:
            analysis += "Bollinger Bands: Price is below the lower band. This could indicate oversold conditions or strong downward momentum.\n"
        else:
            analysis += "Bollinger Bands: Price is within the bands, indicating relatively normal trading conditions.\n"
    
        support_levels, resistance_levels = self.calculate_support_resistance(data)
        if support_levels:
            latest_support = support_levels[-1][1]
            analysis += f"\nNearest Support Level: ${latest_support:.2f}\n"
        if resistance_levels:
            latest_resistance = resistance_levels[-1][1]
            analysis += f"Nearest Resistance Level: ${latest_resistance:.2f}\n"
    
        if support_levels and resistance_levels:
            if last_close < latest_support:
                analysis += "Price is currently below the nearest support level. Watch for a potential bounce or a breakdown.\n"
            elif last_close > latest_resistance:
                analysis += "Price is currently above the nearest resistance level. Watch for a potential pullback or a breakout.\n"
            else:
                analysis += "Price is between support and resistance levels. Watch for a potential breakout in either direction.\n"
    
        patterns = self.detect_patterns(data)
        if patterns:
            analysis += "\nDetected Chart Patterns:\n"
            for pattern in patterns:
                analysis += f"- {pattern}\n"
            analysis += "\nPattern Implications:\n"
            if "Double Top" in patterns:
                analysis += "- Double Top: This pattern often indicates a potential reversal of an uptrend. Consider cautious or bearish strategies.\n"
            if "Double Bottom" in patterns:
                analysis += "- Double Bottom: This pattern often signals a potential reversal of a downtrend. Consider bullish strategies.\n"
            if "Head and Shoulders" in patterns:
                analysis += "- Head and Shoulders: This pattern typically suggests a bearish reversal. Be prepared for potential downward movement.\n"
            if "Bullish Flag" in patterns:
                analysis += "- Bullish Flag: This continuation pattern suggests the current uptrend may continue. Look for potential breakouts to the upside.\n"
            if "Bearish Flag" in patterns:
                analysis += "- Bearish Flag: This continuation pattern indicates the current downtrend may persist. Watch for potential breakdowns.\n"
        else:
            analysis += "\nNo clear chart patterns detected in the recent price action.\n"
    
        analysis += "\nSuggestion: "
        if (last_close > sma20 > sma50 and rsi < 70) or (last_close < lower_bb and rsi < 30) or "Double Bottom" in patterns or "Bullish Flag" in patterns:
            analysis += "Consider bullish strategies. The stock shows positive momentum, but monitor for potential reversal signals.\n"
        elif (last_close < sma20 < sma50 and rsi > 30) or (last_close > upper_bb and rsi > 70) or "Double Top" in patterns or "Head and Shoulders" in patterns or "Bearish Flag" in patterns:
            analysis += "Consider bearish strategies. The stock shows negative momentum, but monitor for potential reversal signals.\n"
        else:
            analysis += "The stock shows mixed signals. Consider neutral strategies or wait for clearer directional indications.\n"
    
        analysis += "\nNote: This analysis is based on technical indicators only. Always consider fundamental factors and overall market conditions before making investment decisions."
    
        return analysis

    def update_analysis_text(self, analysis):
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, analysis)

    def update_chart(self, data, ticker, expiration_date):
        self.chart_ax.clear()
        
        self.chart_ax.plot(data.index, data['Close'], label='Close Price', color='blue')
        
        self.chart_ax.plot(data.index, data['SMA20'], label='20-day SMA', color='orange', alpha=0.7)
        self.chart_ax.plot(data.index, data['SMA50'], label='50-day SMA', color='red', alpha=0.7)
        
        self.chart_ax.plot(data.index, data['Upper_BB'], label='Upper BB', color='gray', linestyle='--', alpha=0.7)
        self.chart_ax.plot(data.index, data['Lower_BB'], label='Lower BB', color='gray', linestyle='--', alpha=0.7)
        
        support_levels, resistance_levels = self.calculate_support_resistance(data)
        for date, level in support_levels:
            self.chart_ax.axhline(y=level, color='green', linestyle=':', alpha=0.5)
        for date, level in resistance_levels:
            self.chart_ax.axhline(y=level, color='red', linestyle=':', alpha=0.5)
        
        ax2 = self.chart_ax.twinx()
        ax2.bar(data.index, data['Volume'], label='Volume', color='lightblue', alpha=0.3)
        ax2.set_ylabel('Volume')
        
        last_date = data.index[-1]
        last_close = data['Close'].iloc[-1]
        last_volume = data['Volume'].iloc[-1]
        
        self.chart_ax.annotate(f'Close: ${last_close:.2f}', 
                            xy=(last_date, last_close), 
                            xytext=(5, 5), textcoords='offset points', 
                            color='blue', fontweight='bold')
        
        ax2.annotate(f'Volume: {last_volume:,.0f}', 
                    xy=(last_date, last_volume), 
                    xytext=(5, 5), textcoords='offset points', 
                    color='darkblue', fontweight='bold')
        
        patterns = self.detect_patterns(data)
        if patterns:
            pattern_text = "Detected Patterns: " + ", ".join(patterns)
            self.chart_ax.text(0.05, 0.95, pattern_text, transform=self.chart_ax.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.chart_ax.set_title(f'Technical Analysis for {ticker} (Expiration: {expiration_date})')
        self.chart_ax.set_xlabel('Date')
        self.chart_ax.set_ylabel('Price')
        
        self.chart_ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        self.chart_ax.grid(True, alpha=0.3)
        
        plt.setp(self.chart_ax.get_xticklabels(), rotation=45, ha='right')
        
        self.chart_fig.tight_layout()
        self.chart_canvas.draw()

    def detect_trend(self, data, window=20):
        short_ma = data['Close'].rolling(window=window).mean()
        long_ma = data['Close'].rolling(window=window*2).mean()
        
        if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
            return "Uptrend Starting"
        elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
            return "Downtrend Starting"
        elif short_ma.iloc[-1] > long_ma.iloc[-1]:
            return "Uptrend Continuing"
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return "Downtrend Continuing"
        else:
            return "No Clear Trend"
        
    def perform_technical_analysis(self, data):
        analysis = "Technical Analysis:\n"
        
        trend = self.detect_trend(data)
        analysis += f"Current Trend: {trend}\n"
        
        sma20 = data['SMA20'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        analysis += f"20-day SMA: ${sma20:.2f}\n"
        analysis += f"50-day SMA: ${sma50:.2f}\n"
        
        if current_price > sma20 > sma50:
            analysis += "Price is above both 20-day and 50-day SMAs, indicating bullish momentum.\n"
        elif current_price < sma20 < sma50:
            analysis += "Price is below both 20-day and 50-day SMAs, indicating bearish momentum.\n"
        else:
            analysis += "Price is between SMAs, indicating potential trend reversal or consolidation.\n"
        
        rsi = data['RSI'].iloc[-1]
        analysis += f"RSI (14-day): {rsi:.2f}\n"
        if rsi > 70:
            analysis += "RSI indicates overbought conditions.\n"
        elif rsi < 30:
            analysis += "RSI indicates oversold conditions.\n"
        
        upper_bb = data['Upper_BB'].iloc[-1]
        lower_bb = data['Lower_BB'].iloc[-1]
        if current_price > upper_bb:
            analysis += "Price is above the upper Bollinger Band, suggesting potential overbought conditions.\n"
        elif current_price < lower_bb:
            analysis += "Price is below the lower Bollinger Band, suggesting potential oversold conditions.\n"
        
        return analysis

    def generate_comprehensive_analysis(self, ticker, expiry):
        if not hasattr(self, 'data') or self.data.empty:
            self.prepare_data(ticker, expiry)

        technical_analysis = self.perform_technical_analysis(self.data)
        fundamental_analysis = self.perform_fundamental_analysis(ticker)
        quant_analysis = self.perform_quant_analysis(self.data, ticker, expiry)

        current_price = self.data['Close'].iloc[-1]
        volatility = self.quant_analyzer.calculate_volatility(self.data)
        
        astrological_analysis = self.perform_astrological_analysis(ticker)
        
        summary = f"""
Comprehensive Analysis for {ticker}:

Current Price: ${current_price:.2f}

{technical_analysis}

{fundamental_analysis}

{quant_analysis}

{astrological_analysis}

Option Trading Considerations:
1. Trend-based Strategy:
   Consider {self.detect_trend(self.data).lower()} strategies aligned with the current trend.

2. Volatility-based Strategy:
   {'High' if volatility > 0.4 else 'Low'} volatility environment suggests 
   {'selling options for premium' if volatility > 0.4 else 'buying options for potential movement'}.

3. Support/Resistance Levels:
   Consider these levels for potential option strike prices.

4. Expiration Selection:
   Based on the trend strength and volatility, consider 
   {'shorter-term' if volatility > 0.4 else 'longer-term'} options.

5. Risk Management:
   Set stop-loss orders and consider using spreads to limit risk.

Long-term Investment Outlook:
1. Financial Health: {'Strong' if self.data['RSI'].iloc[-1] > 50 else 'Weak'} based on fundamental metrics.
2. Growth Potential: {'Positive' if self.detect_trend(self.data) in ['Uptrend Starting', 'Uptrend Continuing'] else 'Negative'} based on current trend and financials.
3. Risk Assessment: {'High' if volatility > 0.4 else 'Moderate to Low'} based on volatility and beta.

"""
        return summary

    def perform_astrological_analysis(self, ticker):
        analysis = "🔮 Astrological Analysis 🔮\n\n"     

        date = datetime.now()
        
        sun = ephem.Sun()
        moon = ephem.Moon()
        
        sun.compute(date)
        moon.compute(date)
        
        moon_phase = moon.phase
        if moon_phase < 5:
            analysis += "New Moon: A time for new beginnings. The stock may start a new trend.\n"
        elif moon_phase < 95:
            analysis += "Waxing Moon: Growth phase. The stock might see some gains.\n"
        elif moon_phase < 100:
            analysis += "Full Moon: Peak energy. Expect high volatility.\n"
        else:
            analysis += "Waning Moon: A time for reflection. The stock might consolidate.\n"
        
        sun_sign = ephem.constellation(sun)[1]
        analysis += f"The Sun is in {sun_sign}. "
        if sun_sign in ["Aries", "Leo", "Sagittarius"]:
            analysis += "Fire signs suggest passion and growth. The stock might heat up!\n"
        elif sun_sign in ["Taurus", "Virgo", "Capricorn"]:
            analysis += "Earth signs suggest stability. The stock might show steady performance.\n"
        elif sun_sign in ["Gemini", "Libra", "Aquarius"]:
            analysis += "Air signs suggest communication and ideas. News might affect the stock.\n"
        else:
            analysis += "Water signs suggest emotion and intuition. Trust your gut feeling about this stock.\n"
        
        events = [
            "Mercury is in retrograde. Communication might be tricky. Double-check your orders!",
            "Venus aligns with Mars. Love and war collide. The stock might see a passionate battle between bulls and bears.",
            "Jupiter's influence is strong. Expansion and growth might be on the horizon.",
            "Saturn's rings are particularly bright. Patience may be rewarded in the long term.",
            "Uranus is causing disruptions. Expect the unexpected with this stock!",
            "Neptune's dreamy influence is strong. The stock's true value might be obscured.",
            "Pluto's transformative energy is at play. A major change could be coming."
        ]
        analysis += random.choice(events) + "\n\n"
        
        change_percent = random.uniform(-5, 5)
        direction = "rise" if change_percent > 0 else "fall"
        analysis += f"The stars suggest the stock price may {direction} by about {abs(change_percent):.2f}% in the next lunar cycle.\n"
        
        advice = [
            "Consider meditation before making any trades.",
            "Consult your horoscope before major investment decisions.",
            "Aligning your chakras might improve your trading performance.",
            "Remember, the universe has a plan... but it might not be about your stock picks.",
            "If in doubt, consult a magic 8-ball for your investment strategy."
        ]
        analysis += f"\nCelestial Advice: {random.choice(advice)}\n"
        
        return analysis

    def detect_patterns(self, data, window=20):
        patterns = []
        
        local_max = argrelextrema(data['Close'].values, np.greater, order=window)[0]
        local_min = argrelextrema(data['Close'].values, np.less, order=window)[0]
        
        # double top
        if len(local_max) >= 2:
            if abs(data['Close'].iloc[local_max[-1]] - data['Close'].iloc[local_max[-2]]) / data['Close'].iloc[local_max[-2]] < 0.03:
                patterns.append("Double Top")
        
        # double bottom
        if len(local_min) >= 2:
            if abs(data['Close'].iloc[local_min[-1]] - data['Close'].iloc[local_min[-2]]) / data['Close'].iloc[local_min[-2]] < 0.03:
                patterns.append("Double Bottom")
        
        # head and shoulders
        if len(local_max) >= 3:
            if data['Close'].iloc[local_max[-2]] > data['Close'].iloc[local_max[-1]] and \
               data['Close'].iloc[local_max[-2]] > data['Close'].iloc[local_max[-3]] and \
               abs(data['Close'].iloc[local_max[-1]] - data['Close'].iloc[local_max[-3]]) / data['Close'].iloc[local_max[-3]] < 0.03:
                patterns.append("Head and Shoulders")
        
        # bullish flag
        if data['Close'].iloc[-1] > data['Close'].iloc[-window] and \
           all(data['High'].iloc[-i] <= data['High'].iloc[-i-1] for i in range(1, window)):
            patterns.append("Bullish Flag")
        
        # bearish flag
        if data['Close'].iloc[-1] < data['Close'].iloc[-window] and \
           all(data['Low'].iloc[-i] >= data['Low'].iloc[-i-1] for i in range(1, window)):
            patterns.append("Bearish Flag")
        
        return patterns

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedOptionChainAnalyzer(root)
    root.mainloop()