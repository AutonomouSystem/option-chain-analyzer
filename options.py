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

class OptionChainAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Option Chain Analyzer")
        self.master.geometry("1200x800")

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both")

        self.option_frame = ttk.Frame(self.notebook)
        self.chart_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.option_frame, text="Option Chain")
        self.notebook.add(self.chart_frame, text="Bollinger Bands")

        self.recent_tickers = deque(maxlen=5)
        
        self.setup_option_frame()
        self.setup_chart_frame()

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

        columns = ("Type", "Strike", "Premium", "Last Price", "Bid", "Ask", "Volume", "Open Interest", "Implied Volatility")
        self.result_tree = ttk.Treeview(self.option_frame, columns=columns, show="headings")
        for col in columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=100)
        self.result_tree.pack(expand=True, fill="both")

        self.note_label = tk.Label(self.option_frame, text="Note: Greek values are not available through this data source.", fg="red")
        self.note_label.pack(pady=5)

        self.ticker_entry.bind("<Return>", lambda event: self.fetch_expiry_dates())
        self.recent_tickers_dropdown.bind("<<ComboboxSelected>>", self.use_recent_ticker)

    def setup_chart_frame(self):
        self.chart_fig, self.chart_ax = plt.subplots(figsize=(10, 6))
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(expand=True, fill="both")

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

    def start_analysis(self):
        ticker = self.ticker_entry.get().upper()
        expiry = self.expiry_var.get()
        
        if not ticker or not expiry:
            messagebox.showerror("Error", "Please enter a ticker symbol and select an expiration date")
            return

        self.analyze_button.config(state="disabled")
        threading.Thread(target=self.analyze_option_chain, args=(ticker, expiry), daemon=True).start()
        threading.Thread(target=self.update_bollinger_bands, args=(ticker, expiry), daemon=True).start()

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

            self.master.after(0, self.update_results, calls, puts, current_price)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}\nPlease try again or check your internet connection."
            self.master.after(0, messagebox.showerror, "Error", error_message)
            self.master.after(0, lambda: self.analyze_button.config(state="normal"))

    def update_results(self, calls, puts, current_price):
        self.result_tree.delete(*self.result_tree.get_children())

        self.result_tree.insert("", "end", values=("Current Price", f"${current_price:.2f}", "", "", "", "", "", "", ""), tags=('header',))
        self.result_tree.tag_configure('header', background='lightgrey')

        for option_type, df in [("Call", calls), ("Put", puts)]:
            for _, row in df.iterrows():
                last_price = row['lastPrice']
                premium = last_price * 100  
                self.result_tree.insert("", "end", values=(
                    option_type,
                    f"${row['strike']:.2f}",
                    f"${premium:.2f}",  
                    f"${last_price:.2f}",  
                    f"${row['bid']:.2f}",
                    f"${row['ask']:.2f}",
                    row['volume'],
                    row['openInterest'],
                    f"{row['impliedVolatility']:.2%}"
                ))

        self.analyze_button.config(state="normal")

    def get_bollinger_bands(self, data, window=20, num_std=2):
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return rolling_mean, upper_band, lower_band

    def update_bollinger_bands(self, ticker, expiration_date):
        try:
            end_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            days_to_expiration = (end_date - datetime.now()).days
        
            if days_to_expiration <= 7:
                start_date = end_date - timedelta(days=180)  # 6 months for daily data
                data_freq = 'Daily'
                window = 20
            else:
                start_date = end_date - timedelta(days=365)  # 1 year for weekly data
                data_freq = 'Weekly'
                window = 10  

            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)

            if data_freq == 'Weekly':
                data = data.resample('W').last()

            middle_bb, upper_bb, lower_bb = self.get_bollinger_bands(data, window, num_std=2)

            self.chart_ax.clear()
            self.chart_ax.plot(data.index, data['Close'], label='Close Price', color='blue')
            self.chart_ax.plot(upper_bb.index, upper_bb, label='Upper BB', color='red', alpha=0.7)
            self.chart_ax.plot(lower_bb.index, lower_bb, label='Lower BB', color='green', alpha=0.7)
            self.chart_ax.plot(middle_bb.index, middle_bb, label='Middle BB', color='orange', alpha=0.7)

        
            last_date = data.index[-1]
            last_close = data['Close'].iloc[-1]
        
            self.chart_ax.annotate(f'Close: ${last_close:.2f}', 
                                xy=(last_date, last_close), 
                                xytext=(5, 5), textcoords='offset points', 
                                color='blue', fontweight='bold')
            self.chart_ax.annotate(f'Upper: ${upper_bb.iloc[-1]:.2f}', 
                                xy=(last_date, upper_bb.iloc[-1]), 
                                xytext=(5, 5), textcoords='offset points', 
                                color='red', fontweight='bold')
            self.chart_ax.annotate(f'Middle: ${middle_bb.iloc[-1]:.2f}', 
                                xy=(last_date, middle_bb.iloc[-1]), 
                                xytext=(5, 5), textcoords='offset points', 
                                color='orange', fontweight='bold')
            self.chart_ax.annotate(f'Lower: ${lower_bb.iloc[-1]:.2f}', 
                                xy=(last_date, lower_bb.iloc[-1]), 
                                xytext=(5, 5), textcoords='offset points', 
                                color='green', fontweight='bold')

            self.chart_ax.set_title(f'{data_freq} Bollinger Bands for {ticker} (Expiration: {expiration_date})')
            self.chart_ax.set_xlabel('Date')
            self.chart_ax.set_ylabel('Price')
            self.chart_ax.legend()
            self.chart_ax.grid(True, alpha=0.3)

            plt.setp(self.chart_ax.get_xticklabels(), rotation=45, ha='right')

            self.chart_fig.tight_layout()
            self.chart_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update Bollinger Bands: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OptionChainAnalyzer(root)
    root.mainloop()