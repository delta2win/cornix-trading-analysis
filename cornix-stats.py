#!/usr/bin/env python3
"""
Cornix Trading Signal Analysis Tool
==================================

This script analyzes trading signals logs from Cornix trading bot channels,
generating statistical reports and visualizations of trading performance.

Author: delta2win
Created: 2024-09-10
Last Modified: 2024-09-11
Version: 1.0.0

Related Resources:
----------------
- CryptoWelt Telegram Group:
  https://t.me/+9WWwDamOJsk2MGI0
- Private TradingView Strategy: 
  [CRYPTOWELT] GG Standard V1.0
  https://www.tradingview.com/script/2QEK8PSr-CRYPTOWELT-GG-Standard-V1-0/

Description:
-----------
This tool reads trading signal data from CSV files organized by channel,
analyzes the performance metrics, and generates:
- Visual charts showing daily and monthly profit/loss
- Win/loss distribution
- Cumulative profit tracking
- Detailed PDF reports with statistics and trade history

Usage:
------
1. Ensure your trading signal data is organized in the following structure:
   - Each channel should have its own directory called "channelname"
   - Each channel directory should contain two CSV files:
     * channelname-open.csv (for open trades)
     * channelname-closed.csv (for closed trades)

2. Run the script:
   $ python cornix-stats.py

3. Select a channel from the displayed list when prompted

4. The script will generate:
   - A chart image in the 'output' directory
   - A detailed PDF report in the 'output' directory
   - The PDF will automatically open when complete

Requirements:
------------
- Python 3.6+
- pandas
- matplotlib
- reportlab
- numpy

History:
--------
- 1.0.0: Initial release with basic functionality
  - Channel selection
  - Performance analysis
  - Chart generation
  - PDF report creation
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import subprocess
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import numpy as np
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# Create output directory if it doesn't exist
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_signal_channels():
    """Get list of available signal channels from CSV files"""
    channels = set()
    
    # Look for directories that might contain signal files
    for item in os.listdir('.'):
        if os.path.isdir(item) and not item.startswith('.') and item != OUTPUT_DIR:
            # Look for CSV files in the directory
            dir_path = os.path.join('.', item)
            for file in os.listdir(dir_path):
                if file.endswith('-open.csv') or file.endswith('-closed.csv'):
                    # Extract channel name from directory name
                    channels.add(item)
                    break
    
    return sorted(list(channels))

def get_channel_files(channel_name):
    """Get open and closed CSV files for a specific channel"""
    dir_path = os.path.join('.', channel_name)
    open_file = os.path.join(dir_path, f"{channel_name}-open.csv")
    closed_file = os.path.join(dir_path, f"{channel_name}-closed.csv")
    
    if not os.path.exists(open_file) or not os.path.exists(closed_file):
        print(f"Error: Missing CSV files for channel {channel_name}")
        return None, None
        
    return open_file, closed_file

def select_channel():
    """Let user select a channel to analyze"""
    channels = get_signal_channels()
    
    if not channels:
        print("No signal channels found. Please ensure CSV files are in the correct format:")
        print("  - channelname_open.csv")
        print("  - channelname_closed.csv")
        return None
    
    print("\nAvailable signal channels:")
    print("-" * 80)
    print(f"{'#':<4} {'Channel Name':<30} {'Total Trades':<15} {'Win Rate':<15} {'Total Profit %':<15}")
    print("-" * 80)
    
    for i, channel in enumerate(channels, 1):
        open_file, closed_file = get_channel_files(channel)
        if open_file and closed_file:
            try:
                # Get basic stats about the channel
                open_signals = pd.read_csv(open_file, encoding='utf-8-sig')
                closed_signals = pd.read_csv(closed_file, encoding='utf-8-sig')
                total_trades = len(open_signals) + len(closed_signals)
                win_trades = len(closed_signals[closed_signals['Signal Gained Profit %'] > 0])
                win_rate = (win_trades / len(closed_signals) * 100) if len(closed_signals) > 0 else 0
                total_profit = closed_signals['Signal Gained Profit %'].sum()
                print(f"{i:<4} {channel:<30} {total_trades:<15} {win_rate:.2f}%{'':<12} {total_profit:.2f}%{'':<12}")
            except Exception as e:
                print(f"Error reading {channel}: {str(e)}")
                continue
    
    while True:
        try:
            choice = int(input("\nSelect channel number to analyze: "))
            if 1 <= choice <= len(channels):
                return channels[choice-1]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def analyze_channel(channel_name):
    """Analyze a specific channel's trading signals"""
    print(f"\nAnalyzing {channel_name}...")
    
    # Get channel files
    open_signals_path, closed_signals_path = get_channel_files(channel_name)
    if not open_signals_path or not closed_signals_path:
        return
    
    # Read the CSV files
    try:
        open_signals = pd.read_csv(open_signals_path, encoding='utf-8-sig')
        closed_signals = pd.read_csv(closed_signals_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        # Try with different encoding if utf-8-sig fails
        open_signals = pd.read_csv(open_signals_path, encoding='latin1')
        closed_signals = pd.read_csv(closed_signals_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV files: {str(e)}")
        return

    # Convert Date column to datetime
    open_signals['Date'] = pd.to_datetime(open_signals['Date'])
    closed_signals['Date'] = pd.to_datetime(closed_signals['Date'])

    # Filter data starting from March 1, 2025
    start_date = pd.to_datetime('2025-03-01')
    open_signals = open_signals[open_signals['Date'] >= start_date]
    closed_signals = closed_signals[closed_signals['Date'] >= start_date]

    # Combine open and closed signals
    all_signals = pd.concat([open_signals, closed_signals], ignore_index=True)
    
    # Filter data starting from March 1, 2025
    start_date = pd.to_datetime('2025-03-01')
    all_signals = all_signals[all_signals['Date'] >= start_date]
    
    # Sort by date
    all_signals = all_signals.sort_values('Date')
    
    # Add a column to distinguish between open and closed trades
    all_signals['Status'] = 'Closed'
    all_signals.loc[all_signals.index.isin(open_signals.index), 'Status'] = 'Open'
    
    # Add month and year columns for grouping
    all_signals['Month'] = all_signals['Date'].dt.strftime('%Y-%m')
    all_signals['Day'] = all_signals['Date'].dt.strftime('%Y-%m-%d')
    
    # Calculate profit/loss based on $100 standard investment
    all_signals['Profit_Loss'] = all_signals['Signal Gained Profit %'] * 100 / 100
    
    # Calculate trade start dates based on previous profitable trades with the same symbol
    all_signals['Trade_Start_Date'] = None
    
    # Group by symbol to process each symbol's trades separately
    for symbol, symbol_trades in all_signals.groupby('Symbol'):
        # Sort trades by date
        symbol_trades = symbol_trades.sort_values('Date')
        
        # For each trade, find the start date
        for i, trade in symbol_trades.iterrows():
            # Get all previous trades with the same symbol
            prev_trades = symbol_trades[symbol_trades['Date'] < trade['Date']]
            
            # Find the most recent profitable trade
            profitable_trades = prev_trades[prev_trades['Signal Gained Profit %'] > 0]
            
            if not profitable_trades.empty:
                # Use the date of the most recent profitable trade as the start date
                start_date = profitable_trades.iloc[-1]['Date']
                all_signals.at[i, 'Trade_Start_Date'] = start_date
            else:
                # If no previous profitable trades, use the trade's own date as start date
                all_signals.at[i, 'Trade_Start_Date'] = trade['Date']
    
    # Calculate concurrent trades for each day
    all_signals['Trade_Start_Month'] = pd.to_datetime(all_signals['Trade_Start_Date']).dt.strftime('%Y-%m')
    
    # Create a list of all days in the date range
    date_range = pd.date_range(start=all_signals['Date'].min(), end=all_signals['Date'].max(), freq='D')
    
    # Initialize a dictionary to store concurrent trades for each day
    concurrent_trades = {}
    
    # For each day, count how many trades were open
    for day in date_range:
        day_str = day.strftime('%Y-%m-%d')
        # Count trades that started before or on this day and closed after this day
        open_trades = all_signals[
            (pd.to_datetime(all_signals['Trade_Start_Date']) <= day) & 
            (pd.to_datetime(all_signals['Date']) >= day)
        ]
        concurrent_trades[day_str] = len(open_trades)
    
    # Convert to DataFrame
    concurrent_df = pd.DataFrame(list(concurrent_trades.items()), columns=['Day', 'Concurrent_Trades'])
    concurrent_df['Month'] = pd.to_datetime(concurrent_df['Day']).dt.strftime('%Y-%m')
    
    # Calculate max concurrent trades per month
    max_concurrent_by_month = concurrent_df.groupby('Month')['Concurrent_Trades'].max().reset_index()
    
    # Calculate the maximum needed margin across all months
    max_concurrent_trades = max_concurrent_by_month['Concurrent_Trades'].max()
    
    # Calculate the standard trade amount (assuming $100 per trade)
    standard_trade_amount = 100
    
    # Calculate the maximum needed margin
    max_needed_margin = max_concurrent_trades * standard_trade_amount
    
    # Group by month and calculate statistics
    monthly_stats = all_signals.groupby('Month').agg({
        'Signal Gained Profit %': ['count', 'mean', 'sum'],
        'Profit_Loss': ['sum']
    }).reset_index()
    
    # Group by day for daily chart
    daily_stats = all_signals.groupby('Day').agg({
        'Signal Gained Profit %': ['count', 'mean', 'sum'],
        'Profit_Loss': ['sum']
    }).reset_index()
    
    # Flatten column names
    monthly_stats.columns = ['Month', 'Trade_Count', 'Avg_Profit_Percent', 'Total_Profit_Percent', 'Total_Profit_Dollars']
    daily_stats.columns = ['Day', 'Trade_Count', 'Avg_Profit_Percent', 'Total_Profit_Percent', 'Total_Profit_Dollars']
    
    # Calculate cumulative profit
    monthly_stats['Cumulative_Profit'] = monthly_stats['Total_Profit_Dollars'].cumsum()
    daily_stats['Cumulative_Profit'] = daily_stats['Total_Profit_Dollars'].cumsum()
    
    # Merge max concurrent trades with monthly stats
    monthly_stats = pd.merge(monthly_stats, max_concurrent_by_month, on='Month', how='left')
    
    # Calculate win rate
    win_trades = all_signals[all_signals['Signal Gained Profit %'] > 0]
    loss_trades = all_signals[all_signals['Signal Gained Profit %'] < 0]
    win_rate = len(win_trades) / len(all_signals) * 100 if len(all_signals) > 0 else 0
    
    # Calculate average profit and loss
    avg_win = win_trades['Signal Gained Profit %'].mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades['Signal Gained Profit %'].mean() if len(loss_trades) > 0 else 0
    
    # Calculate profit factor
    total_profit = win_trades['Signal Gained Profit %'].sum() if len(win_trades) > 0 else 0
    total_loss = abs(loss_trades['Signal Gained Profit %'].sum()) if len(loss_trades) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Create charts (now stacked vertically)
    plt.style.use('dark_background')
    plt.figure(figsize=(15, 25), facecolor='#1a1a1a')  # Darker gray background, increased height
    
    # Daily profit chart
    plt.subplot(6, 1, 1)
    
    # Separate open and closed trades
    closed_daily = daily_stats.copy()
    open_daily = all_signals[all_signals['Status'] == 'Open'].groupby('Day').agg({
        'Signal Gained Profit %': ['count', 'mean', 'sum'],
        'Profit_Loss': ['sum']
    }).reset_index()
    
    if not open_daily.empty:
        open_daily.columns = ['Day', 'Trade_Count', 'Avg_Profit_Percent', 'Total_Profit_Percent', 'Total_Profit_Dollars']
        
        # Plot closed trades
        closed_colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in closed_daily['Total_Profit_Dollars']]
        closed_bars = plt.bar(closed_daily['Day'], closed_daily['Total_Profit_Dollars'], color=closed_colors, label='Closed Trades')
        
        # Add trade count labels on top of closed trade bars
        for i, bar in enumerate(closed_bars):
            height = bar.get_height()
            count = closed_daily['Trade_Count'].iloc[i]
            # Position the label above the bar if positive, below if negative
            y_pos = height + 0.5 if height >= 0 else height - 0.5
            plt.text(bar.get_x() + bar.get_width()/2., y_pos, 
                    f'{count}', ha='center', va='bottom' if height >= 0 else 'top', 
                    color='white', fontsize=10)
        
        # Plot open trades
        open_colors = ['#ff9999' if x < 0 else '#99ff99' for x in open_daily['Total_Profit_Dollars']]
        open_bars = plt.bar(open_daily['Day'], open_daily['Total_Profit_Dollars'], color=open_colors, label='Open Trades')
        
        # Add trade count labels on top of open trade bars
        for i, bar in enumerate(open_bars):
            height = bar.get_height()
            count = open_daily['Trade_Count'].iloc[i]
            # Position the label above the bar if positive, below if negative
            y_pos = height + 0.5 if height >= 0 else height - 0.5
            plt.text(bar.get_x() + bar.get_width()/2., y_pos, 
                    f'{count}', ha='center', va='bottom' if height >= 0 else 'top', 
                    color='white', fontsize=10)
    else:
        # If no open trades, just plot closed trades
        closed_colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in closed_daily['Total_Profit_Dollars']]
        closed_bars = plt.bar(closed_daily['Day'], closed_daily['Total_Profit_Dollars'], color=closed_colors)
        
        # Add trade count labels on top of closed trade bars
        for i, bar in enumerate(closed_bars):
            height = bar.get_height()
            count = closed_daily['Trade_Count'].iloc[i]
            # Position the label above the bar if positive, below if negative
            y_pos = height + 0.5 if height >= 0 else height - 0.5
            plt.text(bar.get_x() + bar.get_width()/2., y_pos, 
                    f'{count}', ha='center', va='bottom' if height >= 0 else 'top', 
                    color='white', fontsize=10)
    
    plt.axhline(y=0, color='#95a5a6', linestyle='-', alpha=0.3)
    plt.title('Daily Profit/Loss ($)', fontsize=14, pad=20, color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.grid(True, alpha=0.2)
    plt.legend(facecolor='#1a1a1a', edgecolor='#333333')
    plt.tight_layout()
    
    # Daily cumulative profit chart
    plt.subplot(6, 1, 2)
    
    # Calculate cumulative profit for closed trades
    closed_daily_cumsum = closed_daily['Total_Profit_Dollars'].cumsum()
    
    # Plot closed trades cumulative
    plt.plot(closed_daily['Day'], closed_daily_cumsum, color='#3498db', linewidth=2, label='Closed Trades')
    plt.fill_between(closed_daily['Day'], closed_daily_cumsum, alpha=0.2, color='#3498db')
    
    # Calculate and plot open trades cumulative if they exist
    if not open_daily.empty:
        open_daily_cumsum = open_daily['Total_Profit_Dollars'].cumsum()
        plt.plot(open_daily['Day'], open_daily_cumsum, color='#9b59b6', linewidth=2, linestyle='--', label='Open Trades')
        plt.fill_between(open_daily['Day'], open_daily_cumsum, alpha=0.1, color='#9b59b6')
    
    plt.axhline(y=0, color='#95a5a6', linestyle='-', alpha=0.3)
    plt.title('Daily Cumulative Profit/Loss ($)', fontsize=14, pad=20, color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.grid(True, alpha=0.2)
    plt.legend(facecolor='#1a1a1a', edgecolor='#333333')
    plt.tight_layout()
    
    # Monthly profit chart
    plt.subplot(6, 1, 3)
    
    # Separate open and closed trades
    closed_monthly = monthly_stats.copy()
    open_monthly = all_signals[all_signals['Status'] == 'Open'].groupby('Month').agg({
        'Signal Gained Profit %': ['count', 'mean', 'sum'],
        'Profit_Loss': ['sum']
    }).reset_index()
    
    if not open_monthly.empty:
        open_monthly.columns = ['Month', 'Trade_Count', 'Avg_Profit_Percent', 'Total_Profit_Percent', 'Total_Profit_Dollars']
        
        # Plot closed trades
        closed_colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in closed_monthly['Total_Profit_Dollars']]
        plt.bar(closed_monthly['Month'], closed_monthly['Total_Profit_Dollars'], color=closed_colors, label='Closed Trades')
        
        # Plot open trades
        open_colors = ['#ff9999' if x < 0 else '#99ff99' for x in open_monthly['Total_Profit_Dollars']]
        plt.bar(open_monthly['Month'], open_monthly['Total_Profit_Dollars'], color=open_colors, label='Open Trades')
    else:
        # If no open trades, just plot closed trades
        closed_colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in closed_monthly['Total_Profit_Dollars']]
        plt.bar(closed_monthly['Month'], closed_monthly['Total_Profit_Dollars'], color=closed_colors)
    
    plt.axhline(y=0, color='#95a5a6', linestyle='-', alpha=0.3)
    plt.title('Monthly Profit/Loss ($)', fontsize=14, pad=20, color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.grid(True, alpha=0.2)
    plt.legend(facecolor='#1a1a1a', edgecolor='#333333')
    plt.tight_layout()
    
    # Monthly concurrent trades chart
    plt.subplot(6, 1, 4)
    
    # Plot max concurrent trades per month
    plt.bar(monthly_stats['Month'], monthly_stats['Concurrent_Trades'], color='#f39c12')
    
    # Add labels for max concurrent trades
    for i, (month, count) in enumerate(zip(monthly_stats['Month'], monthly_stats['Concurrent_Trades'])):
        plt.text(i, count + 0.1, f'Max: {count}', ha='center', color='white', fontsize=10)
    
    plt.axhline(y=0, color='#95a5a6', linestyle='-', alpha=0.3)
    plt.title('Maximum Concurrent Trades per Month', fontsize=14, pad=20, color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    # Add a new chart for margin utilization
    plt.subplot(6, 1, 5)
    
    # Calculate margin needed for each month
    margin_needed = []
    for month in monthly_stats['Month']:
        month_max_concurrent = max_concurrent_by_month[max_concurrent_by_month['Month'] == month]['Concurrent_Trades'].iloc[0] if not max_concurrent_by_month[max_concurrent_by_month['Month'] == month].empty else 0
        margin_needed.append(month_max_concurrent * standard_trade_amount)
    
    # Plot margin needed
    plt.bar(monthly_stats['Month'], margin_needed, color='#9b59b6', alpha=0.7, label='Margin Needed')
    
    # Add a line for total profit
    plt.plot(monthly_stats['Month'], monthly_stats['Total_Profit_Dollars'], color='#3498db', linewidth=2, label='Total Profit')
    
    plt.axhline(y=0, color='#95a5a6', linestyle='-', alpha=0.3)
    plt.title('Margin Needed vs Total Profit', fontsize=14, pad=20, color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.grid(True, alpha=0.2)
    plt.legend(facecolor='#1a1a1a', edgecolor='#333333')
    plt.tight_layout()
    
    # Cumulative profit chart
    plt.subplot(6, 1, 6)
    plt.plot(monthly_stats['Month'], monthly_stats['Cumulative_Profit'], color='#3498db', linewidth=2)
    plt.fill_between(monthly_stats['Month'], monthly_stats['Cumulative_Profit'], alpha=0.2, color='#3498db')
    plt.axhline(y=0, color='#95a5a6', linestyle='-', alpha=0.3)
    plt.title('Monthly Cumulative Profit/Loss ($)', fontsize=14, pad=20, color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    # Win/Loss distribution
    plt.subplot(6, 1, 6)
    win_loss_data = [len(win_trades), len(loss_trades)]
    pie_colors = ['#2ecc71', '#e74c3c']
    plt.pie(win_loss_data, labels=['Winning Trades', 'Losing Trades'], autopct='%1.1f%%', 
            colors=pie_colors, startangle=90, shadow=True)
    plt.title('Win/Loss Distribution', fontsize=14, pad=20, color='white')
    plt.tight_layout()
    
    # Save the chart with higher quality and better padding
    plt.savefig(os.path.join(OUTPUT_DIR, f"{channel_name.lower().replace(' ', '_')}_chart.png"), dpi=300, bbox_inches='tight', facecolor='#1a1a1a')

    # Create PDF report
    doc = SimpleDocTemplate(
        os.path.join(OUTPUT_DIR, f"{channel_name.lower().replace(' ', '_')}_report.pdf"),
        pagesize=letter,
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    styles = getSampleStyleSheet()
    
    # Create custom centered Heading2 style with black background
    styles.add(ParagraphStyle(
        name='CenteredHeading2',
        parent=styles['Heading2'],
        alignment=1,  # 1 = center alignment
        spaceAfter=12,
        textColor=colors.black,  # White text for dark mode
        backColor=colors.white  # Black background
    ))
    styles.add(ParagraphStyle(
        name='CenteredHeading1',
        parent=styles['Heading1'],
        alignment=1,  # 1 = center alignment
        spaceAfter=12,
        textColor=colors.black,  # White text for dark mode
        backColor=colors.white  # Black background
    ))
    
    # Create dark mode styles
    styles.add(ParagraphStyle(
        name='DarkNormal',
        parent=styles['Normal'],
        textColor=colors.white,  # White text
        fontSize=10
    ))
    
    elements = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['CenteredHeading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.black,  # White text for dark mode
        backColor=colors.white  # Black background
    )
    elements.append(Paragraph(f"{channel_name} Trading Analysis", title_style))

    # Summary statistics
    elements.append(Paragraph("Summary Statistics", styles['CenteredHeading2']))
    summary_data = [
        ["Total Trades", len(all_signals)],
        ["Winning Trades", len(win_trades)],
        ["Losing Trades", len(loss_trades)],
        ["Win Rate", f"{win_rate:.2f}%"],
        ["Average Profit (Winning Trades)", f"{avg_win:.2f}%"],
        ["Average Loss (Losing Trades)", f"{avg_loss:.2f}%"],
        ["Profit Factor", f"{profit_factor:.2f}"],
        ["Total Profit/Loss", f"${monthly_stats['Total_Profit_Dollars'].sum():.2f}"],
        ["Max Concurrent Trades", f"{max_concurrent_trades}"],
        ["Max Needed Margin", f"${max_needed_margin:.2f}"],
        ["Return on Margin", f"{(monthly_stats['Total_Profit_Dollars'].sum() / max_needed_margin * 100):.2f}%"]
    ]

    summary_table = Table(summary_data, colWidths=[200, 100])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),  # Dark background for header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # White text for header
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),  # Light background for data
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),  # Black text for data
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),  # Grey grid
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Monthly statistics
    elements.append(Paragraph("Monthly Statistics", styles['CenteredHeading2']))
    monthly_data = [['Month', 'Trades', 'Wins', 'Losses', 'Max Conc', 'Margin', 'Total %', 'P/L ($)', 'ROI %']]
    
    # Calculate wins and losses per month
    for _, row in monthly_stats.iterrows():
        # Get the max concurrent trades for this month
        month_max_concurrent = max_concurrent_by_month[max_concurrent_by_month['Month'] == row['Month']]['Concurrent_Trades'].iloc[0] if not max_concurrent_by_month[max_concurrent_by_month['Month'] == row['Month']].empty else 0
        
        # Calculate margin needed for this month
        month_margin_needed = month_max_concurrent * standard_trade_amount
        
        # Calculate return on margin for this month
        return_on_margin = (row['Total_Profit_Dollars'] / month_margin_needed * 100) if month_margin_needed > 0 else 0
        
        # Calculate total percentage based on margin
        total_percent_on_margin = (row['Total_Profit_Dollars'] / month_margin_needed * 100) if month_margin_needed > 0 else 0
        
        # Get trades for this month
        month_trades = all_signals[all_signals['Month'] == row['Month']]
        
        # Count wins and losses
        wins = len(month_trades[month_trades['Signal Gained Profit %'] > 0])
        losses = len(month_trades[month_trades['Signal Gained Profit %'] < 0])
        
        monthly_data.append([
            row['Month'],
            row['Trade_Count'],
            wins,
            losses,
            month_max_concurrent,
            f"${month_margin_needed:.0f}",
            f"{total_percent_on_margin:.1f}%",
            f"${row['Total_Profit_Dollars']:.0f}",
            f"{return_on_margin:.1f}%"
        ])

    monthly_table = Table(monthly_data, colWidths=[70, 40, 40, 40, 50, 60, 50, 60, 50])
    monthly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),  # Dark background for header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # White text for header
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),  # Light background for data
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),  # Black text for data
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),  # Grey grid
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(monthly_table)
    elements.append(Spacer(1, 20))

    # Add page break before the actual chart image
    elements.append(PageBreak())

    # Add charts title and text before page break
    elements.append(Paragraph("Trading Charts", styles['CenteredHeading2']))
    elements.append(Paragraph("Daily and Monthly Performance Charts", styles['DarkNormal']))
    elements.append(Spacer(1, 20))

    # Add the chart image to the PDF
    img = Image(os.path.join(OUTPUT_DIR, f"{channel_name.lower().replace(' ', '_')}_chart.png"), width=552, height=620)  # Full width (612 - 60 margins) and full height (792 - 72 margins)
    elements.append(img)
    elements.append(Spacer(1, 20))

    # Add page break before the actual chart image
    elements.append(PageBreak())

    # Daily statistics
    elements.append(Paragraph("Daily Statistics", styles['CenteredHeading2']))
    daily_data = [['Date', 'Trades', 'Avg Profit %', 'Total Profit %', 'Profit/Loss ($)', 'Cumulative Profit ($)']]
    for _, row in daily_stats.iterrows():
        daily_data.append([
            row['Day'],
            row['Trade_Count'],
            f"{row['Avg_Profit_Percent']:.2f}%",
            f"{row['Total_Profit_Percent']:.2f}%",
            f"${row['Total_Profit_Dollars']:.2f}",
            f"${row['Cumulative_Profit']:.2f}"
        ])

    daily_table = Table(daily_data, colWidths=[80, 50, 80, 80, 100, 100])
    daily_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),  # Dark background for header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # White text for header
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),  # Light background for data
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),  # Black text for data
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),  # Grey grid
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(daily_table)
    elements.append(Spacer(1, 20))

    # Add page break before the actual chart image
    elements.append(PageBreak())

    # Symbol summary statistics
    elements.append(Paragraph("Symbol Performance Summary", styles['CenteredHeading2']))
    
    # Group by symbol and calculate statistics
    symbol_stats = all_signals.groupby('Symbol').agg({
        'Signal Gained Profit %': ['count', 'mean', 'sum'],
        'Profit_Loss': ['sum']
    }).reset_index()
    
    # Flatten column names
    symbol_stats.columns = ['Symbol', 'Trade_Count', 'Avg_Profit_Percent', 'Total_Profit_Percent', 'Total_Profit_Dollars']
    
    # Calculate win rate for each symbol
    symbol_win_rates = []
    for symbol in symbol_stats['Symbol']:
        symbol_trades = all_signals[all_signals['Symbol'] == symbol]
        win_trades = symbol_trades[symbol_trades['Signal Gained Profit %'] > 0]
        win_rate = len(win_trades) / len(symbol_trades) * 100 if len(symbol_trades) > 0 else 0
        symbol_win_rates.append(win_rate)
    
    # Add win rate column
    symbol_stats['Win_Rate'] = symbol_win_rates
    
    # Sort by win rate (descending) instead of total profit
    symbol_stats = symbol_stats.sort_values('Win_Rate', ascending=False)
    
    # Create symbol summary table
    symbol_data = [['Symbol', 'Trades', 'Win Rate', 'Avg Profit %', 'Total Profit %', 'Profit/Loss ($)']]
    for _, row in symbol_stats.iterrows():
        symbol_data.append([
            row['Symbol'],
            row['Trade_Count'],
            f"{row['Win_Rate']:.2f}%",
            f"{row['Avg_Profit_Percent']:.2f}%",
            f"{row['Total_Profit_Percent']:.2f}%",
            f"${row['Total_Profit_Dollars']:.2f}"
        ])

    symbol_table = Table(symbol_data, colWidths=[80, 50, 80, 80, 80, 100])
    symbol_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),  # White background for header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Black text for header
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),  # Light background for data
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),  # Black text for data
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),  # Grey grid
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(symbol_table)
    elements.append(Spacer(1, 20))

    # Add page break before the actual chart image
    elements.append(PageBreak())

    # All trades grouped by month
    elements.append(Paragraph("All Trades by Month", styles['CenteredHeading2']))
    trades_data = [['Month', 'Date', 'Symbol', 'Direction', 'Entry', 'Last Target', 'Profit %', 'Profit $']]
    
    # Sort signals by date
    all_signals_sorted = all_signals.sort_values('Date')
    
    for _, row in all_signals_sorted.iterrows():
        entry = row.get('Entry Targets', 0)
        last_target = row.get('Last Target', 0)
        trades_data.append([
            row['Month'],
            row['Date'].strftime('%Y-%m-%d %H:%M'),
            row['Symbol'],
            row['Direction'],
            str(entry),
            str(last_target),
            f"{row['Signal Gained Profit %']:.2f}%",
            f"${row['Profit_Loss']:.2f}"
        ])

    trades_table = Table(trades_data, colWidths=[60, 100, 60, 40, 60, 60, 60, 60])
    trades_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),  # Dark background for header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # White text for header
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),  # Light background for data
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),  # Black text for data
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),  # Grey grid
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(trades_table)

    # Build the PDF with dark background
    doc.build(elements)

    print(f"\nAnalysis complete. PDF report saved to {doc.filename}")
    print(f"Total profit/loss: ${monthly_stats['Total_Profit_Dollars'].sum():.2f}")
    print(f"Cumulative profit/loss: ${monthly_stats['Cumulative_Profit'].iloc[-1]:.2f}")
    print(f"Win rate: {win_rate:.2f}%")
    
    # Open the PDF file
    try:
        if os.name == 'nt':  # Windows
            os.startfile(doc.filename)
        else:  # macOS and Linux
            subprocess.run(['xdg-open', doc.filename])
        print(f"Opening PDF report: {doc.filename}")
    except Exception as e:
        print(f"Could not open PDF automatically: {e}")
        print(f"Please open the PDF manually: {doc.filename}")

def main():
    # Let user select channel
    selected_channel = select_channel()
    if not selected_channel:
        return
        
    # Analyze the selected channel
    analyze_channel(selected_channel)

if __name__ == "__main__":
    main()
