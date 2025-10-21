# Trading Strategy with High Sharpe Ratio
# Predicts 3-day price movements with ensemble models
# Only trades when confidence is high
# Includes proper risk management and position sizing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import glob

# Hyperparameters - OPTIMIZED FOR ACCURACY
LEARNING_RATE = 0.0005 
EPOCHS = 300 
BATCH_SIZE = 128 
EARLY_STOPPING_PATIENCE = 40 
WEIGHT_DECAY = 0.00005 
DROPOUT = 0.3 
N_MODELS = 7 
PREDICTION_DAYS = 2 
MIN_CONFIDENCE = 0.53 

INITIAL_CAPITAL = 100000
MAX_POSITION_SIZE = 0.65 
MAX_TOTAL_EXPOSURE = 2.5 
STOP_LOSS_PCT = 0.018   
TAKE_PROFIT_PCT = 0.09 
TRAILING_STOP = True
TRAILING_STOP_PCT = 0.015 
USE_KELLY = True
COMPOUND_RETURNS = True
KELLY_MULTIPLIER = 1.7 
MIN_POSITION_SIZE = 0.08 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PricePredictor(nn.Module):
    def __init__(self, input_size, dropout=0.3, hidden_size=256):
        super(PricePredictor, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, 
                            batch_first=True, dropout=dropout)
        
        self.attention = nn.Linear(hidden_size, 1)
        
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  
        
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :] 
        
        x = self.fc1(lstm_out)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

def load_data(data_folder='optidata'):
    csv_files = glob.glob(os.path.join(data_folder, '*_opti.csv'))
    if not csv_files:
        csv_files = glob.glob(os.path.join(data_folder, '*Optimized.csv'))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_folder}")
    
    print(f"Found {len(csv_files)} file(s)")
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['symbol'] = os.path.basename(csv_file).replace('_opti.csv', '').replace('Optimized.csv', '')
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    return data, csv_files

def prepare_data(data, prediction_days=3):
    feature_columns = ['dayoftheyear', 'open', 'high', 'low', 'close', 'WLLDE']
    
    # Technical indicators
    data['pct_change'] = data['close'].pct_change() * 100
    data['high_low_range'] = ((data['high'] - data['low']) / data['close']) * 100
    data['open_close_diff'] = ((data['close'] - data['open']) / data['open']) * 100
    data['price_momentum'] = data['close'].diff()
    
    # Moving averages
    data['close_ma3'] = data['close'].rolling(window=3, min_periods=1).mean()
    data['close_ma7'] = data['close'].rolling(window=7, min_periods=1).mean()
    data['close_ma14'] = data['close'].rolling(window=14, min_periods=1).mean()
    data['close_ma21'] = data['close'].rolling(window=21, min_periods=1).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(window=20, min_periods=1).mean()
    bb_std = data['close'].rolling(window=20, min_periods=1).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_width'] = ((data['bb_upper'] - data['bb_lower']) / data['bb_middle']) * 100
    data['bb_position'] = ((data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-10)) * 100
    
    # Volatility
    data['volatility'] = data['close'].rolling(window=5, min_periods=1).std()
    data['volatility_ratio'] = data['volatility'] / data['close']
    data['dist_from_ma7'] = ((data['close'] - data['close_ma7']) / data['close_ma7']) * 100
    data['dist_from_ma14'] = ((data['close'] - data['close_ma14']) / data['close_ma14']) * 100
    
    # Fill NaN
    data.fillna(method='bfill', inplace=True)
    data.fillna(0, inplace=True)
    
    extended_features = feature_columns + ['pct_change', 'high_low_range', 'open_close_diff', 
                                           'price_momentum', 'close_ma3', 'close_ma7', 'close_ma14', 'close_ma21',
                                           'rsi', 'macd', 'macd_signal', 'macd_hist',
                                           'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                                           'volatility', 'volatility_ratio', 'dist_from_ma7', 'dist_from_ma14']
    
    X = data[extended_features].values[:-prediction_days]
    current_close = data['close'].values[:-prediction_days]
    future_close = data['close'].values[prediction_days:]
    
    y = (future_close > current_close).astype(np.float32)
    
    actual_prices = data[['close']].values[:-prediction_days]
    future_prices = data[['close']].values[prediction_days:]
    
    return X, y, current_close, future_prices.flatten()

def train_single_model(X_train, y_train, X_val, y_val, model_id, pos_weight, symbol):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PricePredictor(input_size=X_train.shape[1], dropout=DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=15, verbose=False)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining Model {model_id + 1}/{N_MODELS}...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
            preds = torch.sigmoid(outputs.squeeze()) > 0.5
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total * 100
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                preds = torch.sigmoid(outputs.squeeze()) > 0.5
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total * 100
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} - Best Val Acc: {best_val_acc:.2f}%")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"  Final Model Validation Accuracy: {best_val_acc:.2f}%")
    
    model_path = f'models/{symbol}/model_{model_id + 1}.pth'
    os.makedirs(f'models/{symbol}', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_train.shape[1],
        'best_val_acc': best_val_acc
    }, model_path)
    
    return model

def ensemble_predict(models, X_test):
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    all_predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_predictions.append(probs)
    
    ensemble_probs = np.mean(all_predictions, axis=0)
    return ensemble_probs

def kelly_criterion(win_prob, win_loss_ratio, max_fraction=0.60, multiplier=1.5):
    """Calculate optimal position size using Kelly Criterion with multiplier"""
    lose_prob = 1 - win_prob
    kelly_fraction = (win_prob * win_loss_ratio - lose_prob) / win_loss_ratio
    kelly_fraction *= multiplier
    return max(0, min(kelly_fraction, max_fraction))

def backtest_strategy(predictions, confidence, current_prices, future_prices, prediction_days):
    """
    Aggressive multi-position strategy with leverage and compounding
    """
    capital = INITIAL_CAPITAL
    positions = [] 
    trades = []
    equity_curve = [capital]
    daily_returns = []
    peak_equity = capital
    
    max_len = min(len(predictions), len(current_prices), len(future_prices))
    
    for i in range(max_len):
        if i == 0:
            continue
            
        current_price = current_prices[i]
        pred_prob = predictions[i]
        
        new_positions = []
        for pos in positions:
            pnl_pct = (current_price - pos['entry']) / pos['entry'] * pos['direction']
            
            close_position = False
            exit_type = None
            
            if TRAILING_STOP and pnl_pct > TRAILING_STOP_PCT:
                if pnl_pct < pos['max_profit'] - TRAILING_STOP_PCT:
                    close_position = True
                    exit_type = 'trailing_stop'
                else:
                    pos['max_profit'] = max(pos['max_profit'], pnl_pct)
            
            if pnl_pct <= -STOP_LOSS_PCT:
                close_position = True
                exit_type = 'stop_loss'
            elif pnl_pct >= TAKE_PROFIT_PCT:
                close_position = True
                exit_type = 'take_profit'
            elif i >= pos['entry_index'] + prediction_days:
                close_position = True
                exit_type = 'time_exit'
            
            if close_position:
                pnl = pos['size'] * pnl_pct
                if COMPOUND_RETURNS:
                    capital += pnl  # Compound the gains
                else:
                    capital += pnl
                
                trades.append({
                    'entry': pos['entry'],
                    'exit': current_price,
                    'profit': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'type': exit_type,
                    'direction': 'long' if pos['direction'] == 1 else 'short',
                    'size': pos['size']
                })
            else:
                new_positions.append(pos)
        
        positions = new_positions
        
        total_exposure = sum(p['size'] for p in positions) / capital if capital > 0 else 0
        
        if capital > 0 and total_exposure < MAX_TOTAL_EXPOSURE:
            should_trade = False
            direction = 0
            
            if pred_prob >= MIN_CONFIDENCE:
                direction = 1
                should_trade = True
            elif pred_prob <= (1 - MIN_CONFIDENCE):
                direction = -1
                should_trade = True
            
            if should_trade:
                available_capital = capital * MAX_TOTAL_EXPOSURE - sum(p['size'] for p in positions)
                
                if USE_KELLY:
                    win_loss_ratio = TAKE_PROFIT_PCT / STOP_LOSS_PCT
                    effective_prob = pred_prob if direction == 1 else (1 - pred_prob)
                    kelly_size = kelly_criterion(effective_prob, win_loss_ratio, MAX_POSITION_SIZE, KELLY_MULTIPLIER)
                    position_size = min(capital * kelly_size, available_capital, capital * MAX_POSITION_SIZE)
                else:
                    position_size = min(capital * MAX_POSITION_SIZE, available_capital)
                
                if position_size > capital * 0.05: 
                    positions.append({
                        'entry': current_price,
                        'entry_index': i,
                        'direction': direction,
                        'size': position_size,
                        'max_profit': 0
                    })
        
        open_pnl = sum(p['size'] * ((current_price - p['entry']) / p['entry'] * p['direction']) for p in positions)
        current_equity = capital + open_pnl
        equity_curve.append(current_equity)
        
        if len(equity_curve) > 1:
            daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
            daily_returns.append(daily_return)
        
        peak_equity = max(peak_equity, current_equity)
    
    return trades, equity_curve, daily_returns

def calculate_metrics(trades, equity_curve, daily_returns=None):
    """Calculate performance metrics including Sharpe Ratio"""
    equity_curve = np.array(equity_curve)
    
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'return_pct': 0,
            'final_capital': equity_curve[-1] if len(equity_curve) > 0 else INITIAL_CAPITAL
        }
    
    total_profit = sum(t['profit'] for t in trades)
    winning_trades = [t for t in trades if t['profit'] > 0]
    win_rate = len(winning_trades) / len(trades) * 100
    
    equity_curve = np.array(equity_curve)
    if daily_returns is None:
        returns = np.diff(equity_curve) / equity_curve[:-1]
    else:
        returns = np.array(daily_returns)
    
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    years = len(equity_curve) / 252
    if years > 0:
        cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1
    else:
        cagr = 0
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = np.min(drawdown) * 100
    
    return_pct = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_profit': total_profit,
        'sharpe_ratio': sharpe_ratio,
        'cagr': cagr * 100,
        'max_drawdown': max_drawdown,
        'return_pct': return_pct,
        'final_capital': equity_curve[-1]
    }

if __name__ == "__main__":
    print("="*60)
    print("SYMBOL-SPECIFIC TRADING STRATEGY")
    print("="*60)
    
    print(f"\nStrategy Parameters:")
    print(f"  Prediction Window: {PREDICTION_DAYS} days")
    print(f"  Minimum Edge Required: {MIN_CONFIDENCE*100:.1f}%")
    print(f"  Max Position Size: {MAX_POSITION_SIZE*100:.1f}% of capital")
    print(f"  Position Sizing: Kelly Criterion")
    print(f"  Stop Loss: {STOP_LOSS_PCT*100:.1f}%")
    print(f"  Take Profit: {TAKE_PROFIT_PCT*100:.1f}%")
    print(f"  Ensemble Models per Symbol: {N_MODELS}")
    print(f"  Trading Symbols: SPY, QQQ")
    print(f"  Benchmark: GSPC (S&P 500 Index)")
    
    TRADE_SYMBOLS = ['SPY', 'QQQ']
    BENCHMARK_SYMBOL = 'GSPC'
    
    all_results = {}
    
    for symbol in TRADE_SYMBOLS:
        print(f"\n{'='*60}")
        print(f"TRAINING AND TESTING: {symbol}")
        print(f"{'='*60}")
        
        csv_file = f'optidata/{symbol}_opti.csv'
        if not os.path.exists(csv_file):
            print(f"ERROR: {csv_file} not found! Skipping {symbol}")
            continue
        
        data = pd.read_csv(csv_file)
        X, y, current_prices, future_prices = prepare_data(data, PREDICTION_DAYS)
    
        print(f"Total samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
        current_temp, current_test = current_prices[:-int(len(X)*0.20)], current_prices[-int(len(X)*0.20):]
        future_temp, future_test = future_prices[:-int(len(X)*0.20)], future_prices[-int(len(X)*0.20):]
        
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, shuffle=False)
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Class balance - UP: {np.sum(y_train == 1)/len(y_train)*100:.1f}%, DOWN: {np.sum(y_train == 0)/len(y_train)*100:.1f}%")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    
        # Calculate class weights
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        pos_weight = torch.tensor([neg_count / pos_count]).to(device)
        
        # Train ensemble of models for this symbol
        print(f"\nTraining {N_MODELS} Models for {symbol}...")
        print("-" * 60)
        
        models = []
        for i in range(N_MODELS):
            model = train_single_model(X_train_scaled, y_train, X_val_scaled, y_val, i, pos_weight, symbol)
            models.append(model)
    
        # Make predictions on test set
        print(f"\nMaking Ensemble Predictions for {symbol}...")
        print("-" * 60)
        
        test_predictions = ensemble_predict(models, X_test_scaled)
        test_confidence = np.abs(test_predictions - 0.5) * 2
        
        # Calculate prediction accuracy
        pred_binary = (test_predictions > 0.5).astype(int)
        accuracy = np.mean(pred_binary == y_test) * 100
        
        bullish_mask = test_predictions >= MIN_CONFIDENCE
        bearish_mask = test_predictions <= (1 - MIN_CONFIDENCE)
        high_conf_mask = bullish_mask | bearish_mask
        
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(pred_binary[high_conf_mask] == y_test[high_conf_mask]) * 100
        else:
            high_conf_accuracy = 0
        
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        print(f"  Tradeable Signals: {np.sum(high_conf_mask)} ({np.sum(high_conf_mask)/len(test_predictions)*100:.1f}%)")
        print(f"  Tradeable Signal Accuracy: {high_conf_accuracy:.2f}%")
        
        # Backtest strategy
        print(f"\nBacktesting {symbol}...")
        
        trades, equity_curve, daily_returns = backtest_strategy(test_predictions, test_confidence, 
                                                                current_test, future_test, PREDICTION_DAYS)
        
        metrics = calculate_metrics(trades, equity_curve, daily_returns)
        
        print(f"\n{'-'*60}")
        print(f"{symbol} PERFORMANCE")
        print(f"{'-'*60}")
        print(f"Final Capital:      ${metrics['final_capital']:,.2f}")
        print(f"Total Return:       {metrics['return_pct']:.2f}%")
        print(f"CAGR:               {metrics['cagr']:.2f}%")
        print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown:       {metrics['max_drawdown']:.2f}%")
        print(f"Total Trades:       {metrics['total_trades']}")
        print(f"Win Rate:           {metrics['win_rate']:.2f}%")
        
        all_results[symbol] = metrics
        
        import pickle
        model_dir = f'models/{symbol}'
        os.makedirs(model_dir, exist_ok=True)
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"\nModels saved to '{model_dir}/' directory")
    
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {BENCHMARK_SYMBOL}")
    print(f"{'='*60}")
    
    benchmark_file = f'optidata/{BENCHMARK_SYMBOL}_opti.csv'
    if os.path.exists(benchmark_file):
        benchmark_data = pd.read_csv(benchmark_file)
        X_bench, y_bench, prices_bench, future_bench = prepare_data(benchmark_data, PREDICTION_DAYS)
        
        test_size = int(len(X_bench) * 0.20)
        X_bench_test = X_bench[-test_size:]
        y_bench_test = y_bench[-test_size:]
        prices_bench_test = prices_bench[-test_size:]
        future_bench_test = future_bench[-test_size:]
        
        buy_hold_return = ((prices_bench_test[-1] - prices_bench_test[0]) / prices_bench_test[0]) * 100
        years = len(prices_bench_test) / 252
        buy_hold_cagr = ((prices_bench_test[-1] / prices_bench_test[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Buy & Hold CAGR:   {buy_hold_cagr:.2f}%")
        
        all_results[BENCHMARK_SYMBOL] = {
            'return_pct': buy_hold_return,
            'cagr': buy_hold_cagr,
            'sharpe_ratio': 0,
            'strategy': 'Buy & Hold'
        }
    
    print(f"\n{'='*60}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Symbol':<10} {'Strategy':<15} {'Return':<12} {'CAGR':<12} {'Sharpe':<10} {'Win%':<10}")
    print("-" * 70)
    
    for symbol in TRADE_SYMBOLS:
        if symbol in all_results:
            r = all_results[symbol]
            print(f"{symbol:<10} {'ML Trading':<15} {r['return_pct']:>10.2f}% {r['cagr']:>10.2f}% {r['sharpe_ratio']:>9.4f} {r['win_rate']:>8.2f}%")
    
    if BENCHMARK_SYMBOL in all_results:
        r = all_results[BENCHMARK_SYMBOL]
        print(f"{BENCHMARK_SYMBOL:<10} {r['strategy']:<15} {r['return_pct']:>10.2f}% {r['cagr']:>10.2f}% {'N/A':<10} {'N/A':<10}")
    
    results_list = []
    for symbol, metrics in all_results.items():
        result = {'symbol': symbol}
        result.update(metrics)
        results_list.append(result)
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('symbol_performance.csv', index=False)
    
    print(f"\n{'='*60}")
    print("Training completed! Results saved to 'symbol_performance.csv'")
    print(f"Models saved in 'models/SPY/' and 'models/QQQ/'")
    print(f"{'='*60}")
