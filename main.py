import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch

from nnScript import (
    PricePredictor,
    prepare_data,
    train_single_model,
    ensemble_predict,
    backtest_strategy,
    calculate_metrics,
    MIN_CONFIDENCE,
    PREDICTION_DAYS,
    N_MODELS,
    device,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def infer_symbol_from_csv(csv_path: str) -> str:
    basename = os.path.basename(csv_path)
    symbol = basename.replace('_opti.csv', '').replace('Optimized.csv', '')
    return symbol


def list_optidata_csvs(optidata_dir: str = 'optidata'):
    paths = glob.glob(os.path.join(optidata_dir, '*_opti.csv'))
    if not paths:
        paths = glob.glob(os.path.join(optidata_dir, '*Optimized.csv'))
    return sorted(paths)


def load_trained_models(symbol: str, limit: int | None = None):
    model_dir = os.path.join('models', symbol)
    paths = sorted(glob.glob(os.path.join(model_dir, 'model_*.pth')))
    if limit is not None:
        paths = paths[:limit]

    models = []
    input_sizes = []
    for p in paths:
        chk = torch.load(p, map_location=device)
        input_size = chk.get('input_size')
        m = PricePredictor(input_size=input_size).to(device)
        m.load_state_dict(chk['model_state_dict'])
        m.eval()
        models.append(m)
        input_sizes.append(input_size)
    return models, input_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to optidata CSV')
    parser.add_argument('--symbol', type=str, help='Symbol name (expects optidata/<symbol>_opti.csv)')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--n-models', type=int, default=N_MODELS)
    parser.add_argument('--model-index', type=int, help='Use a single trained model index (1-based). If omitted, uses ensemble.')
    args = parser.parse_args()

    if not args.csv and not args.symbol:
        candidates = list_optidata_csvs()
        if not candidates:
            raise SystemExit('No CSVs found in optidata/. Provide --csv or --symbol.')
        print('Found CSVs:')
        for i, p in enumerate(candidates, 1):
            print(f'  [{i}] {p}')
        choice = input('Select CSV number: ').strip()
        try:
            idx = int(choice) - 1
            csv_path = candidates[idx]
        except Exception:
            raise SystemExit('Invalid selection')
    else:
        if args.csv:
            csv_path = args.csv
        else:
            csv_guess = os.path.join('optidata', f'{args.symbol}_opti.csv')
            alt_guess = os.path.join('optidata', f'{args.symbol}Optimized.csv')
            csv_path = csv_guess if os.path.exists(csv_guess) else alt_guess

    if not os.path.exists(csv_path):
        raise SystemExit(f'CSV not found: {csv_path}')

    symbol = infer_symbol_from_csv(csv_path)
    print(f'Using symbol: {symbol}')

    data = pd.read_csv(csv_path)
    X, y, current_prices, future_prices = prepare_data(data, PREDICTION_DAYS)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    current_temp, current_test = current_prices[:-int(len(X)*0.20)], current_prices[-int(len(X)*0.20):]
    future_temp, future_test = future_prices[:-int(len(X)*0.20)], future_prices[-int(len(X)*0.20):]

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = torch.tensor([neg_count / pos_count]).to(device) if pos_count > 0 else torch.tensor([1.0]).to(device)

    model_dir = os.path.join('models', symbol)
    os.makedirs(model_dir, exist_ok=True)

    models = []
    if args.mode == 'train':
        k = max(1, int(args.n_models))
        for i in range(k):
            m = train_single_model(X_train_scaled, y_train, X_val_scaled, y_val, i, pos_weight, symbol)
            models.append(m)
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    else:
        try:
            with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
        except Exception:
            pass
        models, _ = load_trained_models(symbol)
        if not models:
            raise SystemExit(f'No trained models found for {symbol} in {model_dir}. Run with --mode train first.')

    if args.model_index:
        idx = args.model_index - 1
        if idx < 0 or idx >= len(models):
            raise SystemExit(f'--model-index out of range. Have {len(models)} models.')
        use_models = [models[idx]]
    else:
        use_models = models

    test_predictions = ensemble_predict(use_models, X_test_scaled)
    test_confidence = np.abs(test_predictions - 0.5) * 2

    pred_binary = (test_predictions > 0.5).astype(int)
    accuracy = np.mean(pred_binary == y_test) * 100

    bullish_mask = test_predictions >= MIN_CONFIDENCE
    bearish_mask = test_predictions <= (1 - MIN_CONFIDENCE)
    high_conf_mask = bullish_mask | bearish_mask

    if np.sum(high_conf_mask) > 0:
        high_conf_accuracy = np.mean(pred_binary[high_conf_mask] == y_test[high_conf_mask]) * 100
    else:
        high_conf_accuracy = 0

    print('\nEvaluation Metrics:')
    print(f'  Models used: {len(use_models)} (of {len(models)})')
    print(f'  Overall Accuracy: {accuracy:.2f}%')
    print(f'  Tradeable Signals: {np.sum(high_conf_mask)} ({np.sum(high_conf_mask)/len(test_predictions)*100:.1f}%)')
    print(f'  Tradeable Signal Accuracy: {high_conf_accuracy:.2f}%')

    trades, equity_curve, daily_returns = backtest_strategy(
        test_predictions, test_confidence, current_test, future_test, PREDICTION_DAYS
    )

    metrics = calculate_metrics(trades, equity_curve, daily_returns)

    print('\n------------------------------')
    print(f'{symbol} PERFORMANCE')
    print('------------------------------')
    print(f"Final Capital:      ${metrics['final_capital']:,.2f}")
    print(f"Total Return:       {metrics['return_pct']:.2f}%")
    print(f"CAGR:               {metrics.get('cagr', 0):.2f}%")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown:       {metrics['max_drawdown']:.2f}%")
    print(f"Total Trades:       {metrics['total_trades']}")
    print(f"Win Rate:           {metrics['win_rate']:.2f}%")


if __name__ == '__main__':
    main()
