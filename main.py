"""
Main execution script for Crypto Trading Signal System
Supports: LSTM, GRU, Transformer, XGBoost
"""

import argparse
from train_all_models import train_model, train_all_models
from signal_generator_all import SignalGenerator
from config import SYMBOL, TIMEFRAME


def main():
    parser = argparse.ArgumentParser(
        description='Crypto Trading Signal System - Multiple Models (LSTM, GRU, Transformer, XGBoost)'
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'signal', 'continuous'],
        help='Mode: train (train model), signal (single signal), continuous (continuous signals)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['lstm', 'gru', 'transformer', 'xgboost', 'all'],
        default='lstm',
        help='Model type: lstm, gru, transformer, xgboost, or all (default: lstm)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default=SYMBOL,
        help=f'Trading pair (default: {SYMBOL})'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default=TIMEFRAME,
        help=f'Timeframe (default: {TIMEFRAME})'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model file (auto-detects if not provided)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if args.model == 'all':
            print("Training all models...")
            train_all_models()
        else:
            print(f"Training {args.model.upper()} model...")
            train_model(args.model)
        
    elif args.mode == 'signal':
        print(f"Generating trading signal using {args.model.upper()}...")
        try:
            generator = SignalGenerator(model_type=args.model, model_path=args.model_path)
            signal = generator.generate_signal(symbol=args.symbol, timeframe=args.timeframe)
            
            if signal['signal'] != 'ERROR':
                generator._print_signal(signal)
            else:
                print(f"Error: {signal['message']}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"\nPlease train the model first:")
            print(f"  python main.py train --model {args.model}")
    
    elif args.mode == 'continuous':
        print(f"Starting continuous signal generation using {args.model.upper()}...")
        try:
            generator = SignalGenerator(model_type=args.model, model_path=args.model_path)
            generator.generate_signals_continuous(
                symbol=args.symbol,
                timeframe=args.timeframe,
                interval=300  # 5 minutes
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"\nPlease train the model first:")
            print(f"  python main.py train --model {args.model}")
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()

