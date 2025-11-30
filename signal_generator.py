"""
Signal Generator for 5-15 minute trading timeframes
Uses Multivariate LSTM predictions to generate buy/sell/hold signals
"""

import numpy as np
import pandas as pd
from datetime import datetime
from data_fetcher import CCXTDataFetcher
from multivariate_lstm import MultivariateLSTMTrainer
from config import (
    SYMBOL, TIMEFRAME, SEQUENCE_LENGTH,
    BUY_THRESHOLD, SELL_THRESHOLD, MIN_CONFIDENCE
)
import os


class SignalGenerator:
    """Generates trading signals from LSTM predictions"""
    
    def __init__(self, model_path=None):
        """
        Initialize signal generator
        
        Args:
            model_path: Path to trained model (auto-detects if None)
        """
        self.fetcher = CCXTDataFetcher()
        self.lstm = MultivariateLSTMTrainer()
        
        # Load model
        if model_path is None:
            model_path = self._find_model_path()
        
        if model_path and os.path.exists(model_path):
            self.lstm.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found. Train model first using train_model.py")
    
    def _find_model_path(self):
        """Find model path automatically"""
        model_dir = "models"
        if os.path.exists(model_dir):
            files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if files:
                return os.path.join(model_dir, files[0])
        return None
    
    def generate_signal(self, symbol=SYMBOL, timeframe=TIMEFRAME):
        """
        Generate trading signal for current market conditions
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            
        Returns:
            Dictionary with signal information
        """
        # Fetch latest data
        data = self.fetcher.fetch_latest(symbol, timeframe, limit=SEQUENCE_LENGTH + 50)
        
        if data is None or len(data) < SEQUENCE_LENGTH:
            return {
                'signal': 'ERROR',
                'confidence': 0.0,
                'predicted_return': 0.0,
                'message': 'Insufficient data'
            }
        
        # Prepare features
        features = self.lstm.prepare_multivariate_features(data)
        
        if len(features) < SEQUENCE_LENGTH:
            return {
                'signal': 'ERROR',
                'confidence': 0.0,
                'predicted_return': 0.0,
                'message': 'Not enough features after processing'
            }
        
        # Get prediction
        try:
            predicted_return = self.lstm.predict_next(features)
        except Exception as e:
            return {
                'signal': 'ERROR',
                'confidence': 0.0,
                'predicted_return': 0.0,
                'message': f'Prediction error: {str(e)}'
            }
        
        # Generate signal
        signal, confidence = self._interpret_prediction(predicted_return)
        
        # Get current price
        current_price = data['close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_return)
        
        return {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': signal,
            'confidence': abs(predicted_return),
            'predicted_return': predicted_return,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change_pct': predicted_return * 100,
            'message': self._get_signal_message(signal, predicted_return)
        }
    
    def _interpret_prediction(self, predicted_return):
        """
        Interpret prediction and generate signal
        
        Args:
            predicted_return: Predicted return (e.g., 0.02 = 2% increase)
            
        Returns:
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: Absolute value of predicted return
        """
        confidence = abs(predicted_return)
        
        # Check minimum confidence threshold
        if confidence < MIN_CONFIDENCE:
            return 'HOLD', confidence
        
        # Generate signal based on thresholds
        if predicted_return >= BUY_THRESHOLD:
            return 'BUY', confidence
        elif predicted_return <= SELL_THRESHOLD:
            return 'SELL', confidence
        else:
            return 'HOLD', confidence
    
    def _get_signal_message(self, signal, predicted_return):
        """Get human-readable signal message"""
        pct = predicted_return * 100
        
        if signal == 'BUY':
            return f"BUY signal: Expected {pct:.2f}% price increase"
        elif signal == 'SELL':
            return f"SELL signal: Expected {pct:.2f}% price decrease"
        else:
            return f"HOLD: Predicted {pct:.2f}% change (below threshold)"
    
    def generate_signals_continuous(self, symbol=SYMBOL, timeframe=TIMEFRAME, interval=300):
        """
        Continuously generate signals at specified interval
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            interval: Update interval in seconds (default: 300 = 5 minutes)
        """
        import time
        
        print(f"Starting continuous signal generation...")
        print(f"Symbol: {symbol}, Timeframe: {timeframe}")
        print(f"Update interval: {interval} seconds ({interval/60:.1f} minutes)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                signal_data = self.generate_signal(symbol, timeframe)
                self._print_signal(signal_data)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nStopped signal generation")
    
    def _print_signal(self, signal_data):
        """Print signal in formatted way"""
        print("=" * 60)
        print(f"Time: {signal_data['timestamp']}")
        print(f"Symbol: {signal_data['symbol']} | Timeframe: {signal_data['timeframe']}")
        print(f"Current Price: ${signal_data['current_price']:,.2f}")
        print(f"Predicted Return: {signal_data['price_change_pct']:+.2f}%")
        print(f"Predicted Price: ${signal_data['predicted_price']:,.2f}")
        print(f"\n🔔 SIGNAL: {signal_data['signal']}")
        print(f"Confidence: {signal_data['confidence']*100:.2f}%")
        print(f"Message: {signal_data['message']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test signal generator
    try:
        generator = SignalGenerator()
        signal = generator.generate_signal()
        
        if signal['signal'] != 'ERROR':
            generator._print_signal(signal)
        else:
            print(f"Error: {signal['message']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train the model first:")
        print("  python train_model.py")

