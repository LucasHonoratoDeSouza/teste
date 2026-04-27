import argparse
import asyncio
from backtest import WalkForwardBacktester
from features import MODEL_FEATURE_COLUMNS
from forward_test import ForwardTestEngine
from model import ModelEngine

def main():
    parser = argparse.ArgumentParser(description="Polymarket BTC 5-Min Quant Trading System")
    parser.add_argument('--mode', choices=['backtest', 'forward'], required=True, 
                        help='Escolha entre backtest ou forward test live.')
    parser.add_argument('--market-url', default=None,
                        help='Opcional: link/slug inicial do mercado Polymarket. Depois o robô volta ao auto-discovery.')
    parser.add_argument('--duration', type=int, default=None,
                        help='Opcional: duração do forward test em segundos para smoke test.')
    parser.add_argument('--results-file', default="results.csv",
                        help='CSV onde os trades paper serão gravados.')
    parser.add_argument('--status-file', default="status.json",
                        help='JSON de heartbeat/status atualizado durante o forward.')
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        backtester = WalkForwardBacktester()
        backtester.run()
        
    elif args.mode == 'forward':
        print("Inicializando Modelo...")
        model = ModelEngine(input_dim=len(MODEL_FEATURE_COLUMNS))
        
        forward_engine = ForwardTestEngine(
            model,
            initial_market=args.market_url,
            duration_seconds=args.duration,
            results_file=args.results_file,
            status_file=args.status_file,
        )
        
        try:
            asyncio.run(forward_engine.run())
        except KeyboardInterrupt:
            print("\nFinalizando Forward Test...")

if __name__ == "__main__":
    main()
