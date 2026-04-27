import argparse
import time
from datetime import datetime

import pandas as pd
import requests


def download_1s_klines(symbol='BTCUSDT', hours=24, output_path="historical_1s.csv"):
    print(f"Baixando {hours} horas de dados históricos de 1 segundo para {symbol}...")
    limit = 1000
    end_time = int(time.time() * 1000)
    start_time = end_time - (hours * 3600 * 1000)
    
    all_klines = []
    current_start = start_time
    
    while current_start < end_time:
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1s&startTime={current_start}&limit={limit}"
            res = requests.get(url, timeout=10)
            data = res.json()
            if not data or type(data) is dict: # Error dict
                break
            all_klines.extend(data)
            current_start = data[-1][0] + 1000
            print(f"Baixado até: {datetime.utcfromtimestamp(current_start/1000)}", end='\r')
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}")
            break
            
    print("\nProcessando dados...")
    df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['price'] = df['close'].astype(float)
    # Volumes como proxies
    df['bid_vol'] = df['taker_buy_base'].astype(float)
    df['ask_vol'] = df['volume'].astype(float) - df['bid_vol']
    
    df = df[['timestamp', 'price', 'bid_vol', 'ask_vol']]
    df.to_csv(output_path, index=False)
    print(f"Download concluído! Salvo em {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Baixa klines Binance 1s para treino do bot.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output-path", default="historical_1s.csv")
    args = parser.parse_args()
    download_1s_klines(symbol=args.symbol, hours=args.hours, output_path=args.output_path)


if __name__ == "__main__":
    main()
