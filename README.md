# BTC 5m Polymarket Forward Test

Forward test paper para mercados `btc-updown-5m-*` do Polymarket.
Os resultados novos são gravados em `results.csv`.

## Rodar

```bash
python -u main.py --mode forward
```

Para deixar rodando durante a noite até você parar com `Ctrl+C`:

```bash
python -u main.py --mode forward --results-file results.csv --status-file status.json 2>&1 | tee overnight.log
```

Opcionalmente, force um mercado inicial:

```bash
python main.py --mode forward --market-url "https://polymarket.com/event/btc-updown-5m-1777248300"
```

Smoke test curto, só quando quiser validar rapidamente:

```bash
python -u main.py --mode forward --duration 160
```

## Treinar de forma mais robusta

Baixe mais histórico Binance antes de retreinar:

```bash
python download_data.py --hours 336 --output-path historical_14d_1s.csv
```

Treine o modelo tabular com amostragem esparsa por rodada:

```bash
python train_model.py --csv-path historical_14d_1s.csv --sample-every-seconds 10
```

Backtest quantitativo walk-forward:

```bash
python main.py --mode backtest
```

## Fontes reais usadas

- Chainlink BTC/USD via Polymarket RTDS (`wss://ws-live-data.polymarket.com`) para preço atual e alvo da rodada.
- Gamma API para descobrir o mercado atual e tokens `Up`/`Down`.
- CLOB REST/WS para order book, preço executável no ask e saída antecipada pelo bid.
- Binance WS fica só como microestrutura auxiliar para OBI e warmup de features.

Se o bot iniciar tarde demais dentro de uma rodada e não houver alvo Chainlink confiável no snapshot, ele espera a próxima rodada. Isso evita forward test falso com `S0` inventado pela Binance.

## Regras de risco do forward

- Usa blending entre probabilidade do modelo e `mid` do mercado para reduzir sobreconfiança.
- Só entra na janela intermediária da rodada, evitando início muito cedo e entradas coladas no vencimento.
- Simula o fill usando a profundidade real do book e desconta fee `taker` do CLOB no EV e no PnL.
- Limita a uma única operação por rodada para evitar overtrading no mesmo mercado.
- Vende antes do vencimento só para travar lucro perto do fim ou quando o bid líquido está claramente acima do valor de hold do contrato.

## Rodar por muitas horas

O forward test faz recalibração online das probabilidades: ele guarda previsões espaçadas no tempo dentro de cada rodada, rotula essas previsões quando a rodada resolve pelo Chainlink e atualiza o calibrador em janela móvel. O modelo base fica congelado durante a sessão; isso evita drift instável no meio da noite e deixa a adaptação focada em calibração.

Durante runs longos, `status.json` é atualizado a cada poucos segundos com heartbeat, mercado atual, PnL paper, posições abertas e progresso da calibração. O `overnight.log` guarda tudo que apareceu no terminal.

Parâmetros em `config.py`:

- `ONLINE_CALIBRATION_ENABLED`: liga/desliga a recalibração live.
- `ONLINE_CALIBRATION_MIN_SAMPLES`: mínimo de amostras antes de trocar o calibrador.
- `ONLINE_CALIBRATION_MIN_ROUNDS`: mínimo de rodadas resolvidas antes de trocar o calibrador.
- `ONLINE_CALIBRATION_WINDOW_SAMPLES`: tamanho da janela móvel usada na recalibração.
- `MODEL_MARKET_BLEND`: peso do sinal do modelo sobre o prior do mercado.
- `TAKER_FEE_RATE`: fee usada no paper trade para aproximar o custo real do CLOB.
