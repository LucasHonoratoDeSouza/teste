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

## Fontes reais usadas

- Chainlink BTC/USD via Polymarket RTDS (`wss://ws-live-data.polymarket.com`) para preço atual e alvo da rodada.
- Gamma API para descobrir o mercado atual e tokens `Up`/`Down`.
- CLOB REST/WS para order book, preço executável no ask e saída antecipada pelo bid.
- Binance WS fica só como microestrutura auxiliar para OBI e warmup de features.

Se o bot iniciar tarde demais dentro de uma rodada e não houver alvo Chainlink confiável no snapshot, ele espera a próxima rodada. Isso evita forward test falso com `S0` inventado pela Binance.

## Regras de risco do forward

- Não compra acima de `MAX_ENTRY_PRICE` para evitar pegar contratos de 99c com pouco upside e cauda grande.
- Simula o fill usando a profundidade real do book, não assume liquidez infinita no melhor ask.
- Limita posições abertas por rodada e por outcome para evitar piramidar a mesma ideia várias vezes.
- Vende pelo bid antes do vencimento só em caso conservador: lucro quase travado perto do fim. A saída `SOLD_OVERPRICED` fica desligada por padrão; se for reativada em `config.py`, ela só vende com PnL positivo.

## Rodar por muitas horas

O forward test faz recalibração online das probabilidades: ele guarda as previsões brutas feitas durante cada rodada, rotula essas previsões quando a rodada resolve pelo Chainlink e atualiza o calibrador em janela móvel. Os pesos da rede neural ficam congelados durante a sessão; isso evita overfit instável com poucos minutos de dados, mas corrige drift de calibração ao longo do tempo.

Durante runs longos, `status.json` é atualizado a cada poucos segundos com heartbeat, mercado atual, PnL paper, posições abertas e progresso da calibração. O `overnight.log` guarda tudo que apareceu no terminal.

Parâmetros em `config.py`:

- `ONLINE_CALIBRATION_ENABLED`: liga/desliga a recalibração live.
- `ONLINE_CALIBRATION_MIN_SAMPLES`: mínimo de amostras antes de trocar o calibrador.
- `ONLINE_CALIBRATION_MIN_ROUNDS`: mínimo de rodadas resolvidas antes de trocar o calibrador.
- `ONLINE_CALIBRATION_WINDOW_SAMPLES`: tamanho da janela móvel usada na recalibração.
