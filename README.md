# Code for evaluation the market-driven loss strategies
Advancing time-series forecasting with market-driven asymmetric merits for solar integration

## Usage

1. Install Python 3.9. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# catalonia case
bash ./scripts/catalonia/catalonia.sh
# california case
bash ./scripts/california/california.sh
# victoria case
bash ./scripts/victoria/victoria_Crossformer.sh
bash ./scripts/victoria/victoria_MLP.sh
bash ./scripts/victoria/victoria_LSTM.sh
bash ./scripts/victoria/victoria_GRu.sh
```