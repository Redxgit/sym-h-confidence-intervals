# Operational SYM-H forecasting with confidence intervals using Deep Neural Networks
## Authors
### Armando Collado-Villaverde, Pablo Muñoz and Consuelo Cid

This repository contains the predictions made by the model in the Operational SYM-H forecasting with confidence intervals using Deep Neural Networks paper on the test and test key storms and using the model in https://doi.org/10.1029/2023SW003485 as comparison.

## File Descriptions

- `generate_graphs_and_metrics.ipynb`: Jupyter notebook for reading the predictions, calculating the metrics and generating the graphs.
- `metrics.py`: functions for calculating the metrics and generating the graphs.
- `storm_dates.py`: dates for all the subsets.
- `data/`: Folder containing the predictions.
- `figs/`: Folder containing generated figures.
- `figs_SI/`: Folder containing the figures for the supplementary information.

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the author.

## DNN hyperparams

![DNN sketch](arch_q_alpha.png "DNN sketch")

Input timesteps: 48 of 5 minutes averages.
Output: SYM-H forecast at the next 1 or 2 hours and the 5% and 95% Quantile forecasts.
Same hyperparams are used for the 1 and 2 hours models.

* Top convolutional layers:
    * kernel size: 7
    * filters: 128
    * activation function: linear, swish, swish
* Multi-head attention layers:
  * num heads: 4
  * key and value dim: 128
  * dropout: 0.2
  * bias: True
* LSTM layers:
  * units: 128
* Top output dense layers:
  * units: 512
  * activation: swish
* Bottom output dense layers:
  * units: 1
  * activation: linear

Forecasted SYM-H ouput layer trained with MSE and the quantile layers with the Quantile loss in the `quantile_loss.py` file setting `q` at 0.05 and 0.95.

Trained with the AdaBelief optimizer (https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdaBelief) on TensorFlow 2.14

---
