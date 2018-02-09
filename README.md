# Hidden-Markov-Model
## Requirements: 
* scipy>=0.15.1
* numpy>=1.8.0

## File Descriptions:

* main.py: primary file to run. Contains implementation to parse input and call inference and learning modules.
* hmms.py: Implementation of forward backward functions, inference and learning methods.
* data.txt: Observable outcome for time series data. total 50 instance, each of length 30.
* model-input-file.txt: Pre-estimated parameters for inference.

## Instructions to run the implementation: 

* Perform Inference: Infer hidden status given outcomes of time series data.
    ```
    python main.py --mode test --data data.txt --predictions-file predictions-file.txt \
    --model-input-file model-input-file.txt
    ```
* Learning-Estimate model Parameters:  
    ```
    python main.py --mode train --data data.txt --model-input-file model-input-file.txt \
    --model-output-file model-output-file.txt --iterations 100
    ```

* Predictions: Predict the hidden states for new data using the parameter learned.
    ```
    python main.py --mode test --data data.txt --predictions-file predictions-file-after-learning.txt \
    --model-input-file model-output-file.txt\
    ```

* For more details run help:
    ````
    $ python main.py --help
    usage: main.py [-h] --mode {train,test} [--data DATA]
    [--predictions-file PREDICTIONS_FILE]
    [--model-input-file MODEL_INPUT_FILE]
    [--model-output-file MODEL_OUTPUT_FILE]
    [--iterations ITERATIONS]
    Perform HMM inference or learning.
    ````

