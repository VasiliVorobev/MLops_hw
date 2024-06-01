#!/bin/bash
echo "MLops Home work 1"
echo "Data creation..."
python3 data_creation.py
echo "Model prepation..."
python3 model_preparation.py
echo "Model processing..."
python3 model_preprocessing.py
echo "Model testing..."
python3 model_testing.py
echo "Done)"
bash