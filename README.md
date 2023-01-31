# -*- coding: utf-8 -*-
# Copyright (c) 2022 xxx Inc. All Rights Reserved.

We provide the code of TCF and the test demo of simulated data mentioned in the pdf.

## Dependencies

The model was implemented in Python 3.7. The following packages are needed for running the model:
- numpy==1.21.6
- pandas==1.3.5
- scipy==1.7.3
- scikit-learn==1.0.2
- tensorflow-gpu==1.14.0

## Running and evaluating the model:

TCFimt_main.py : main for start

TCFimt_encoder_evaluate.py : training for encoder, especially balanced represetation

TCFimt_decoder_evaluate.py : training for decoder

TCFimt_model.py : model based on tensorflow==1.14.0

