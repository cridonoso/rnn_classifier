#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

ds_name = sys.argv[1]
for rnn_type in ['phased', 'lstm']:
    for fold_n in range(3):
        start = time.time()
        command1 = 'python -m presentation.scripts.train \
                    --data ./data/records/{}/fold_{} \
                    --rnn-type {} \
                    --p ./runs/{}/fold_{}/{} ' \
                    .format(ds_name, fold_n,
                            rnn_type,
                            ds_name, fold_n, rnn_type
                            )
        try:
            subprocess.call(command1, shell=True)
        except Exception as e:
            print(e)

        end = time. time()
        print('{} on fold {} takes {:.2f} sec'.format(rnn_type, fold_n, (end - start)))
