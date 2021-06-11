import sys
import subprocess
import time



for mode in [0, 1, 2]:
	start = time. time()
	command1 = 'python -m presentation.scripts.train --max-obs 51 \
                                                     --units 128 \
                                                     --epochs 2000 \
                                                     --batch-size 256 \
                                                     --repeat 1 \
                                                     --mode {} \
                                                     --data ./astromer/data/records/ogle/ \
                                                     --p ./runs/ogle_{}'.format(mode, mode)
																																		unit_type,
	try:
	    subprocess.call(command1, shell=True)
	except Exception as e:
	    print(e)
	end = time. time()
	print('{} fold_{} takes {}'.format(mode, fold_n,(end - start)))
