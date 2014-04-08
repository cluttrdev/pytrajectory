import os
import subprocess
import re

import sys
sys.path.append('..')
import pytrajectory.log as log

from IPython import embed as IPS

def run_examples():
    examples = [f for f in os.listdir('.') if os.path.isfile(f) and re.match('ex\d+.*\.py', f)]
    
    for ex in sorted(examples):
        log.info('Run '+ex)
        out = open('outputs' + os.sep + ex[:-3] + '.txt', "w")
        try:
            with log.Timer():
                subprocess.call('python2 ' + ex, shell=True, stdout=out)
            log.info('--> succeeded')
        except:
            log.error('--> failed')