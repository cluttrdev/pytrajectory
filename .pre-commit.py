#!/usr/bin/python

'''
Git pre commit hook: write latest commit's datetime

This script writes the date and time of the pending commit into
the source file `doc/index.rst` for the documentation and 
the file `__init__.py` of pytrajectory such that it is obvious
to which current code version they belong.

If changed, this file has to be copied as 'pre-commit' to 
'/path/to/repo/pytrajectory/.git/hooks/'
and should be made executable for the changes to take effect.
'''

import os
import sys
import time

# get current date and time
datetime = time.strftime('%Y-%m-%d %H:%M:%S')
datetime_str = "This documentation is built automatically from the source code (commit: {})\n".format(datetime)

# specify the paths to the files where to replace the placeholder of `datetime` in
file_paths = [['doc', 'source', 'index.rst'],
              ['pytrajectory', '__init__.py']]

# alter the files
for path in file_paths:
    try:
        with open(os.sep.join(path), mode='r+') as f:
            # read the file's lines
            in_lines = f.readlines()
            
            if f.name.endswith('.rst'):
                # get the line in which the datetime string will be written
                idx = in_lines.index('.. Placeholder for the datetime string of latest commit\n') + 2
                
                # replace the placeholder
                out_lines = in_lines[:idx] + [datetime_str] + in_lines[idx+1:]
            elif f.name.endswith('.py'):
                # get the line in which the datetime string will be written
                idx = in_lines.index('# Placeholder for the datetime string of latest commit\n') + 1
                
                # replace the placeholder
                out_lines = in_lines[:idx] + ['__date__ = "{}"\n'.format(datetime)] + in_lines[idx+1:]
            
            # rewind the file
            f.seek(0)
            
            # write the output
            f.writelines(out_lines)
    except Exception as err:
        print "Could not change file: {}".format(path[-1])
        print err.message
        print "Commit will be aborted!"
        sys.exit(1)

# add the files to the commit
for path in file_paths:
    f_path = os.sep.join(path)
    
    try:
        os.system('git add {}'.format(f_path))
    except Exception as err:
        print err.message
        print "Commit will be aborted!"
        sys.exit(1)
    

