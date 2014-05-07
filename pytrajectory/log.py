#!/bin/python2

import time
import sys


class Logger():
    '''
    This class the output of the log data.
    
    It can simultaneously write to a specified file and the standard output as well as just the
    file. In addition a loglevel can be set to suppress some information.
    
    
    Parameters
    ----------
    
    fname : str
        The name of the log file to which all information will be written
    
    mode : str
        Either 'w' if an existing file should be overwritten or 'a' if it should be appended
    
    suppress : bool
        Whether or not to suppress output to the screen
    
    verbosity : int
        The level of verbosity that restricts the output
    '''
    
    def __init__(self, fname=None, mode="w", suppress=False, verbosity=0):
        if not fname:
            fname = sys.argv[0].split('.')[0]+"_"+time.strftime('%y%m%d-%H%M%S') + ".log"
        self.logfile = open(fname, mode)
        self.stdout = sys.stdout
        self.suppressed = suppress
        self.verbosity = verbosity
        sys.stdout = self


    def write(self, text, verblvl=0):
        '''
        Writes log information if :attr:`verblvl` is less or equal to the level of verbosity.
        
        
        Parameters
        ----------
        
        text : str
            The information to log.
        
        verblvl : int
            The 'inportance' of the information.
        '''
        if verblvl <= self.verbosity:
            self.logfile.write(text)
            if not self.suppressed:
                self.stdout.write(text)

    def __del__(self):
        sys.stdout = self.stdout
        self.logfile.close()


class Timer():
    '''
    Provides a context manager that takes the time of a code block.
    
    Parameters
    ----------
    
    label : str
        The 'name' of the code block which is timed
    
    verbose : bool
        Whether or not to output the elapsed time at exit
    '''
    def __init__(self, label="~", verbose=True):
        self.label = label
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.delta = time.time() - self.start
        if self.verbose:
            logtime("---> [%s elapsed %f s]"%(self.label, self.delta))


def set_file(fname=None, suppress=True):
    '''
    Sets a file to which all log information are written.
    '''
    if not fname:
        fname = sys.argv[0].split('.')[0]+"_"+time.strftime('%y%m%d-%H%M%S')+".log"
    sys.stdout = Logger(fname, "w", suppress)


def msg(label, text, lvl=0):
    if isinstance(sys.stdout, Logger):
        #sys.stdout.write(time.strftime('%d-%m-%Y_%H:%M:%S')+"\t"+label+"\t"+text+"\n")
        sys.stdout.write(label+"\t"+text+"\n", lvl)
    else:
        #sys.stdout.write(time.strftime('%d-%m-%Y_%H:%M:%S')+"\t"+label+"\t"+text+"\n")
        sys.stdout.write(label+"\t"+text+"\n")

def info(text, lvl=0):
    msg("INFO:", text, lvl)

def logtime(text, lvl=0):
    msg("TIME:", text, lvl)

def warn(text, lvl=0):
    msg("WARN:", text, lvl)

def error(text, lvl=0):
    msg("ERROR:", text, lvl)


#def IPS(loc=None):
#    shelltime = time.time()
#    try:
#        fname = sys.stdout.logfile.name
#        suppress = sys.stdout.suppressed
#
#        del(sys.stdout)
#        embed(user_ns=loc)
#        sys.stdout = Logger(fname, "a", suppress)
#    except:
#        embed(user_ns=loc)
#    info("Embedded IPython shell")
#    logtime("---> [%s elapsed %f s]"%("IPS",time.time()-shelltime))


if __name__ == "__main__":
    import log

    log.set_file(suppress=False)

    with log.Timer("testing"):
        log.info("Information")
        log.warn("Warning")
        log.err("Error")

    log.info("succeeded")

