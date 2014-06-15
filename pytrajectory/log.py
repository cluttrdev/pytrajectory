import time
import sys


class Logger():
    '''
    This class handles the output of the log data.
    
    It can simultaneously write to a specified file and the standard output as well as just one
    of both. In addition a loglevel can be set to suppress unnecessary information.
    
    
    Parameters
    ----------
    
    log2file : bool
        Whether or not to write information to a logfile
    
    fname : str
        The name of the log file to which all information will be written
    
    mode : str
        Either 'w' if an existing file should be overwritten or 'a' if it should be appended
    
    suppress : bool
        Whether or not to suppress output to the screen
    
    verbosity : int
        The level of verbosity that restricts the output
    '''
    
    def __init__(self, log2file=False, fname=None, mode="w", suppress=False, verbosity=1):
        self.log2file = log2file
        if self.log2file:
            self.logfile = open(fname, mode)
        else:
            self.logfile = None
        self.stdout = sys.stdout
        self.suppressed = suppress
        self.verbosity = verbosity
        sys.stdout = self


    def write(self, text, verb=3):
        '''
        Writes log information if :attr:`verb` is less or equal to the level of verbosity.
        
        
        Parameters
        ----------
        
        text : str
            The information to log.
        
        verb : int
            The 'inportance' of the information.
        '''
        if self.log2file:
            self.logfile.write(text)
        if verb <= self.verbosity and not self.suppressed:
            self.stdout.write(text)

    def __del__(self):
        sys.stdout = self.stdout
        if self.logfile:
            self.logfile.close()


class Timer():
    '''
    Provides a context manager that takes the time of a code block.
    
    Parameters
    ----------
    
    label : str
        The 'name' of the code block which is timed
    
    verb : int
        Level of verbosity
    '''
    def __init__(self, label="~", verb=4):
        self.label = label
        self.verb = verb

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.delta = time.time() - self.start
        logtime("---> [%s elapsed %f s]"%(self.label, self.delta), verb=self.verb)


def log_on(verbosity=1, log2file=False, fname=None, suppress=False):
    '''
    Sets a file to which all log information are written.
    '''
    if log2file and not fname:
        fname = sys.argv[0].split('.')[0]+"_"+time.strftime('%y%m%d-%H%M%S')+".log"
    sys.stdout = Logger(log2file, fname, "w", suppress, verbosity)


def msg(label, text, verb=3):
    if isinstance(sys.stdout, Logger):
        #sys.stdout.write(time.strftime('%d-%m-%Y_%H:%M:%S')+"\t"+label+"\t"+text+"\n")
        sys.stdout.write(label+"\t"+text+"\n", verb)
    else:
        #sys.stdout.write(time.strftime('%d-%m-%Y_%H:%M:%S')+"\t"+label+"\t"+text+"\n")
        sys.stdout.write(label+"\t"+text+"\n")

def info(text, verb=3):
    msg("INFO:", text, verb)

def logtime(text, verb=3):
    msg("TIME:", text, verb)

def warn(text, verb=0):
    msg("WARN:", text, verb)

def error(text, verb=0):
    msg("ERROR:", text, verb)
