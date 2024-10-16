import time
from functools import wraps
from contextlib import contextmanager
import tempfile
import os

def timing(logger=None):
    """ Function timing decorator 
    
    Keywords
    --------
    logger: None or logging object
        If none, print is used to output the timing results. If a
        logging object, the timing information is logged with the INFO
        logging priority.
    
    Returns
    -------
    Decorator wrapped function
    
    """
    def inner(f):
        @wraps(f)
        def wrap(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            dt = time.gmtime(te-ts)
            text = f"Function '{f.__name__}' completed in " +\
                   f"{time.strftime('%H:%M:%S', dt)}"
            if logger:
                # Use the specified logger if provided
                logger.info(text)
            else:
                print(text)
            return result
        return wrap
    return inner

class Timing:
    """ Timer context manager for local line execution timing"""

    def __init__(self, title, logger=None):
        self.title = title
        self.logger = logger
        self.ts = time.time()
    
    def __enter__(self):
        self.ts = time.time()
        if self.logger:
            self.logger.info(f"{self.title}: running...")
        else:
            print(f"{self.title}: running...", end="")
    
    def __exit__(self, *args):
        tf = time.time()
        dt = time.gmtime(tf - self.ts)
        if self.logger:
            self.logger.info(f"{self.title} completed in "
                             f"{time.strftime('%H:%M:%S', dt)}")
        else:
            print(f"completed in {time.strftime('%H:%M:%S', dt)}")

# Taken from https://stackoverflow.com/questions/4037481/caching-class-attributes-in-python
class cachedProperty(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)

        return attr

# There doesn't seem to be an easy way to prevent cubit (or other C
# apis) from printing to the console, which can unfortunately clutter
# the console very quickly. Initial digging suggested stopping the
# sys.stdout stream, but cubit doesn't print to this stream. After a lot
# more digging, I found that the python cubit library, which is a
# wrapper to the underlying C library, prints to filedescriptor 1. So, I
# adopted the general std.output capture methodology, but to
# filedescripter 1. See the following for reference:
#   - https://stackoverflow.com/questions/42952623/stop-python-module-from-printing
#   - https://stackoverflow.com/questions/9488560/capturing-print-output-from-shared-library-called-from-python-with-ctypes-module/41262627
@contextmanager
def suppressStream():
    # Open up a silent stream and point it to a temp file
    silent = tempfile.TemporaryFile()

    # Capture the current state of the stdout
    stdout = os.dup(1)

    # Point all output to the silent stream
    os.dup2(silent.fileno(), 1)  

    # Process the command
    try:  
        yield silent
    finally:
        # Upon exit, replace the output stream with the standard stdout
        os.dup2(stdout, 1)


        # Close the temp file stream
        silent.close()

        # Close old stdout stream. Note: this was added as you will
        # eventually run into a "Too many open files" error. See
        # https://stackoverflow.com/questions/36647498/how-to-close-file-descriptors-in-python
        # for reference.
        os.close(stdout)