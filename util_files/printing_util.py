# -*- coding: utf-8 -*-
import sys


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100, use_stderr=False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
        use_stderr  - Optional  : Asks if we want the output to go into stderr instead of stdout (Bool)
    """
    if use_stderr:
        out = sys.stderr
    else:
        out = sys.stdout

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    out.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        out.write('\n')
    out.flush()
