"""
Taken from <https://stackoverflow.com/a/34406129>

------------------------------------------------------------------------------
You can then use it to debug your script like that. Pass in your script name
as the first argument if you want to run pdb on any warning:

$ python debug.py yourscript.py
------------------------------------------------------------------------------
Pass in warning type as the second argument if you only want pdb to run when
some particular type of warning is raised:

$ python debug.py yourscript.py DeprecationWarning
------------------------------------------------------------------------------
Line number as the third argument:

$ python debug.py yourscript.py DeprecationWarning 342
------------------------------------------------------------------------------
"""


import pdb, warnings, sys
import builtins


if __name__ == '__main__':
    args, n = [], len(sys.argv)
    if n < 2:
        sys.exit(1)
    elif n > 2:
        args.append(builtins.__dict__[sys.argv[2]])
        if n > 3:
            args.append(int(sys.argv[3]))
    warnings.simplefilter('error', *args)  # treat warnings as exceptions
    try:
        with open(sys.argv[1]) as f:
            code = compile(f.read(), sys.argv[1], 'exec')
            exec(code)
    except:
        pdb.post_mortem(sys.exc_info()[-1])
