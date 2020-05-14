import sys


def paths():
    if sys.platform == 'win32':
        sys.path.append('E:/GitHub/xray/general')
        sys.path.append('E:/GitHub/xray/temperature')
        sys_folder = 'R:'
    elif sys.platform == 'linux':
        sys.path.append('/mnt/e/GitHub/xray/general')
        sys.path.append('/mnt/e/GitHub/xray/temperature')
        sys_folder = '/mnt/r/'

    return sys_folder
