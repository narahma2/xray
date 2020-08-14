import os
import subprocess
import matplotlib.pyplot as plt


def create_folder(new_folder):
        if not os.path.exists(new_folder):
                os.makedirs(new_folder)

        return new_folder


def exportEmf(savePath, plotName, fig=None, keepSVG=False):
    """
    Save a figure as an emf file.

    Parameters
    ----------
    savePath:   str, the path to the directory you want the image saved in
    plotName:   str, the name of the image
    fig:        matplotlib figure, (optional, default uses gca)
    keepSVG:    bool, whether to keep the interim svg file
    """
    inkscapePath = r"path\to\inkscape.exe"

    figFolder = savePath + r"\{}.{}"
    svgFile = figFolder.format(plotName,"svg")
    emfFile = figFolder.format(plotName,"emf")
    if fig:
        use=fig
    else:
        use=plt
    use.savefig(svgFile)
    subprocess.run([inkscapePath, svgFile, '-M', emfFile])

    if not keepSVG:
        os.system('del "{}"'.format(svgFile))
