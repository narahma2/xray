"""
Created on Wed May 22 17:12:11 2019

@author: rahmann
"""

import glob
import numpy as np
import tecplot as tp

from os import path
from libim7 import readim7
from scipy.ndimage.interpolation import shift
from general.misc import create_folder


def load_davis(im7_path):
    """
    Function that loads in the LaVision DaVis reconstructed volume. Returns
    the volume as a structured 3D numpy array and the X/Y/Z extents in mm.

    For full conversion to a Tecplot .PLT file, use convert_tp() below,
    this function only does the preliminary step of loading the volume.

    INPUT VARIABLES
    ---------------
    im7_path:   Path to a single reconstructed volume. Can be a folder path
                containing all B0*.im7 files for that single volume, or a
                file path to the B0*.im7 file containing the entire volume
                (check your project directory!)
    """
    # Check how to read in the im7 files.
    # IF AN INDIVIDUAL VOLUME IS SPLIT UP INTO MULTIPLE B0*.im7
    # FILES FOR EACH Z, THEN VOLUME GRID MUST BE "BUILT UP" BELOW.
    if path.isdir(im7_path):
        im7_files = glob.glob(im7_path + "*.im7")

        for n, im7_file in enumerate(im7_files):
            buf, att = readim7(im7_file)

            if n == 0:
                # Number of voxels in X, Y, Z
                nx = buf.nx
                ny = buf.ny
                nz = len(im7_files)
                volume = np.zeros((nx, ny, nz))

            # Transpose the read in block so that volume is in
            # (nx, ny) dimensions
            volume[:,:,n] = buf.blocks.T[:,:,0]

    # IF AN INDIVIDUAL VOLUME IS CONTAINED WITHIN ONE B0*.im7 FILE,
    # THEN VOLUME GRID CAN BE READ IN DIRECTLY BELOW.
    elif path.isfile(im7_path):
        buf, att = readim7(im7_path)
        # Number of voxels in X, Y, Z
        nx = buf.nx
        ny = buf.ny
        nz = buf.nz
        # Transpose the read in block so that volume is in
        # (nx, ny, nz) dimensions
        volume = buf.blocks.T

    # Millimeters per voxel from calibration
    scale_x = float(att['_SCALE_X'].rsplit(' ')[0])
    scale_y = float(att['_SCALE_Y'].rsplit(' ')[0])
    scale_z = float(att['_SCALE_Z'].rsplit(' ')[0])

    # Starting point in X, Y, Z in millimeters
    start_x = float(att['_SCALE_X'].rsplit('\nmm')[0].rsplit(' ')[1])
    start_y = float(att['_SCALE_Y'].rsplit('\nmm')[0].rsplit(' ')[1])
    start_z = float(att['_SCALE_Z'].rsplit('\nmm')[0].rsplit(' ')[1])

    # Create volume grid extents in X, Y, Z based on calibrated
    # millimeter dimensions
    x_mm = np.linspace(start=start_x, stop=start_x + nx*scale_x, num=nx,
                       dtype=np.float32)
    y_mm = np.linspace(start=start_y, stop=start_y + ny*scale_y, num=ny,
                       dtype=np.float32)
    z_mm = np.linspace(start=start_z, stop=start_z + nz*scale_z, num=nz,
                       dtype=np.float32)

    del buf, att

    return volume, x_mm, y_mm, z_mm


def convert_tp(
               im7_path,
               dataset,
               zone_name,
               crop_x1=None,
               crop_x2=None,
               crop_y1=None,
               crop_y2=None,
               crop_z1=None,
               crop_z2=None,
               vshift=None
               ):
    """
    Function that takes in a reconstructed volume from LaVision DaVis and
    converts into a Tecplot readable form. For ease of use, this function
    only works on *individual volumes*, however the calling procedure for
    the volume depends on how the volumes are saved by LaVision DaVis (as
    individual B0*.im7 files or a series of B0*.im7 files). SEE THE
    <im7_path> VARIABLE.

    If you want to convert a multitude of reconstructed volumes (e.g. a
    time series), simply call this function within a for-loop.

    For saving the files after conversion, call the func ~save_plt~
    defined below in the script.

    INPUT VARIABLES
    ---------------
    im7_path:   Path to a single reconstructed volume. Can be a folder path
                containing all B0*.im7 files for that single volume, or a
                file path to the B0*.im7 file containing the entire volume
                (check your project directory!)

    dataset:    Tecplot dataset

    zone_name:  Name of zone within Tecplot (e.g. Q5, Q14, Mass, etc.)

    crop_x1:    OPTIONAL. If cropping is desired in X range, this is the
                beginning crop point IN MILLIMETERS.

    crop_x2:    OPTIONAL. If cropping is desired in X range, this is the
                ending crop point IN MILLIMETERS.

    crop_y1:    OPTIONAL. If cropping is desired in Y range, this is the
                beginning crop point IN MILLIMETERS.

    crop_y2:    OPTIONAL. If cropping is desired in Y range, this is the
                ending crop point IN MILLIMETERS.

    crop_z1:    OPTIONAL. If cropping is desired in Z range, this is the
                beginning crop point IN MILLIMETERS.

    crop_z2:    OPTIONAL. If cropping is desired in Z range, this is the
                ending crop point IN MILLIMETERS.

    vshift:     OPTIONAL. I don't remember what this does. I assume it
                shifts the volume in a certain direction. Might be useful
                when needing to overlap different volumes (e.g. Q14/Q5 of
                OH transitions for temperature calculation)?
    """
    # Load in the DaVis file
    volume, x_mm, y_mm, z_mm = load_davis(im7_path)

    # OPTIONAL: Cropping the volume down to size (in millimeters)
    if crop_x1 is not None:
        x_mm_1 = np.argmin(abs(x_mm - crop_x1))
    else:
        x_mm_1 = 0

    if crop_x2 is not None:
        x_mm_2 = np.argmin(abs(x_mm - crop_x2))
    else:
        x_mm_2 = nx

    if crop_y1 is not None:
        y_mm_1 = np.argmin(abs(y_mm - crop_y2))
    else:
        y_mm_1 = 0

    if crop_y2 is not None:
        y_mm_2 = np.argmin(abs(y_mm - crop_y1))
    else:
        y_mm_2 = ny

    if crop_z1 is not None:
        z_mm_1 = np.argmin(abs(z_mm - crop_z1))
    else:
        z_mm_1 = 0

    if crop_z2 is not None:
        z_mm_2 = np.argmin(abs(z_mm - crop_z2))
    else:
        z_mm_2 = nz

    # Crops down the X/Y/Z extents if needed
    x_mm = x_mm[x_mm_1:x_mm_2]
    y_mm = y_mm[y_mm_1:y_mm_2]
    z_mm = z_mm[z_mm_1:z_mm_2]

    # Reset number of voxels in X, Y, Z (in case cropping was done)
    nx = len(x_mm)
    ny = len(y_mm)
    nz = len(z_mm)

    # Crop down volume grid
    volume = volume[x_mm_1:x_mm_2, y_mm_1:y_mm_2, z_mm_1:z_mm_2]

    if vshift is not None:
        volume = shift(volume, shift=vshift)

    # Create 'Ordered' zone
    zone = dataset.add_ordered_zone(zone_name, (nx,ny,nz))

    # Resize the X values to fill the entire grid
    zone.values('X')[:] = np.tile(np.tile(x_mm, (1, ny))[0], (1, nz))[0]

    # Resize the Y values to fill the entire grid
    zone.values('Y')[:] = np.tile(np.repeat(y_mm, nx), (1, nz))[0]

    # Resize the Z values to fill the entire grid
    zone.values('Z')[:] = np.repeat(z_mm, nx*ny)

    # Flatten the volume to fill the entire grid (in 'F'=Fortran or
    # column-wise order)
    zone.values('Intensity')[:] = volume.flatten(order='F')



def save_plt(dataset, plt_folder, plt_name, zone=None):
    """
    Function that saves the Tecplot dataset/zone into a .plt file.
    This function saves to an individual .plt file. If you want to save a
    series of volumes (such as for each zone), then call this function
    inside a for-loop.

    INPUT VARIABLES
    ---------------
    dataset:        Tecplot dataset.

    plt_folder:     Output folder path where the .plt file will be saved.

    plt_name:       Desired name of the .plt file.

    zone:           OPTIONAL. Specific zone to save (can call by name or
                    index starting at 1). If not specified, will save
                    the entire dataset.
    """
    # Create folder if it doesn't exist
    plt_fld = create_folder(plt_fld)

    # Save the dataset to a PLT file
    tp.data.save_tecplot_plt(
                             '{0}/{1}.plt'.format(plt_fld, plt_name),
                             dataset=dataset,
                             zones=zone
                             )


def volume_masking(data, x_val, y_val, z_val, zone_index=1):
    """
    Function that masks out cells based on the projections onto the
    X/Y/Z planes. Useful to cut out any noise.

    INPUT VARIABLES
    ---------------
    data:           Tecplot dataset.

    x_val:          Threshold for intensities for masking based on
                    projection onto the X plane (YZ view).

    y_val:          Threshold for intensities for masking based on
                    projection onto the Y plane (XZ view).

    z_val:          Threshold for intensities for masking based on
                    projection onto the Z plane (XY view).

    zone_index:     OPTIONAL. Index of the zone to be masked out, if not
                    specified will do the first one.
    """
    volume = data.zone(zone_index).values('Intensity').as_numpy_array()
    (x, y, z) = data.zone(zone_index).dimensions
    volume.resize((z,y,x))

    x_proj = np.sum(volume, axis=2)
    y_proj = np.sum(volume, axis=1)
    z_proj = np.sum(volume, axis=0)

    x_proj[x_proj < x_val] = 0
    x_proj[x_proj > 0] = 1
    y_proj[y_proj < y_val] = 0
    y_proj[y_proj > 0] = 1
    z_proj[z_proj < z_val] = 0
    z_proj[z_proj > 0] = 1

    masked = volume
    for i in range(x):
        masked[:,:,i] = masked[:,:,i] * x_proj
    for j in range(y):
        masked[:,j,:] = masked[:,j,:] * y_proj
    for k in range(z):
        masked[k,:,:] = masked[k,:,:] * z_proj

    data.zone(zone_index).values('Intensity')[:] = masked.flatten(order='C')


def volume_smoothing(zone_index=1, passes=10, strength=0.5):
    """
    Function that smooths the volume. Falls back to default parameters
    if none are given.

    INPUT VARIABLES
    ---------------
    zone_index:     OPTIONAL. Index of the zone to be smoothed, if not
                    specified will do default to the first one.

    passes:         OPTIONAL. Number of smoothing passes to be done, if
                    not specified will default to 10.

    strength:       OPTIONAL. Smoothing strength coefficient, if not
                    specified will default to 0.5.
    """
    tp.macro.execute_command('''$!Smooth Zone = ''' + str(zone_index+1) +
                             '''Var = 4 NumSmoothPasses = ''' +
                             str(passes) + '''SmoothWeight = ''' +
                             str(strength) + '''SmoothBndryCond = Fixed''')
