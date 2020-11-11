import glob
import h5py
import numpy as np

from pyevtk.hl import imageToVTK
from pyevtk.vtk import VtkGroup
from PIL import Image
from scipy.ndimage import rotate, shift

from general.Tecplot.d10_to_tecplot import load_davis
from general.misc import create_folder
from general.spectrum_modeling import density_KIinH2O


# Input and output folders
inp_fld = '/mnt/e/DaVisProjects/Xray_IJ_2018_Binned2x2/'
out_fld = '/mnt/r/X-ray Tomography/2018_DataSets/'


def convertVTK(ij_inp, rot, ij_out, density):
    """
    Converts individual IJ volumes.

    VARIABLES
    ---------
    ij_inp:     Input folder containing each of the LaVision DaVis volumes.
    rot:        Rotation dict to correct the volumes (see Jupyter notebook).
    ij_out:     Location to save the VTK files.
    density:    Density of the liquid to use for conversion to LVF.
                Use density=1 for no correction for the mixing cases.
    """
    # Create folders as needed
    create_folder('{0}/VTK'.format(ij_out))
    create_folder('{0}/SliceIJ'.format(ij_out))
    create_folder('{0}/Slice+4'.format(ij_out))
    create_folder('{0}/Slice+8'.format(ij_out))

    # Get total number of volumes and initialize the impingement point var 
    ijPt0_den = np.zeros((len(ij_inp), 1))
    ijPt4_den = np.zeros((len(ij_inp), 1))
    ijPt8_den = np.zeros((len(ij_inp), 1))

    for i, inp in enumerate(ij_inp):
        # Load in the density volume
        # Only load in the X/Y/Z values on the first volume (always the same)
        if i == 0:
            volume, x_mm, y_mm, z_mm = load_davis(inp)

            # Flip the Y values
            y_mm = y_mm[::-1]

            # Get the location of the impingement point in mm space
            X, Y, Z = np.meshgrid(x_mm, y_mm, z_mm, indexing='ij')

            # Rotate the X, Y, Z meshgrids
            X = rotate(X, rot['Angle'], axes=rot['Axes'], reshape=False)
            Y = rotate(Y, rot['Angle'], axes=rot['Axes'], reshape=False)
            Z = rotate(Z, rot['Angle'], axes=rot['Axes'], reshape=False)

            # Center point in voxels
            center_X = 90
            center_Y = 129
            center_Z = 195

            # Impingement point in voxel coordinates (found from the rotated
            # projection images)
            ij_X = X[center_X, center_Y, center_Z]
            ij_Y = Y[center_X, center_Y, center_Z]
            ij_Z = Z[center_X, center_Y, center_Z]

            # Center the x, y, z vectors
            x_mm -= ij_X
            y_mm -= ij_Y
            z_mm -= ij_Z

            # Crop down the volume (see Jupyter notebook)
            x1 = np.argmin(np.abs(x_mm - -4.7))
            x2 = np.argmin(np.abs(x_mm - 4.7))
            y1 = np.argmin(np.abs(y_mm - -5))
            y2 = np.argmin(np.abs(y_mm - 14))
            z1 = np.argmin(np.abs(z_mm - -10))
            z2 = np.argmin(np.abs(z_mm - 10))

            # Get the origin (defines corner of the grid)
            origin = [x_mm[x1], y_mm[y1], z_mm[z1]]

            # Cropping
            x_mm = x_mm[x1:x2]
            y_mm = y_mm[y1:y2]
            z_mm = z_mm[z1:z2]

            # Save the volume extents
            np.save('{0}/VTK/x_mm.npy'.format(ij_out), x_mm,
                    allow_pickle=False)
            np.save('{0}/VTK/y_mm.npy'.format(ij_out), y_mm,
                    allow_pickle=False)
            np.save('{0}/VTK/z_mm.npy'.format(ij_out), z_mm,
                    allow_pickle=False)

            # Get the APS locations
            ind0X = np.argmin(np.abs(x_mm - 0))
            ind0Z = np.argmin(np.abs(z_mm - 0))
            ind0Y = np.argmin(np.abs(y_mm - 0))
            ind4Y = np.argmin(np.abs(y_mm - 4))
            ind8Y = np.argmin(np.abs(y_mm - 8))

        else:
            volume, _, _, _ = load_davis(inp)

        # Rotate the volume
        volume = rotate(volume, rot['Angle'], axes=rot['Axes'], reshape=False)

        # Calculate optical depth for the mixing cases
        # Volumes were inverted (1 - buffer) in DaVis
        if density == 1:
            volume = -np.log(1 - volume)

        # Get voxel size
        dx = np.abs(x_mm[1] - x_mm[0])
        dy = np.abs(y_mm[1] - y_mm[0])
        dz = np.abs(z_mm[1] - z_mm[0])
        voxelSize = (dx, dy, dz)

        # Cropping
        volume = volume[x1:x2, y1:y2, z1:z2].astype('float32')

        # Get the intensity value at the impingement point (as density)
        ijPt0_den[i] = volume[ind0X, ind0Y, ind0Z] / dx
        ijPt4_den[i] = volume[ind0X, ind4Y, ind0Z] / dx
        ijPt8_den[i] = volume[ind0X, ind8Y, ind0Z] / dx

        # Extract the slices as solution density
        slice0 = volume[:, ind0Y, :].astype('float32') / dx
        slice4 = volume[:, ind4Y, :].astype('float32') / dx
        slice8 = volume[:, ind8Y, :].astype('float32') / dx

        # Save slices as numpy files
        np.save('{0}/SliceIJ/{1:03d}'.format(ij_out, i), slice0,
                allow_pickle=False)
        np.save('{0}/Slice+4/{1:03d}'.format(ij_out, i), slice4,
                allow_pickle=False)
        np.save('{0}/Slice+8/{1:03d}'.format(ij_out, i), slice8,
                allow_pickle=False)

        # Convert density to liquid volume fraction (LVF)
        LVF = volume / density

        # Convert files to VTK format (make sure to normalize by grid size!)
        imageToVTK(
                   '{0}/VTK/{1:03d}'.format(ij_out, i),
                   origin=origin,
                   spacing=voxelSize,
                   pointData={'LVF': LVF/dx},
                   )

    # Create a time series file
    seriesVTK(ij_out)

    # Save the impingement point values as density and LVF
    np.save('{0}/ijPt0_den'.format(ij_out), ijPt0_den)
    np.save('{0}/ijPt0_LVF'.format(ij_out), ijPt0_den / density)

    np.save('{0}/ijPt4_den'.format(ij_out), ijPt4_den)
    np.save('{0}/ijPt4_LVF'.format(ij_out), ijPt4_den / density)

    np.save('{0}/ijPt8_den'.format(ij_out), ijPt8_den)
    np.save('{0}/ijPt8_LVF'.format(ij_out), ijPt8_den / density)


def seriesVTK(ij_out):
    """
    Creates a PVD file that can visualize all time steps in ParaView.

    VARIABLES
    ---------
    ij_out:     Location of the VTK files from convertVTK()
    """
    # Read in the VTK files made from convertVTK()
    files = glob.glob('{0}/VTK/*.vti'.format(ij_out))

    # Initialize the group
    g = VtkGroup('{0}/VTK/timeSeries'.format(ij_out))

    for vtkFile in files:
        # Get time step
        i = int(vtkFile.rsplit('/')[-1].rsplit('.')[0])

        # Calculate time step in us (20 kHz imaging)
        timeStep = (i / (20E3)) * 1E6

        # Add file to group
        g.addFile(filepath=vtkFile, sim_time=timeStep)

    # Save the group (will be a ParaView PVD file)
    g.save()


def main():
    # Get 50% KI in H2O density in g/cm^3, convert to ug/mm^3
    density = density_KIinH2O(50) * 1000

    # Location of the 0.30 gpm input and outputs
    ij_0p30_inp = glob.glob('{0}/IJ_0p30_Filtered_Masked_125um/'
                            'Binned_2x2/FastMART_500iter_Smooth500iter/'
                            'Data/S0*/'.format(inp_fld))
    ij_0p30_out = create_folder('{0}/SprayVol/IJ_0p30gpm/'
                                .format(out_fld))

    # Location of the 0.45 gpm input and outputs
    ij_0p45_inp = glob.glob('{0}/IJ_0p45_Filtered_Masked_125um/'
                            'Binned_2x2/FastMART_500iter_Smooth500/'
                            'Data/S0*/'.format(inp_fld))
    ij_0p45_out = create_folder('{0}/SprayVol/IJ_0p45gpm/'
                                .format(out_fld))

    # Location of the mixing 0.30 gpm input and outputs
    ij_mix0p30_inp = glob.glob('{0}/Mixing_0p30/CompressExpand/'
                               'FastMART_500iter_Smooth500iter/'
                               'Data/S0*/'.format(inp_fld))
    ij_mix0p30_out = create_folder('{0}/SprayVol/IJ_Mixing_0p30gpm/'
                                   .format(out_fld))

    # Location of the mixing 0.45 gpm input and outputs
    ij_mix0p45_inp = glob.glob('{0}/Mixing_0p45/CompressExpand/'
                               'FastMART_500iter_Smooth500iter/'
                               'Data/S0*/'.format(inp_fld))
    ij_mix0p45_out = create_folder('{0}/SprayVol/IJ_Mixing_0p45gpm/'
                                .format(out_fld))

    # Rotations for the volumes (see Jupyter Notebook)
    rot = {'Angle': 13, 'Axes': (2, 0)}

    # Convert the individual volumes
    convertVTK(ij_0p30_inp, rot, ij_0p30_out, density)
    convertVTK(ij_0p45_inp, rot, ij_0p45_out, density)
    convertVTK(ij_mix0p30_inp, rot, ij_mix0p30_out, density=1)
    convertVTK(ij_mix0p45_inp, rot, ij_mix0p45_out, density=1)


if __name__ is '__main__':
    main()
