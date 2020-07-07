import glob
import h5py
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d

from general.misc import create_folder
from general.calc_statistics import rmse


prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'
hdf5_fld = '{0}/HDF5'.format(prj_fld)
mb_fld = create_folder('{0}/Monobeam/AeroECN'.format(prj_fld))


def vert_scan(xx, image_v_x, yy, image_v_y, img, EPL, x, y):
    # Get vertical scan indices
    v_x = xx[image_v_x]
    v_y1 = yy[image_v_y[0]]
    v_y2 = yy[image_v_y[1]]
    v_yslice = list(range(
                          np.argmin(abs(v_y1 - np.mean(y, axis=1))),
                          np.argmin(abs(v_y2 - np.mean(y, axis=1))) + 1
                          )
                    )
    v_xslice = np.argmin(np.abs(v_x - x[0]))

    # Injector data
    inj_xdata = [y[z][0] for z in v_yslice]
    inj_ydata = [EPL[z][v_xslice] for z in v_yslice]

    # Get vertical scan image data for plots
    img_xdata = yy[image_v_y[0]:image_v_y[1]]
    img_ydata = img[image_v_y[0]:image_v_y[1], image_v_x]

    return inj_xdata, inj_ydata, img_xdata, img_ydata, v_x


def horiz_scan(xx, image_h_x, yy, image_h_y, img, EPL, x, y):
    # Get horizontal scan indices
    h_y = yy[image_h_y]
    h_x1 = xx[image_h_x[0]]
    h_x2 = xx[image_h_x[1]]
    h_xslice = list(range(
                          np.argmin(abs(h_x1 - x[0])),
                          np.argmin(abs(h_x2 - x[0])) + 1
                          )
                    )
    h_yslice = np.argmin(np.abs(h_y - np.mean(y, axis=1)))

    # Injector data
    inj_xdata = [x[0][z] for z in h_xslice]
    inj_ydata = [EPL[h_yslice][z] for z in h_xslice]

    # Get horizontal scan image data for plots
    img_xdata = xx[image_h_x[0]:image_h_x[1]+1]
    img_ydata = img[image_h_y, image_h_x[0]:image_h_x[1]+1]

    return inj_xdata, inj_ydata, img_xdata, img_ydata, h_y


def plot_fig(img_x, img_y, inj_x, inj_y, xlabel, const, const_val, name):
    plt.figure()
    plt.plot(
             img_x,
             img_y,
             label='WB',
             linestyle='solid',
             linewidth=2.0,
             color='k'
             )
    plt.plot(
             inj_x,
             inj_y,
             label='MB',
             linestyle='dashed',
             linewidth=2.0,
             color='b'
             )
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(r'EPL ($\mu$m)')
    plt.title('{0}: {1} = {2:0.2f} mm'.format(name, const, const_val))
    plt.savefig('{0}/{1}.png'.format(mb_fld, name))
    plt.close()

    return


def main():
    # See 'Spray Imaging' in Excel workbook
    cm_px = np.loadtxt('{0}/cm_px.txt'.format(prj_fld))

    fly_scan = 'Fly_Scans_208_340'
    images = ['center', 'left', 'lower']

    # Injector center position on image (in pixels)
    image_inj_x = [390, 542, 694]
    image_inj_y = [18, 18, -185]

    # Image positions (see 'WB & MB Python' in Excel workbook)
    image_v_x = [[390, 115], [238, 137], [250, 165]]
    image_v_y = [29, 330]
    image_h_x = [0, 767]
    image_h_y = [list(range(29, 330)), [180, 275], [200, 315]]

    # Load in fly scan data
    f = h5py.File('{0}/{1}.hdf5'.format(hdf5_fld, fly_scan), 'r')
    x = np.array(f['X'])
    y = np.array(f['Y'])
    ext_lengths = np.array(f['Radiography'])

    # Attenuation coefficient (total w/o coh. scattering - cm^2/g)
    # Convert to mm^2/g for mm, multiply by density in g/mm^3
    # Pure water @ 8 keV
    # <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>
    atten_coeff = (1.006*10*(10*10))*(0.001)

    # Calculate EPL and convert to um from mm
    EPL = (ext_lengths / atten_coeff) * 1000

    # Set the image type to load
    img_types = ['TimeAvg', 'TimeRes']

    for img_type in img_types:
        for n, image in enumerate(images):
            # Load in corresponding EPL image and convert to um from cm
            img_path = glob.glob(
                                 '{0}/Images/*AeroECN*/EPL/{1}/*{2}B*'
                                 .format(prj_fld, img_type, image)
                                 )[0]
            img = np.array(Image.open(img_path))
            img *= 10000

            xx = np.linspace(1, 768, 768)
            xx = xx - image_inj_x[n]
            # Convert to mm
            xx = xx * cm_px * 10

            yy = np.linspace(1, 352, 352)
            yy = yy - image_inj_y[n]
            # Convert to mm
            yy = yy * cm_px * 10

            # Get vertical scan data (V1)
            [inj_xdata, inj_ydata, img_xdata, img_ydata, v_x] = \
                vert_scan(xx, image_v_x[n][0], yy, image_v_y, img, EPL, x, y)
            plot_fig(img_xdata, img_ydata, inj_xdata, inj_ydata,
                     'Vertical Location (mm)', 'x', v_x,
                     '{0}_{1}_v1'.format(img_type, image))

            # Get vertical scan data (V2)
            [inj_xdata, inj_ydata, img_xdata, img_ydata, v_x] = \
                vert_scan(xx, image_v_x[n][1], yy, image_v_y, img, EPL, x, y)
            plot_fig(img_xdata, img_ydata, inj_xdata, inj_ydata,
                     'Vertical Location (mm)', 'x', v_x,
                     '{0}_{1}_v2'.format(img_type, image))

            if n == 0:
                u = 60-29
                v = 150-29
            else:
                u = 0
                v = 1

            # Get horizontal scan data (H1)
            [inj_xdata, inj_ydata, img_xdata, img_ydata, h_y] = \
                horiz_scan(xx, image_h_x, yy, image_h_y[n][u], img, EPL, x, y)
            plot_fig(img_xdata, img_ydata, inj_xdata, inj_ydata,
                     'Horizontal Location (mm)', 'y', h_y,
                     '{0}_{1}_h1'.format(img_type, image))

            # Get horizontal scan data (H2)
            [inj_xdata, inj_ydata, img_xdata, img_ydata, h_y] = \
                horiz_scan(xx, image_h_x, yy, image_h_y[n][v], img, EPL, x, y)
            plot_fig(img_xdata, img_ydata, inj_xdata, inj_ydata,
                     'Horizontal Location (mm)', 'y', h_y,
                     '{0}_{1}_h2'.format(img_type, image))

            # Build up RMSE array for the horizontal center scan
            if n == 0:
                center_rmse = np.zeros(len(image_h_y[n]))
                image_pos = np.zeros(len(image_h_y[n]))
                scan_pos = np.zeros(len(image_h_y[n]))

                for q, k in enumerate(image_h_y[n]):
                    [inj_xdata, inj_ydata, img_xdata, img_ydata, h_y] = \
                        horiz_scan(xx, image_h_x, yy, k, img, EPL, x, y)
                    interp = interp1d(inj_xdata, inj_ydata, fill_value=0,
                                      bounds_error=False)
                    center_rmse[q] = rmse(img_ydata, interp(img_xdata))
                    mid_pt = int(len(inj_ydata)/2)
                    char_dim = np.median(inj_ydata[mid_pt-10:mid_pt+10])
                    center_rmse[q] /= char_dim
                    image_pos[q] = k
                    scan_pos[q] = h_y

            # Plot the RMSE curve
            plt.figure()
            plt.plot(
                     image_pos,
                     center_rmse,
                     linewidth=2.0,
                     linestyle='solid',
                     color='k'
                     )
            plt.xlabel('Vertical Location (px)')
            plt.ylabel('Norm. RMSE (-)')
            plt.title('RMSE for AeroECN @ 11.1% KI')
            plt.savefig('{0}/{1}_rmse.png'.format(mb_fld, img_type))
            plt.close()


# Run this script
if __name__ == '__main__':
    main()
