import pandas as pd
import numpy as np


def xcom(xcom_path, att_column=3):
    """
    Takes in the older XCOM .txt file and returns the desired attenuation
    coefficient.
    =============
    --VARIABLES--
    xcom_path:      Path to .txt file.
    att_column:     Desired attenuation coefficient.
    """
    xcom_spct = pd.read_csv(xcom_path, sep='\t')

    # Convert abscissa from MeV to eV
    xcom_spct.iloc[:, 0] = xcom_spct.iloc[:, 0] * 1E6

    # Add residual value to edge locations for interpolation
    for i in range(len(xcom_spct.iloc[:, 0]) - 1):
        if xcom_spct.iloc[i, 0] == xcom_spct.iloc[i + 1, 0]:
            xcom_spct.iloc[i, 0] -= 1E-6
            xcom_spct.iloc[i + 1, 0] += 1E-6

    # Rename column name
    xcom_spct = xcom_spct.rename(columns={'Photon Energy (MeV)':
                                          'Photon Energy (eV)'})

    # Energy is in eV, Attenuation is in cm^2/g
    xcom_spct = {
                 'Energy': xcom_spct.iloc[:, 0],
                 'Attenuation': xcom_spct.iloc[:, att_column]
                 }

    return xcom_spct


def mass_atten(molec, comp=[1]):
    """
    Retrieves the mass attenuation coefficient data from NIST.
    See: <https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients>
    =============
    --VARIABLES--
    molec:      Element/compound chemical formula string as a list. If it's a
                mixture of compounds, pass materials in as a comma-separated
                list of strings, e.g.:
                Pure H2O: ['H2O']
                10% KI in H2O: ['H2O', 'KI']
    comp:       Mass composition of the mixture (optional). Order matches the
                molec variable, e.g. for 10% KI in H2O: [0.9, 0.1].
    """
    import re

    # Convert to arrays
    molec = np.array(molec)
    comp = np.array(comp)

    # Check to make sure data passed in proper format
    if comp.size is 1:
        if molec.size is not 1:
            print('Incorrect input format. Check function docstring.')
            return
    else:
        if molec.size is 1:
            print('Incorrect input format. Check function docstring.')
            return

    # Create list that contain the data for each material
    molec_mu = len(molec) * [None]
    molec_mu_en = len(molec) * [None]
    molec_energy = len(molec) * [None]

    for i, x in enumerate(molec):
        # Get their molecule formula
        ele_list = re.findall(
                              r'[A-Z][a-z]*|\d+',
                              re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', x)
                              )
        atoms = ele_list[::2]
        atom_count = np.array([int(x) for x in ele_list[1::2]])

        atom_data = len(atoms) * [None]
        interp_mu = len(atoms) * [None]
        interp_mu_en = len(atoms) * [None]
        raw_mu = len(atoms) * [None]
        raw_mu_en = len(atoms) * [None]
        raw_energy = len(atoms) * [None]

        # Read in data for each atom
        for j, y in enumerate(atoms):
            atom_data[j] = pd.read_excel(
                                         'nist_mass_atten_coeff.xlsx',
                                         sheet_name=y,
                                         skiprows=2
                                         )

            # Convert MeV to keV
            raw_energy[j] = atom_data[j].Energy.to_numpy() * 1000

            # Get coefficient values
            raw_mu[j] = atom_data[j].mu
            raw_mu_en[j] = atom_data[j].mu_en

        # Combine coefficients into a common energy axis
        [molec_energy[i], atom_mu] = common_energy(raw_mu, raw_energy)
        [_, atom_mu_en] = common_energy(raw_mu_en, raw_energy)

        # Get mass fraction for each atom
        atom_frac = molecule_weight(atoms, atom_count)

        # Calculate combined molecule coefficients
        molec_mu[i] = np.array(
                               [
                                atom_frac[j] * atom_mu[j]
                                for j,_ in enumerate(atoms)
                                ]
                               )
        molec_mu[i] = molec_mu[i].sum(axis=0)

        molec_mu_en[i] = np.array(
                                  [
                                   atom_frac[j] * atom_mu_en[j]
                                   for j,_ in enumerate(atoms)
                                   ]
                                  )
        molec_mu_en[i] = molec_mu_en[i].sum(axis=0)

    # Get common energy axis for all the molecules
    [energy, mu] = common_energy(molec_mu, molec_energy)
    [_, mu_en] = common_energy(molec_mu_en, molec_energy)

    # Calculate combined mixture coefficients
    mu = np.array(
                  [
                   comp[j] * mu[j]
                   for j,_ in enumerate(molec)
                   ]
                  )
    mu = mu.sum(axis=0)

    mu_en = np.array(
                     [
                      comp[j] * mu_en[j]
                      for j,_ in enumerate(molec)
                      ]
                     )
    mu_en = mu_en.sum(axis=0)

    return energy, mu, mu_en


def common_energy(coeffs, energies):
    """
    Takes in attenuation coefficients and their corresponding energy axes and
    returns coefficients interpolated to a common energy axis.
    =============
    --VARIABLES--
    coeffs:         List of attenuation coefficients.
    energies:       List of the energy axes for the corresponding coeffs.
    """
    from scipy.interpolate import interp1d

    interp_coeff = len(coeffs) * [None]

    # Add residual value to edge locations for interpolation
    for n, coeff in enumerate(coeffs):
        for i in range(len(coeff) - 1):
            if coeff[i] == coeff[i + 1]:
                coeff[i] -= 1E-5
                coeff[i + 1] += 1E-5

        # Create interpolate function for the attenuation
        interp_coeff[n] = interp1d(
                                   x=energies[n],
                                   y=coeff
                                   )

    # Combine energy axes into one unique, sorted array
    energy = pd.DataFrame([item for sublist in energies for item in sublist])
    energy.drop_duplicates(inplace=True)
    energy.sort_values(by=0, inplace=True)
    energy.reset_index(drop=True, inplace=True)
    energy = energy.to_numpy()

    # Interpolate the coefficients to the combined energy axis
    for n, coeff in enumerate(coeffs):
        coeffs[n] = interp_coeff[n](energy)
        coeffs[n] = np.array([x[0] for x in coeffs[n]])

    # Energy returns as an array of lists, flatten it out
    energy = np.array([x[0] for x in energy])

    return energy, coeffs


def molecule_weight(atoms, atom_count):
    """
    Get the mass fraction for each atom in a molecule.
    =============
    --VARIABLES--
    atoms:          List of atoms by symbol.
    atom_count:     List of atom count as integers.
    """
    import xlrd

    atom_mass = np.zeros(len(atoms),)

    for i, atom in enumerate(atoms):
        atom_mass[i] = xlrd.open_workbook('nist_mass_atten_coeff.xlsx') \
                           .sheet_by_name(atom) \
                           .cell(0, 1) \
                           .value

    # Get total molecule mass
    molecule_mass = atom_mass.sum()

    # Calculate mass fraction for each atom
    atom_frac = atom_mass / molecule_mass

    return atom_frac


def test_plots():
    import matplotlib.pyplot as plt

    h2o = mass_atten(['H2O'])
    ki = mass_atten(['KI'])
    ki50 = mass_atten(['H2O', 'KI'], [0.5, 0.5])

    plt.figure()
    plt.plot(
             h2o[0],
             h2o[1],
             color='b',
             linestyle='solid',
             label='H$_2$O'
             )
    plt.plot(
             ki[0],
             ki[1],
             color='r',
             linestyle='solid',
             linewidth=2.0,
             label='KI'
             )
    plt.plot(
             ki50[0],
             ki50[1],
             color='k',
             linestyle='dashed',
             linewidth=2.0,
             label='50% KI/H$_2$O'
             )
    plt.xlim([0, 200])
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Attenuation Coefficient (cm$^2$/g)')
    plt.title('$\mu$')
    plt.savefig('mu.png')
    plt.close()

    plt.figure()
    plt.plot(
             h2o[0],
             h2o[2],
             color='b',
             linestyle='solid',
             label='H$_2$O'
             )
    plt.plot(
             ki[0],
             ki[2],
             color='r',
             linestyle='solid',
             linewidth=2.0,
             label='KI'
             )
    plt.plot(
             ki50[0],
             ki50[2],
             color='k',
             linestyle='dashed',
             linewidth=2.0,
             label='50% KI/H$_2$O'
             )
    plt.xlim([0, 200])
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Attenuation Coefficient (cm$^2$/g)')
    plt.title('$\mu_{en}$')
    plt.savefig('mu_en.png')
    plt.close()

    return
