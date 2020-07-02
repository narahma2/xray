import pandas as pd
import numpy as np


fld = '/mnt/e/GitHub/xray/general/resources'


def mass_atten(molec, comp=[1], xcom=1, col=None, keV=200):
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
                molec variable, e.g. for 10% KI in H2O: [0.9, 0.1]. Can be
                entered in decimal form (0-1) or percent form (0-100), data
                will be normalized in this code to the 0-1 scale.
    xcom:       1: first database (w/ specific interactions listed) (default)
                2: new updated energy-absorption coefficient (total coeff).
    col:        Which attenuation coefficient to return, depedent on xcom.
                if xcom == 1, col corresponds to:
                    1: Coherent scattering
                    2: Incoherent scattering
                    3: Photoelectric absorption
                    4: Pair production in nuclear field
                    5: Pair production in electron field
                    6: Total attenuation with coherent scattering
                    7: Total attenuation without coherent scattering (default)
                if xcom == 2, col corresponds to:
                    1: Total mass atten. coeff.
                    2: Energy-absorption corrected atten. coeff. (default)
    keV:        Cut-off for energy axis in keV. Defaults to 200 keV.
    """
    # Check if common name is passed and manually set the formula/composition
    if molec == ['Air'] or molec == ['air']:
        molec = ['N2', 'O2', 'Ar']
        comp = [0.78, 0.21, 0.01]
    elif molec == ['YAG']:
        molec = ['Y3Al5O12']
    elif molec == ['LuAG']:
        molec = ['Lu3Al5O12']

    # Select desired database and set default column
    if xcom == 1:
        database = '{0}/nist_xcom1.xlsx'.format(fld)
        if col is None:
            col = 7
    elif xcom == 2:
        database = '{0}/nist_xcom2.xlsx'.format(fld)
        if col is None:
            col = 2

    # Convert to arrays
    molec = np.array(molec)
    comp = np.array(comp)

    # Normalize the composition parameter if needed
    comp = comp / comp.sum()

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
    molec_coeff = len(molec) * [None]
    molec_energy = len(molec) * [None]

    for i, x in enumerate(molec):
        # Get molecule information
        [atoms, atom_count, atom_frac] = molecule_info(x)

        atom_data = len(atoms) * [None]
        interp_mu = len(atoms) * [None]
        raw_coeff = len(atoms) * [None]
        raw_energy = len(atoms) * [None]

        # Read in data for each atom
        for j, y in enumerate(atoms):
            atom_data[j] = pd.read_excel(
                                         database,
                                         sheet_name=y,
                                         skiprows=2,
                                         engine='openpyxl'
                                         )

            # Convert MeV to keV
            raw_energy[j] = atom_data[j].Energy.to_numpy() * 1000

            # Get coefficient values
            raw_coeff[j] = atom_data[j].iloc[:, col]

        # Combine coefficients into a common energy axis
        [molec_energy[i], atom_coeff] = common_energy(raw_coeff, raw_energy)

        # Calculate combined molecule coefficients
        molec_coeff[i] = np.array(
                               [
                                atom_frac[j] * atom_coeff[j]
                                for j,_ in enumerate(atoms)
                                ]
                               )
        molec_coeff[i] = molec_coeff[i].sum(axis=0)

    # Get common energy axis for all the molecules
    [energy, coeff] = common_energy(molec_coeff, molec_energy)

    # Calculate combined mixture coefficients
    coeff = np.array(
                     [
                      comp[j] * coeff[j]
                      for j,_ in enumerate(molec)
                      ]
                     )
    coeff = coeff.sum(axis=0)

    # Mask out values greater than keV cut-off
    mask = energy <= keV

    return {'Energy': energy[mask], 'Attenuation': coeff[mask]}


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
    for n, energy in enumerate(energies):
        for i in range(len(energy) - 1):
            if energies[n][i] == energies[n][i + 1]:
                energies[n][i] -= 1E-5
                energies[n][i + 1] += 1E-5

        # Create interpolate function for the attenuation
        interp_coeff[n] = interp1d(
                                   x=energies[n],
                                   y=coeffs[n]
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


def molecule_info(molec):
    """
    Get the mass fraction for each atom in a molecule.
    =============
    --VARIABLES--
    molec:          String containing chemical formula. Must be its simplest
                    form without parentheses, i.e. C8H18 not CH3(CH2)6CH3.
    """
    import re

    # Excel workbook to read atom mass from
    xlsx = '{0}/nist_xcom1.xlsx'.format(fld)

    # Get molecule formula
    ele_list = re.findall(
                          r'[A-Z][a-z]*|\d+',
                          re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', molec)
                          )
    atoms = ele_list[::2]
    atom_count = np.array([int(x) for x in ele_list[1::2]])
    atom_mass = np.zeros(len(atoms),)

    for i, atom in enumerate(atoms):
        atom_mass[i] = pd.read_excel(
                                     xlsx,
                                     sheet_name=atom,
                                     header=None,
                                     nrows=1,
                                     engine='openpyxl'
                                     )[1].to_numpy()
        atom_mass[i] *= atom_count[i]

    # Get total molecule mass
    molecule_mass = atom_mass.sum()
    # Calculate mass fraction for each atom
    atom_frac = atom_mass / molecule_mass

    return atoms, atom_count, atom_frac


def test_plots():
    """Tests out the NIST attenuation coefficient functions above."""
    import matplotlib.pyplot as plt

    # Calculated mass attenuation coefficient
    ki50_xcom1 = mass_atten(['H2O', 'KI'], [0.5, 0.5], xcom=1, col=7)

    # Calculated mass energy-absorption attenuation coefficient
    ki50_xcom2 = mass_atten(['H2O', 'KI'], [0.5, 0.5], xcom=2, col=2)

    # Attenuation coefficient taken directly from the NIST mixture web portal
    # See: <https://physics.nist.gov/cgi-bin/Xcom/xcom2>
    ki50_web = pd.read_csv('{0}/nist_50percKI_H2O.txt'.format(fld), sep='\t')

    # Drop values greater than the given keV cut-off
    ki50_web.drop(ki50_web[ki50_web.Energy > 200/1000].index, inplace=True)

    plt.figure()
    plt.plot(
             ki50_xcom1['Energy'],
             ki50_xcom1['Attenuation'],
             color='k',
             linestyle='solid',
             linewidth=3.0,
             label='Calc. $\mu$'
             )
    plt.plot(
             ki50_web['Energy']*1000,
             ki50_web['Tot wo Coh'],
             color='b',
             linestyle='dashed',
             linewidth=2.0,
             label='Web $\mu$'
             )
    plt.plot(
             ki50_xcom2['Energy'],
             ki50_xcom2['Attenuation'],
             color='g',
             linestyle='dotted',
             linewidth=2.0,
             label='$\mu_{en}$'
             )
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Attenuation Coefficient (cm$^2$/g)')
    plt.title('50% KI/H$_2$O $\mu$ Values')
    plt.savefig('{0}/mu_comparison.png'.format(fld))
    plt.close()

    return
