""" Plugins for external packages to access the pyEFPE waveform model(s)
"""

import numpy as np

# No imports should be placed here outside those required for the pyEFPE 
# package itself to avoid increasing any install requirements for the 
# model.

def pycbc_plugin(**parameters):
    """ Interface for the PyCBC package
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE
    
    from pycbc.types import FrequencySeries
    from pycbc.pnutils import f_SchwarzISCO
    
    # Parameter name conversions, all others passed directly
    # This helps keep a consistent naming convention set for downstream 
    # Where possible, sticking to 
    # https://github.com/gwastro/pycbc/blob/master/pycbc/waveform/parameters.py
    #
    # Note that the native names may be provided in most cases in lieue of
    # these, except in the case of 'f_lower'.
    conversions = {'f_lower': 'f22_start',
                   'eccentricity': 'e_start',
                   'dquad_mon1': 'q1',
                   'dquad_mon2': 'q2',
                   'f_final': 'f22_end',
                   'coa_phase': 'phi_start',
                   'anomaly': 'mean_anomaly_start',
                   }

    renamed_params = {
        new_name: parameters[old_name] 
        for old_name, new_name in conversions.items() 
        if (old_name in parameters) and (parameters[old_name] is not None)
    }
    parameters.update(renamed_params)
    wf = EFPE.pyEFPE(parameters)
    M = parameters['mass1'] + parameters['mass2']
    freqs = np.arange(parameters['f_lower'],
                      f_SchwarzISCO(M),
                      parameters['delta_f']
                      )

    hp, hc = wf.generate_waveform(freqs)

    tref = - 1.0 / parameters['delta_f']
    hp = FrequencySeries(hp, epoch=tref, delta_f=parameters['delta_f'])
    hc = FrequencySeries(hp, epoch=tref, delta_f=parameters['delta_f'])
    return hp, hc
    
