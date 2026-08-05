""" Plugins for external packages to access the pyEFPE waveform model(s)
"""

import numpy as np

# No imports should be placed here outside those required for the pyEFPE 
# package itself to avoid increasing any install requirements for the 
# model.

def rename_parameters(parameters):
    # Parameter name conversions, all others passed directly
    # This helps keep a consistent naming convention set for downstream 
    # Where possible, sticking to 
    # https://github.com/gwastro/pycbc/blob/master/pycbc/waveform/parameters.py
    #
    # Note that the native names may be provided in most cases in lieue of
    # these, except in the case of 'f_lower'.
    #
    # 'f_final' is absent on purpose: it is the ceiling of the frequency array,
    # not 'f22_end', and is used by pycbc_fd_plugin instead.
    # 'coa_phase' -> 'phi_start' identifies the coalescence phase with the phase
    # at f22_start, since pyEFPE has no coalescence-phase parameterisation.
    # 'anomaly' comes last since the comprehension keeps the last match, and
    # PyCBC injects mean_per_ano=0.0 into every call.
    conversions = {'f_lower': 'f22_start',
                   'eccentricity': 'e_start',
                   'coa_phase': 'phi_start',
                   'mean_per_ano': 'mean_anomaly_start',
                   'anomaly': 'mean_anomaly_start',
                   }

    renamed_params = {
        new_name: parameters[old_name]
        for old_name, new_name in conversions.items()
        if (old_name in parameters) and (parameters[old_name] is not None)
    }
    parameters.update(renamed_params)

    # LAL's dQuadMon is the quadrupole parameter minus one, so 0 for a black hole
    for old_name, new_name in (('dquad_mon1', 'q1'), ('dquad_mon2', 'q2')):
        if parameters.get(old_name) is not None:
            parameters[new_name] = 1 + parameters[old_name]

def pycbc_fd_plugin(**parameters):
    """ Interface for the PyCBC package
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE

    from pycbc.types import FrequencySeries
    from pycbc.pnutils import f_SchwarzISCO

    rename_parameters(parameters)
    wf = EFPE.pyEFPE(parameters)
    delta_f = parameters['delta_f']
    delta_t = parameters.get('delta_t')

    # Ceiling of the returned array. PyCBC's f_final defaults to 0, leaving the
    # choice to the approximant: fall back to Nyquist, then to the ISCO.
    if parameters.get('f_final'):
        f_max = parameters['f_final']
    elif delta_t:
        f_max = 0.5/delta_t
    else:
        f_max = f_SchwarzISCO(parameters['mass1'] + parameters['mass2'])

    # A FrequencySeries places sample k at k*delta_f, and pyEFPE returns zero
    # where the waveform has no support, so no padding is needed
    kmax = int(f_max/delta_f) + 1
    freqs = np.arange(kmax)*delta_f

    hp, hc = wf.generate_waveform(freqs)

    epoch = wf.return_start_time()
    hp = FrequencySeries(hp, epoch=epoch, delta_f=delta_f)
    hc = FrequencySeries(hc, epoch=epoch, delta_f=delta_f)
    return hp, hc
    

def pycbc_td_plugin(**parameters):
    """ Interface for the PyCBC package
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE

    from pycbc.types import TimeSeries

    rename_parameters(parameters)
    wf = EFPE.pyEFPE(parameters)

    hp, hc = wf.generate_tdomain_waveform(delta_t=parameters["delta_t"])

    epoch = wf.return_start_time()
    hp = TimeSeries(hp, epoch=epoch, delta_t=parameters['delta_t'])
    hc = TimeSeries(hc, epoch=epoch, delta_t=parameters['delta_t'])
    return hp, hc
