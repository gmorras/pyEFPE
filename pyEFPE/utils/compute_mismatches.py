import pyEFPE.waveform.EFPE as EFPE
import multiprocessing as mp
import pickle, os
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize_scalar, minimize
from scipy.fft import fft
from scipy.signal.windows import tukey
from pyEFPE.utils.constants import *

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
try:
    import lalsimulation as lalsim
    import lal
except:
    print('Warning, some functionality in compute_mismatches requires lalsuite to be installed.')
    pass


# Initialize global variables
generator = None  
params    = None

# Allows global generator to be passed to the workers
def init_worker(generator_instance, shared_params):
    global generator, params
    generator = generator_instance
    params    = shared_params
    warnings.filterwarnings("ignore", category=RuntimeWarning)

# dictionary of variable names as pyEFPE keys
pyEFPE_keys = {'mass1': 'm1', 'mass2': 'm2', 'e_start': 'ecc', 'mean_anomaly_start': 'mean_anomaly', 'spin1x': 's1x', 'spin1y': 's1y',
               'spin1z': 's1z', 'spin2x': 's2x', 'spin2y': 's2y', 'spin2z': 's2z', 'inclination': 'iota', 'phi_start': 'phiref', 'f22_end': 'f_max'}

# dictionary with CBC variables for evaluation
param_keys = ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y',
              's2z', 'iota', 'phiref', 'pol', 'f_max', 'ecc', 'mean_anomaly']

# function to compute the time to ISCO given an initial frequency and component masses (in solar masses)
def t_to_ISCO_0PN(f0, m1, m2):

    # compute the symmetric mass ratio and the total mass in solar masses
    M     = m1+m2
    nu    = m1*m2/(M*M)

    # convert total mass to seconds
    M     = t_sun_s*M

    # compute pi*M*f_ISCO
    piMff = 6**-1.5
    # compute pi*M*f0
    piMf0 = np.pi*M*f0

    # return the time to ISCO from Eq.(4.19) of Maggiore
    return (5/256)*(M/nu)*(piMf0**(-8/3) - piMff**(-8/3))

# function to compute the XPHM Minimum Energy Circular Orbit (MECO) frequency


def compute_fMECO(M, nu, spin_1z, spin_2z):

    # compute the Mf at the MECO with the lalsimulation function
    MfMECO = np.array([lalsim.SimIMRPhenomXfMECO(
        nu[i], spin_1z[i], spin_2z[i]) for i in range(len(nu))])

    # use the total mass to give the right units and return
    return MfMECO/(t_sun_s*M)

# function to get waveform in frequency domain from frequency domain waveform


def fd_from_lalsim_fd(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, waveform_dictionary, approximant):

    # compute the polarizations as lal frequency series
    hp_LAL, hc_LAL = lalsim.SimInspiralChooseFDWaveform(lal.lal.MSUN_SI*m1, lal.lal.MSUN_SI*m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc *
                                                        lal.lal.PC_SI*1.0e6, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, waveform_dictionary, approximant)

    # return the frequencies we want as numpy arrays
    return np.array(hp_LAL.data.data)[int(f_min/delta_f):int(f_max/delta_f)], np.array(hc_LAL.data.data)[int(f_min/delta_f):int(f_max/delta_f)]

# function to get waveform in frequency domain from time domain waveform


def fd_from_lalsim_td(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, waveform_dictionary, approximant, f_min_gen_fact=0.8, max_alpha=0.05, apply_tukey=True, df_factor=2, dt_factor=4, delta_t_max=1/4096):

    # compute the required delta_t
    delta_t = min(delta_t_max, 1/(f_max*max(1, dt_factor)))
    # make this a power of two
    delta_t = 2**np.floor(np.log2(delta_t))

    # compute the size of the array needed for the specified delta_f
    df_factor = max(1, int(df_factor))
    N_tsamples = int(df_factor/(delta_f*delta_t))

    # compute indexes of target frequencies
    i_start = int(df_factor*f_min/delta_f)
    i_end = int(df_factor*f_max/delta_f)

    # compute the minimum frequency to generate the waveform from
    f_min_gen = int(f_min*f_min_gen_fact/delta_f)*delta_f

    # compute lal waveform
    hp_LAL, hc_LAL = lalsim.SimInspiralChooseTDWaveform(lal.lal.MSUN_SI*m1, lal.lal.MSUN_SI*m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc *
                                                        lal.lal.PC_SI*1.0e6, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_t, f_min_gen, f_ref, waveform_dictionary, approximant)

    # loop over polarizations
    hs_fd = []
    for h_LAL in [hp_LAL, hc_LAL]:

        # compute arrays with data
        h = np.array(h_LAL.data.data)

        # if we generated too much stuff at the beginning, remove it
        if len(h) > N_tsamples:
            h = h[-N_tsamples:]

        # if required, apply Tukey window to reduce spectral leakage
        if apply_tukey:

            # compute the roll of from the f_min requirement
            roll_off = 2/f_min
            # compute the time from f_max to ISCO
            t_to_ISCO = t_to_ISCO_0PN(f_max, m1, m2)
            # We want the start of the roll_off at f_max
            n_extra = max(0, 1 + int((roll_off - t_to_ISCO)/delta_t))
            # compute the value of alpha to use in Tukey window
            alpha = min(max_alpha, 2*roll_off/(delta_t*(len(h) + n_extra)))
            # apply the tukey window, removing the samples after the length of h
            h *= tukey(len(h) + n_extra, alpha=alpha)[:len(h)]

        # if requiered, prepend with zeros to get the desired frequency resolution
        if len(h) < N_tsamples:
            h = np.append(np.zeros(N_tsamples - len(h)), h)

        # compute fast fourier transform at the required frequencies
        hs_fd.append(delta_t*fft(h)[i_start:i_end:df_factor])

    return hs_fd

# function to obtain ASD at certain frequencies from lal


def compute_asd(delta_f, f_min, f_max, name='aLIGOO3LowT1800545', asd_folder='./'):

    # first try to load asd from a file
    try:
        asd_raw = np.loadtxt(asd_folder + name +'.txt')

        # compute the frequencies
        freqs = np.arange(f_min, f_max, delta_f)

        # interpolated
        asd = np.interp(freqs, asd_raw[:, 0], asd_raw[:, 1])

    # otherwise, try computing it using lalsimulation
    except:
        # create lalseries, add one frequency point because in the last freq, psd gives 0
        lalseries = lal.CreateREAL8FrequencySeries('', lal.LIGOTimeGPS(
            0), 0, delta_f, lal.DimensionlessUnit, int(f_max/delta_f)+1)

        # guess the PSD function from name
        func = lalsim.__dict__['SimNoisePSD' + name]

        # put the PSD on lalseries
        func(lalseries, f_min)

        # select the frequencies above f_min and compute the ASD as the sqrt of the PSD
        asd = np.sqrt(np.array(lalseries.data.data)[int(f_min/delta_f):-1])

    # return the ASD
    return asd

# function to convert spherical coordinates into cartesian coordinates (inspired in the pyroq function)


def spherical_to_cartesian(sph):
    x = sph[0] * np.sin(sph[1]) * np.cos(sph[2])
    y = sph[0] * np.sin(sph[1]) * np.sin(sph[2])
    z = sph[0] * np.cos(sph[1])
    return x, y, z

# function to compute polarization response


def pol_response(hp, hc, pol):
    return np.cos(2*pol)*hp + np.sin(2*pol)*hc

# function to compute random parameters


def generate_rand_params(Nsamples, mc_low=1.4, mc_high=2.6, q_low=0.25, q_high=1, s1_low=None, s1_high=1, s2_low=None, s2_high=1, ecc_low=0, ecc_high=0, precessing=True):

    # consider precessing and non precessing cases to set s1_low and s2_low
    if precessing:
        s1_low, s2_low = 0, 0
    else:
        s1_low, s2_low = -s1_high, -s2_high

    # generate random input parameters
    params_low = [mc_low, q_low, s1_low, s2_low, ecc_low, -1, 0, 0, 0]
    params_high = [mc_high, q_high, s1_high, s2_high,
                   ecc_high, 1, 2*np.pi, 2*np.pi, np.pi]
    mc, q, s1, s2, ecc, cos_iota, phiref, mean_anomaly, pol = np.transpose(
        np.random.uniform(params_low, params_high, size=(Nsamples, len(params_low))))

    # if q is larger than 1, flip it to make it smaller
    q = np.where(q > 1, 1/q, q)

    # compute m1, m2 from mc, q
    m1 = mc*(q**-0.6)*((1 + q)**0.2)
    m2 = m1*q

    # compute inclination
    iota = np.arccos(cos_iota)

    # In the precessing case, generate perpendicular spins
    if precessing:
        # generate spin tilt angles
        ths1, ths2 = np.arccos(np.random.uniform(-1, 1, size=(2, Nsamples)))
        # generate spin phases
        phs1, phs2 = np.random.uniform(0, 2*np.pi, size=(2, Nsamples))
        # compute component spins
        s1x, s1y, s1z = spherical_to_cartesian([s1, ths1, phs1])
        s2x, s2y, s2z = spherical_to_cartesian([s2, ths2, phs2])
    # in the non-precessing case, set perpendicular spins to 0
    else:
        s1x, s1y, s2x, s2y = np.zeros_like(s1), np.zeros_like(
            s1), np.zeros_like(s2), np.zeros_like(s2)
        s1z, s2z = s1, s2

    # if the system is not precessing, set mean anomaly to 0
    if ecc_high == 0:
        mean_anomaly *= 0

    # compute derived mass parameters
    M = m1 + m2
    nu = m1*m2/(M*M)

    # compute ISCO frequency
    f_ISCO = 1/((6**1.5)*np.pi*t_sun_s*M)

    # compute MECO frequency
    f_MECO = compute_fMECO(M, nu, s1z, s2z)

    # force the MECO to be below the ISCO
    f_MECO = np.minimum(f_ISCO, f_MECO)

    # compute the effective inspiral spin parameter (see arXiv:0909.2867)
    chi_eff = (s1z + q*s2z)/(1 + q)

    # compute the precessing spin parameter (see arXiv:1408.1810)
    s1p = np.sqrt(s1x*s1x + s1y*s1y)
    s2p = np.sqrt(s2x*s2x + s2y*s2y)
    chi_p = np.maximum(s1p, (q*(4*q + 3)/(4 + 3*q))*s2p)

    # make a dictionary with all the samples
    return {'m1': m1, 'm2': m2, 's1x': s1x, 's1y': s1y, 's1z': s1z, 's2x': s2x, 's2y': s2y, 's2z': s2z, 'iota': iota, 'phiref': phiref, 'pol': pol, 'f_ISCO': f_ISCO, 'f_MECO': f_MECO, 'mc': mc, 'q': q, 'chi_eff': chi_eff, 'chi_p': chi_p, 'ecc': ecc, 'mean_anomaly': mean_anomaly}

# function to normalize waveforms and weigh them by asd=sqrt(psd)


def normalize_h(h, asd=None):

    # normalize by psd
    if asd is not None:
        h = h/asd

    # compute the norm of h
    h_norm = np.linalg.norm(h)

    return h/h_norm, h_norm

# function to compute the fft of an overlap using padding


def fft_overlap(overlap, padding_fact=4):

    # compute the new len(overlap) to be a power of 2
    new_len = 2**int(np.ceil(np.log2(padding_fact*len(overlap))))

    # return the fft of the padded overlap
    return fft(np.append(overlap, np.zeros(new_len - len(overlap))))

# function to compute the mismatch without any minimizations


def mismatch_amp_minimized(h1, h2, asd=None):

    # normalize the waveforms
    h1n, h1_norm = normalize_h(h1, asd=asd)
    h2n, h2_norm = normalize_h(h2, asd=asd)

    # return the mismatch
    return 1 - np.real(np.sum(np.conj(h1n)*h2n))

# function to minimize mismatch from initial guess from fft mismatch


def minimize_mm_dt(mm_fft, mm_func_dt, freqs, t_tol=1e-9):

    # find the index of the time that alignes them
    i_min = np.argmin(mm_fft)
    min_mm = mm_fft[i_min]

    # compute the time shift from the FFT, finding smallest difference
    if i_min > 0.5*len(mm_fft):
        di = i_min - len(mm_fft)
    else:
        di = i_min

    # time resotion of fourier transform
    df = freqs[1] - freqs[0]
    dT = 1/(df*len(mm_fft))

    # initial time shift guess from fft
    dt_shift = di*dT

    # try to minimize this function
    try:
        # use a bracketed search algorithm
        minimize_result = minimize_scalar(mm_func_dt, bracket=(
            dt_shift-dT, dt_shift, dt_shift+dT), tol=t_tol)
        if not minimize_result.success:
            print(minimize_result.message)
        # only replace min_mm if the result of the minimization is truly smaller
        if minimize_result.fun < min_mm:
            min_mm = minimize_result.fun
            dt_shift = minimize_result.x
    except Exception as excep:
        print('minimize_scalar failed with exception:', excep)
        print('Using fft minimization of mismatch: min_mm: %.3g' % (min_mm))

    return min_mm, dt_shift

# function to compute the mismatch minimized over time, phase and amplitude


def mismatch_t_ph_amp_minimized(h1, h2, freqs, t_tol=1e-9, asd=None, fft_padding_fact=4):

    # normalize the waveforms
    h1n, h1_norm = normalize_h(h1, asd=asd)
    h2n, h2_norm = normalize_h(h2, asd=asd)

    # compute the overlap
    overlap = np.conj(h1n)*h2n

    # compute their mismatch at all possible times
    mm_fft = 1 - np.abs(fft_overlap(overlap, padding_fact=fft_padding_fact))

    # function to compute the mismatch at any time
    def mm_func_dt(dt): return 1 - \
        np.abs(np.sum(np.exp(-2j*np.pi*freqs*dt)*overlap))

    # now minimize the mismatch
    min_mm, dt_shift = minimize_mm_dt(mm_fft, mm_func_dt, freqs, t_tol=t_tol)

    # compute the relative phases of the waveforms
    ang_h2mh1 = np.angle(np.sum(np.exp(-2j*np.pi*freqs*dt_shift)*overlap))

    # return maximized mismatch, time shift, and relative phase and amplitude
    return min_mm, dt_shift, ang_h2mh1, h2_norm/h1_norm

# function to compute compute mismatch in terms of the rhos from Eq.(27) of 1603.02444


def pol_minimized_mismatch_rhos(rho_p, rho_c, Ipc, return_optimum_u=False):

    # compute gamma from Eq.(22)
    gamma = np.real(rho_p*np.conj(rho_c))

    # compute coefficients of quadratic Eq.(26)
    rho_p2 = np.abs(rho_p)**2
    rho_c2 = np.abs(rho_c)**2
    A = Ipc*rho_p2 - gamma
    B = rho_p2-rho_c2
    C = gamma - Ipc*rho_c2

    # use it to compute it Eq.(27)
    sqrt_part = np.sqrt(B*B - 4*A*C)
    num = rho_p2 - 2*Ipc*gamma + rho_c2 + sqrt_part
    den = 1 - Ipc**2

    # use that the mismatch is mm = 1 - sqrt(2*lambda)
    mm = 1 - np.sqrt(num/(2*den))

    # if required, return optimum polarization solving Eq.(26)
    if return_optimum_u:
        # use that kappa=arctan(1/u) and that u_{-} is the one that gives Eq.(27)
        unum = -sqrt_part - B
        uden = 2*A
        # return mismatch and numerator/denominator of u (to avoid divisions)
        return mm, unum, uden

    # return only the mismatch
    else:
        return mm

# function to compute the mismatch minimized over time, phase, amplitude and polarization (following 1603.02444)


def mismatch_t_ph_amp_pol_minimized(signal, hp, hc, freqs, t_tol=1e-9, asd=None, fft_padding_fact=4, return_optimum_params=False):

    # normalize the waveforms
    sn, s_norm = normalize_h(signal, asd=asd)
    hpn, hp_norm = normalize_h(hp, asd=asd)
    hcn, hc_norm = normalize_h(hc, asd=asd)

    # compute the overlaps
    overlap_sp = np.conj(sn)*hpn
    overlap_sc = np.conj(sn)*hcn

    # compute I_{+x} from Eqs.(23), since it does not depend on time
    Ipc = np.real(np.sum(np.conj(hpn)*hcn))

    # compute \rho_+ and \rho_x as functions of time using fft
    rho_p = fft_overlap(overlap_sp, padding_fact=fft_padding_fact)
    rho_c = fft_overlap(overlap_sc, padding_fact=fft_padding_fact)

    # compute the mismatch as a function of time with these rhos
    mm_fft = pol_minimized_mismatch_rhos(rho_p, rho_c, Ipc)

    # function to compute the mismatch as a function of time
    def rho_p_func_dt(dt): return np.sum(np.exp(-2j*np.pi*freqs*dt)*overlap_sp)
    def rho_c_func_dt(dt): return np.sum(np.exp(-2j*np.pi*freqs*dt)*overlap_sc)
    def mm_func_dt(dt): return pol_minimized_mismatch_rhos(
        rho_p_func_dt(dt), rho_c_func_dt(dt), Ipc)

    # now minimize the mismatch
    min_mm, dt_shift = minimize_mm_dt(mm_fft, mm_func_dt, freqs, t_tol=t_tol)

    # if required, return parameters that have been analytically minimized
    if return_optimum_params:

        # compute rho_p and rho_c for optimum time shift
        rho_p_opt, rho_c_opt = rho_p_func_dt(dt_shift), rho_c_func_dt(dt_shift)

        # compute optimum polarization
        mm, unum, uden = pol_minimized_mismatch_rhos(
            rho_p_opt, rho_c_opt, Ipc, return_optimum_u=True)

        # compute polarization from Eq.(24), where kappa = 2*pol = arctan((hp_norm/hc_norm)/u)
        pol = 0.5*np.arctan2(uden*hp_norm, unum*hc_norm)

        # compute the sqrt(<h|h>) for this polarization
        fact_p, fact_c = hp_norm*np.cos(2*pol), hc_norm*np.sin(2*pol)
        h_norm = np.sqrt(fact_p**2 + fact_c**2 + 2*fact_p*fact_c*Ipc)

        # compute angle(<s|h>) for this polarization
        ang_sh = np.angle(fact_p*rho_p_opt + fact_c*rho_c_opt)

        # return the parameters we have maximized over
        return min_mm, dt_shift, ang_sh, h_norm/s_norm, pol
    else:
        # return only the mismatch
        return min_mm

# function to compute 2d rotation matrix


def rotation_matrix_2D(phi):
    sph, cph = np.sin(phi), np.cos(phi)
    return np.array([[cph, -sph], [sph, cph]])

# function to rotate spins


def rotate_spins(phases, s1p, s2p):

    # compute rotation matrix for spin vector 1
    R1 = rotation_matrix_2D(phases[1])

    # if two arguments are given, rotate spins solidarily
    if len(phases) == 2:
        R2 = R1
    # if three arguments are given, rotate spins independently
    elif len(phases) == 3:
        R2 = rotation_matrix_2D(phases[2])
    else:
        raise Exception('Expected 2 or three arguments in local_mismatch_func')

    # return rotated spins
    return np.dot(R1, s1p), np.dot(R2, s2p)

# function to minimize the mismatch over phase and angle of the spins


def minimize_precessing_mismatch(signal, waveform, freqs, parameters, asd=None, Nph_tries=20, rotate_individual_spins=False, additional_minimize_threshold=1e-2, disp=False):

    # make a local dictionary to update
    p = parameters.copy()

    # define function to compute waveform in terms of these phases
    def local_mismatch_func(phases, return_optimum_params=False):

        # compute the Lal waveform
        hp, hc = waveform(p, phases=phases)

        # compute the mismatch maximized over polarization
        return mismatch_t_ph_amp_pol_minimized(signal, hp, hc, freqs, return_optimum_params=return_optimum_params, asd=asd)

    # generate random phases to try
    if rotate_individual_spins:
        Nphases = 3
    else:
        Nphases = 2
    phases_tries = np.random.uniform(0, 2*np.pi, size=(Nph_tries, Nphases))

    # generate mismatches for these ranom phases
    mismatches_tries = np.array(
        [local_mismatch_func(phases_tries[i]) for i in range(Nph_tries)])

    # find phases that minimize mismatch
    i_min = np.argmin(mismatches_tries)
    min_mm = mismatches_tries[i_min]
    min_phases = phases_tries[i_min]

    # use scipy minimize to minimize further
    if min_mm > additional_minimize_threshold:
        if disp:
            print('\nMismatch of initial guess: %.4g' % (min_mm))
        minimize_result = minimize(local_mismatch_func, phases_tries[i_min], method='Nelder-Mead', options={
                                   'disp': disp, 'fatol': additional_minimize_threshold})
        # extract final result and the value of the optimum phases
        if minimize_result.fun < min_mm:
            min_mm = minimize_result.fun
            min_phases = minimize_result.x

    # compute the parameters that have been analytically maximized
    min_mm_2, dt_shift, ang_sh, amp_sh, pol_h, = local_mismatch_func(
        min_phases, return_optimum_params=True)

    # compute the mismatch for each and minimize
    return min_mm, dt_shift, ang_sh, amp_sh, pol_h, *min_phases


# function to create string ID from relevant parameters
def generate_string_ID(Nsamples, precessing, approximant, psd_name, mc_low, mc_high, q_low, q_high, s1_high, s2_high, flim_type, pn_spin_order, pn_phase_order, params_pyEFPE, ecc_high=0, rotate_individual_spins=False, fmax_flim=1, string_start='_'):

    # if the system is precessing, add a prefix to indicate it
    string_ID = string_start
    if precessing:
        string_ID += 'prec_'

    # if the system is eccentric, add a prefix to indicate it
    if ecc_high > 0:
        string_ID += 'ecc_%.3g_' % (ecc_high)

    # generate generic string ID, with approximant and parameter space used
    string_ID += approximant+'_N_%s_mc_%.3g_%.3g_q_%.3g_%.3g_s1_%.3g_s2_%.3g_fmax_%.3g%s_PN_spin_%s_phase_%s' % (
        Nsamples, mc_low, mc_high, q_low, q_high, s1_high, s2_high, fmax_flim, flim_type, pn_spin_order, pn_phase_order)

    # add it to string different kwargs in params_pyEFPE
    if 'SUA_kmax' in params_pyEFPE.keys():
        string_ID += '_SUA_%s' % (params_pyEFPE['SUA_kmax'])
    if 'Amplitude_tol' in params_pyEFPE.keys():
        string_ID += '_Atol_%.3g' % (params_pyEFPE['Amplitude_tol'])

    # if the system is precessing, add wheather the spins are individually rotated
    if precessing:
        string_ID += '_ROTs1s2_%s' % (rotate_individual_spins)

    # if a PSD is given, add it also to the string ID
    if type(psd_name) == str:
        string_ID += '_'+psd_name

    return string_ID

# function to compute mismatches


def random_mismatches_with_EFPE(
    Nsamples, approx_string, seglen=128, f_min=20, flim_type='ISCO', fmax_flim=1, psd_name=None,
    distance_Mpc=10, mc_low=1.4, mc_high=2.6, q_low=1, q_high=4, s1_low=None, s1_high=1, s2_low=None, s2_high=1,
    ecc_high=0, precessing=True, params_pyEFPE={}, pn_spin_order=None, pn_phase_order=None, pn_amplitude_order=None,
    Nph_tries=20, rotate_individual_spins=False, additional_minimize_threshold=1e-2, outdir='./outdir', nworkers=-1, print_progress=True, asd_folder='./'
):

    # complete params_pyEFPE with things that dont change
    if 'f22_start' not in params_pyEFPE:
        params_pyEFPE['f22_start'] = f_min-1
    if 'pn_spin_order' not in params_pyEFPE:
        params_pyEFPE['pn_spin_order'] = pn_spin_order
    if 'pn_phase_order' not in params_pyEFPE:
        params_pyEFPE['pn_phase_order'] = pn_phase_order

    # overwrite distance even if given in params_pyEFPE
    params_pyEFPE['distance'] = distance_Mpc

    # compute random parameters
    params = generate_rand_params(Nsamples, mc_low=mc_low, mc_high=mc_high, q_low=q_low, q_high=q_high,
                                  s1_low=s1_low, s1_high=s1_high, s2_low=s2_low, s2_high=s2_high, ecc_high=ecc_high, precessing=precessing)

    # compute delta_f from seglen
    delta_f = 1/seglen

    # add to params the necessary things to reconstruct results
    params.update({'approx_string': approx_string, 'psd_name': psd_name, 'dL': distance_Mpc, 'seglen': seglen, 'f_min': f_min, 'flim_type': flim_type, 'fmax_flim': fmax_flim, 'delta_f': delta_f,
                  'precessing': precessing, 'params_pyEFPE': params_pyEFPE, 'pn_spin_order': pn_spin_order, 'pn_phase_order': pn_phase_order, 'pn_amplitude_order': pn_amplitude_order, })

    # compute f_max to be a fraction of the ISCO
    params['f_max'] = np.floor(np.maximum(
        fmax_flim*params['f_'+flim_type], f_min + 10*delta_f)*seglen)*delta_f

    # generate a list with dictionaries of parameters needed for evaluation
    params['param_keys'] = param_keys
    param_dicts = [{key: params[key][i]
                    for key in params['param_keys']} for i in range(Nsamples)]

    # if we want to print progress, number each sample
    if print_progress:
        params['print_progress'] = True
        for ip in range(Nsamples):
            param_dicts[ip]['ip'] = ip

    params['Nsamples']                       = Nsamples
    params['Nph_tries']                      = Nph_tries
    params['rotate_individual_spins']        = rotate_individual_spins
    params['additional_minimize_threshold']  = additional_minimize_threshold

    # initialize the waveform generators
    generator = WaveformGenerator(
        f_min, delta_f, params_pyEFPE, approx_string,
        pn_spin_order, pn_phase_order, pn_amplitude_order
    )

    # compute the asd up to a maximum frequency
    if type(psd_name) == str:
        params['freqs_asd'] = np.arange(
            f_min, np.amax(params['f_max']) + delta_f, delta_f)
        params['asd'] = compute_asd(delta_f, f_min, np.amax(
            params['f_max']) + delta_f, name=psd_name, asd_folder=asd_folder)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Compute mismatches in parallel with progress bar
        with mp.Pool(nworkers, initializer=init_worker, initargs=(generator, params)) as pool:
            # Use `imap_unordered` to process items and update tqdm
            output_mismatches = []
            with tqdm(total=len(param_dicts), desc="Processing mismatches") as pbar:
                for result in pool.imap_unordered(mismatch_with_EFPE, param_dicts):
                    output_mismatches.append(result)
                    pbar.update()

    # Convert this output to a numpy_array
    output_mismatches = np.transpose(output_mismatches)
    params['mismatches'] = output_mismatches[0]
    params['dt_shift_sh'] = output_mismatches[1]
    params['ang_sh'] = output_mismatches[2]
    params['amp_sh'] = output_mismatches[3]
    params['pol_h'] = output_mismatches[4]

    if precessing:
        params['phases_h'] = np.transpose(output_mismatches[5:])

    # Create a dictionary with pyEFPE inputs
    all_params_pyEFPE = []
    for ip, p in enumerate(param_dicts):
        all_params_pyEFPE.append(params_pyEFPE.copy())
        for pyEFPE_key, LAL_key in pyEFPE_keys.items():
            all_params_pyEFPE[ip][pyEFPE_key] = p[LAL_key]

    params['all_params_pyEFPE'] = all_params_pyEFPE

    # if the directory to dump params does not exist, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # create an identifiable name for the file
    string_id = generate_string_ID(Nsamples, precessing, approx_string, psd_name, mc_low, mc_high, q_low, q_high, s1_high, s2_high, flim_type,
                                    pn_spin_order, pn_phase_order, params_pyEFPE, rotate_individual_spins=rotate_individual_spins, fmax_flim=fmax_flim, ecc_high=ecc_high)

    # dump it
    with open(outdir+'/params'+string_id+'.pickle', 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return params

# function to plot waveforms from params


def compare_waveforms_from_params(params, plot_order='maximum mismatch'):

    # initialize the waveform generators
    approx_LAL, approx_pyEFPE = initialize_pyEFPE_and_LAL_approxs(
        params['f_min'], params['delta_f'], params['params_pyEFPE'], params['approx_string'], params['pn_spin_order'], params['pn_phase_order'], params['pn_amplitude_order'])

    # generate a list with dictionaries of parameters needed for evaluation
    param_dicts = [{key: params[key][i] for key in params['param_keys']}
                   for i in range(len(params['mismatches']))]

    # if required, sort mismatches from largest to smaller
    if plot_order == 'maximum mismatch':
        iplot = np.argsort(-params['mismatches'])
    elif plot_order == 'minimum mismatch':
        iplot = np.argsort(params['mismatches'])
    else:
        iplot = np.arange(len(params['mismatches']))

    # now loop over parameters
    for ip in iplot:

        # parameters to use
        p = param_dicts[ip]

        # compute pyEFPE waveform, frequencies and the dictionary we have called pyEFPE with
        h_pyEFPE, freqs = approx_pyEFPE(p)

        # compute the psd
        if type(params['psd_name']) == str:
            asd = np.interp(freqs, params['freqs_asd'], params['asd'])
        else:
            asd = None

        # compute the LAL waveform
        if params['precessing']:
            hp_LAL, hc_LAL = approx_LAL(p, phases=params['phases_h'][ip])
        else:
            hp_LAL, hc_LAL = approx_LAL(p)

        h_LAL = pol_response(hp_LAL, hc_LAL, params['pol_h'][ip])

        # adjust the amplitude, phase and reference time of LAL waveform
        h_LAL = (h_LAL/params['amp_sh'][ip])*np.exp(-1j*(2*np.pi *
                                                         freqs*params['dt_shift_sh'][ip] + params['ang_sh'][ip]))

        # compute mismatch expected from adjusted waveform
        hn_pyEFPE, h_norm_pyEFPE = normalize_h(h_pyEFPE, asd=asd)
        hn_LAL, h_norm_LAL = normalize_h(h_LAL, asd=asd)
        mm_direct = 1 - np.real(np.sum(np.conj(hn_pyEFPE)*hn_LAL))

        # compute title string
        title_str = r'$\mathcal{M}_c = %.3g M_\odot$, $q=%.2f$, $\chi_\mathrm{eff}=%.2g$, $\chi_\mathrm{p}=%.2g$ $\rightarrow$ MM: %.3g' % (
            params['mc'][ip], params['q'][ip], params['chi_eff'][ip], params['chi_p'][ip], params['mismatches'][ip])

        # print informations
        print('\nmc=%.3g, q=%.2f, chi_eff=%.2g, chi_p=%.2g -> MM: %.3g  MM_direct: %.3g' %
              (params['mc'][ip], params['q'][ip], params['chi_eff'][ip], params['chi_p'][ip], params['mismatches'][ip], mm_direct))

        # make a plot of h_pyEFPE vs h_LAL
        fig = plot_h1_h2(h_pyEFPE, h_LAL, freqs, asd=asd, label_1=r'$h_\mathrm{EFPE}$', label_2=r'$h_\mathrm{%s}$' % (
            params['approx_string']), title_str=title_str)

# function to plot frequency domain waveforms


def plot_h1_h2(h1, h2, freqs, asd=None, label_1=r'$h_1$', label_2=r'$h_2$', show=True, figsize=(12, 12), title_str=None):

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=figsize)

    # plot amplitudes
    abs_h1, abs_h2 = np.abs(h1), np.abs(h2)
    axs[0].plot(freqs, abs_h1, label=r'$|\tilde{h}_1|$')
    axs[0].plot(freqs, abs_h2, label=r'$|\tilde{h}_2|$')
    if asd is not None:
        axs[0].plot(freqs, asd, label=r'ASD', color='k')
    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'$|\tilde{h}|$ [Hz]')
    axs[0].legend()

    # plot proxy of phases
    norm_prod = np.conj(h1)*h2/(abs_h1*abs_h2)
    axs[1].plot(freqs, np.unwrap(np.angle(norm_prod), period=2*np.pi) /
                (2*np.pi), label=r'$\frac{\varphi(h_2) - \varphi(h_1)}{2 \pi}$')
    axs[1].legend()

    # set x-axis related stuff
    axs[1].set_xlim(freqs[0], freqs[-1])
    axs[1].set_xlabel(r'$f$ [Hz]')
    axs[1].set_xscale('log')

    # add title
    if title_str is not None:
        title_str += '\n'
    else:
        title_str = ''
    title_str += r'%s $\rightarrow$ $h_1$,  %s $\rightarrow$ $h_2$' % (
        label_1, label_2)
    fig.suptitle(title_str, fontsize=22)

    # remove spaces between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # show plot if required
    if show:
        plt.show()


def mismatch_with_EFPE(p):
    """
    """
    global generator

    # compute pyEFPE waveform, frequencies and the dictionary we have called pyEFPE with
    h_pyEFPE, freqs = generator.approx_pyEFPE(p)

    if type(params['psd_name']) == str:
        asd = np.interp(freqs, params['freqs_asd'], params['asd'])
    else:
        asd = None

    # compute the minimum mismatches and the parameters for which it is minimized for each case
    if params['precessing']:
        output_mismatch = minimize_precessing_mismatch(
            h_pyEFPE, generator.approx_LAL, freqs, p, asd=asd, Nph_tries=params['Nph_tries'],
            rotate_individual_spins=params['rotate_individual_spins'], additional_minimize_threshold=params['additional_minimize_threshold']
        )
    else:
        hp_LAL, hc_LAL = generator.approx_LAL(p)
        output_mismatch = mismatch_t_ph_amp_pol_minimized(
            h_pyEFPE, hp_LAL, hc_LAL, freqs, asd=asd, return_optimum_params=True
        )

    # if required print information
    if params['print_progress']:
        print('%s/%s -> mc=%.3g, q=%.2f, chi_eff=%.2g, chi_p=%.2g -> MM: %.3g' % (
            p['ip'] + 1, params['Nsamples'], params['mc'][p['ip']
                                                          ], params['q'][p['ip']],
            params['chi_eff'][p['ip']
                              ], params['chi_p'][p['ip']], output_mismatch[0]
        ))

    # return the output of mismatch calculation
    return output_mismatch


class WaveformGenerator:
    def __init__(self, f_min, delta_f, params_pyEFPE, approx_string, pn_spin_order, pn_phase_order, pn_amplitude_order):
        self.f_min = f_min
        self.delta_f = delta_f
        self.params_pyEFPE = params_pyEFPE
        self.approx_func = None

        # Store configuration for creating the lal.Dict
        self.pn_spin_order = pn_spin_order
        self.pn_phase_order = pn_phase_order
        self.pn_amplitude_order = pn_amplitude_order

        # Determine approximant and function type
        self.approx_lalsim = lalsim.SimInspiralGetApproximantFromString(
            approx_string)
        if approx_string in ['SpinTaylorT4', 'SEOBNRv4P', 'IMRPhenomT', 'IMRPhenomTP']:
            self.approx_func = fd_from_lalsim_td
        else:
            self.approx_func = fd_from_lalsim_fd

    def create_waveform_dictionary(self):
        # Create and configure a new lal.Dict
        waveform_dictionary = lal.CreateDict()
        if self.pn_spin_order is not None:
            lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(
                waveform_dictionary, int(self.pn_spin_order))
        if self.pn_phase_order is not None:
            lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(
                waveform_dictionary, int(self.pn_phase_order))
        if self.pn_amplitude_order is not None:
            lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(
                waveform_dictionary, int(self.pn_amplitude_order))
        return waveform_dictionary

    def approx_LAL(self, params, phases=None):
        # Create a new waveform dictionary for this call
        waveform_dictionary = self.create_waveform_dictionary()

        # Make a local copy of params
        p = params.copy()

        # Rotate spins and change reference phase if required
        if phases is not None:
            s1p = np.array([p['s1x'], p['s1y']])
            s2p = np.array([p['s2x'], p['s2y']])
            s1p_rot, s2p_rot = rotate_spins(phases, s1p, s2p)
            p['s1x'], p['s1y'] = s1p_rot
            p['s2x'], p['s2y'] = s2p_rot
            p['phiref'] = phases[0]

        # Return polarizations from LAL approximant
        return self.approx_func(
            p['m1'], p['m2'], p['s1x'], p['s1y'], p['s1z'],
            p['s2x'], p['s2y'], p['s2z'], self.params_pyEFPE['distance'],
            p['iota'], p['phiref'], 0, p['ecc'], p['mean_anomaly'],
            self.delta_f, self.f_min, p['f_max'],
            self.params_pyEFPE['f22_start'], waveform_dictionary, self.approx_lalsim
        )

    def approx_pyEFPE(self, params):
        # Create pyEFPE params dictionary
        p_pyEFPE = self.params_pyEFPE.copy()
        for pyEFPE_key, LAL_key in pyEFPE_keys.items():
            p_pyEFPE[pyEFPE_key] = params[LAL_key]

        # Compute frequency array
        freqs = np.arange(self.f_min, params['f_max'], self.delta_f)

        # Initialize and generate pyEFPE waveform
        wf = EFPE.pyEFPE(p_pyEFPE)
        hp_pyEFPE, hc_pyEFPE = wf.generate_waveform(freqs)
        h_pyEFPE = pol_response(hp_pyEFPE, hc_pyEFPE, params['pol'])

        return h_pyEFPE, freqs
