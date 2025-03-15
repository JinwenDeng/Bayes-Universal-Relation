# First, import some useful packages.
import astropy.units as units
from astropy.cosmology import z_at_value, LambdaCDM
import numpy as np
from scipy.interpolate import interp1d
import pycbc.detector
from astropy.utils import iers
iers.conf.auto_download = False
from pycbc.waveform import get_fd_waveform_sequence
from pycbc.waveform import Array
from copy import deepcopy
import os
import mpmath as mp
from astropy import constants as const
from astropy import units as u

# # import luminosity distance and redshift relation
# from dL_z_relation import redshift_from_luminosity_distance_interp, luminosity_distance_from_redshift_interp

G=const.G.cgs.value
c=const.c.cgs.value
Ms=const.M_sun.cgs.value
hbar=const.hbar.cgs.value
m_n=const.m_n.cgs.value
km=10**5
yr=(1.0*u.yr).cgs.value
pi = np.pi

# For GW analysis, the first thing is to generate the strain in detectors when the parameters are given.
# This notebook is mainly for the target. Many other functions and classes have been defined in other files, especially in 'waveform_GR.py'.

# In principle, the GW strain is independent detector. However, in practice, we need to consider the detector response function.
# Besides, considering the calculation efficiency, the sampling points are depend on the detector's frequency band.

current_path = os.path.dirname(__file__)
# Dictionary for the detector name and the corresponding PSD file.
PSD_datapath = {'AdvLIGO_H1': 'PSDdata/AdvLIGO.txt', # The LIGO PSD data is consistent with the pycbc PSD aLIGOAPlusDesignSensitivityT1800042
                'AdvLIGO_L1': 'PSDdata/AdvLIGO.txt', 
                'AdvVirgo': 'PSDdata/AdvVirgo.txt', # Vigro is consistent with the pycbc PSD AdvVirgo from eq6 of 1202.4031
                'CE2_H1': 'PSDdata/CE-2.txt',
                'CE2_L1': 'PSDdata/CE-2.txt',
                'ETD_E1': 'PSDdata/ET-D.txt',
                'ETD_E2': 'PSDdata/ET-D.txt',
                'ETD_E3': 'PSDdata/ET-D.txt', }
PSD_datapath = {key: os.path.join(current_path, PSD_datapath[key]) for key in PSD_datapath.keys()}
pycbc_name = {'AdvLIGO_H1': 'H1',
              'AdvLIGO_L1': 'L1',
              'AdvVirgo': 'V1',
              'CE2_H1': 'H1',
              'CE2_L1': 'L1',
              'ETD_E1': 'E1',
              'ETD_E2': 'E2',
              'ETD_E3': 'E3', }
form_factor = {'AdvLIGO_H1': 1,
               'AdvLIGO_L1': 1,
               'AdvVirgo': 1,
               'CE2_H1': 1,
               'CE2_L1': 1,
               'ETD_E1': (3**0.5)/2,
               'ETD_E2': (3**0.5)/2,
               'ETD_E3': (3**0.5)/2, }

# Define a class for arranging the detector information.


class single_PSD:  # (very like the Curve class in 'curve.py')
    def __init__(self, det_name):
        self.det_name = det_name
        if det_name not in PSD_datapath.keys():
            raise Exception('The detector name is not in the list.')
        self.PSD_datapath = PSD_datapath[det_name]
        self.frequencies, self.PSD = np.loadtxt(self.PSD_datapath).T
        self.freqMax = self.frequencies[-1]
        self.freqMin = self.frequencies[0]
        self.interp_lgPSD_lgf = interp1d(np.log10(self.frequencies), np.log10(
            self.PSD), bounds_error=False, fill_value=np.inf)

    def __call__(self, f):
        """ beyond the range of frequency, this program will return np.inf
        """
        lg_f = np.log10(f)
        Res_PSD = 10 ** np.asarray(self.interp_lgPSD_lgf(lg_f))
        return Res_PSD


# Define a class for obtaining GW strain, storing the used detector-network information.
class GWtool:  # (very like the Fisher class in 'waveform_GR.py')
    def __init__(self, detectors, f_lower=0, f_upper=np.infty, f_num_factor=100, ref_time=1126259462.0):
        if isinstance(detectors, str):
            detectors = [detectors]
        CurveLibs = set(PSD_datapath.keys())
        if set(detectors) <= CurveLibs:
            self.detectors = detectors
        else:
            raise Exception('Wrong Detector (Network) Input! Plz Check Again')
        self.detector_handler = {
            det: single_PSD(det) for det in self.detectors}

        f_low_tmp = np.inf
        for det in self.detectors:
            f_low_tmp = min(f_low_tmp, self.detector_handler[det].freqMin)
        self.f_lower = max(f_lower, f_low_tmp)
        f_high_tmp = - np.inf
        for det in self.detectors:
            f_high_tmp = max(f_high_tmp, self.detector_handler[det].freqMax)
        self.f_upper = min(f_upper, f_high_tmp)
        if self.f_upper <= self.f_lower:
            raise Exception("The frequency range is improper!")
        self.frequencies = np.logspace(np.log10(self.f_lower), np.log10(self.f_upper),
                                       int(f_num_factor*100*(np.log10(self.f_upper)-np.log10(self.f_lower))))

        self.ref_time = ref_time
        self.set_detectors(self.ref_time)
        self.set_PSD(self.frequencies)
        self.mpdps = None

        # Numerical differentiation precisions for the calculation of the derivatives of the GW strain.
        self.delta1_dic = {'mass1': 1e-7, 'mass2': 1e-7,
                'distance': 1e-9, 'delta_tc': 1e-8, 'B': 1e-10, 'spin1z': 1e-6,
                'spin2z': 1e-6,'beta':1e-8,
                'ra': 1e-6, 'dec': 1e-6,'pol': 1e-6, 'inclination': 1e-6,
                'chirp_mass': 10 ** (-7.5), 'eta': 10 ** (-8), 'q': 1e-7, 
                'lambda1': 1e-4, 'lambda2': 1e-4, 'Lambda_tilde': 1e-4, 'delta_Lambda_tilde': 1e-6,
                'quadrupole1': 1e-5, 'quadrupole2': 1e-5, 'q_tilde': 1e-7, 'delta_q_tilde': 1e-7,
                'phi_c': 1e-6
                }
        self.delta2_dic = {'mass1': 10 ** (-3.7), 'mass2': 10 ** (-3.7), 'chirp_mass': 10 ** (-6.8 - 0.2 - 0.5 + 3),
              'eta': 10 ** (-4 - 0.5 - 1.5 +0.5), 'q': 10 ** (-5.3 - 0.2), 'ra': 1e-6, 'dec': 1e-6,'distance': 10 ** (-3.5+2),
              'spin1z': 10 ** (-4.5),
              'spin2z': 10 ** (-4.5),
              'pol': 1e-4, 'inclination': 1e-4, 'comoving_volume': 10 ** (0.3 + 3), 'beta': 10 ** (-5),
              'delta_tc': 10 ** (-5), 'lambda1': 10 ** (-0.5), 'lambda2': 10 ** (-0.5)}
        self.delta3_dic = {'mass1': 10 ** (-3), 'mass2': 10 ** (-3.7), 'chirp_mass': 10 ** (-4),
              'eta': 10 ** (-5), 'q': 10 ** (-5.3 - 0.2), 'ra': 1e-4, 'dec': 1e-4,'distance': 10 ** (-1),
              'spin1z': 10 ** (-4.5),
              'spin2z': 10 ** (-4.5),
              'pol': 1e-3, 'inclination': 1e-3, 'comoving_volume': 10 ** (0.3 + 3), 'beta': 10 ** (-5),
              'delta_tc': 10 ** (-6), 'lambda1': 10 ** (-0.5), 'lambda2': 10 ** (-0.5)}
        self.masses_perturbation_functions = {
            'chirp_mass': lambda p, param, d: self.m1m2_mceta(p[param] + d, p['eta']),
            'eta': lambda p, param, d: self.m1m2_mceta(p['chirp_mass'], p[param] + d),
            'q': lambda p, param, d: self.m1m2_mcq(p['chirp_mass'], p[param] + d),
        }  # These special perturbations are set for the mass input format of the get_fd_waveform function in pycbc.waveform.
        self.other_perturbation_functions = {
            'beta' : ('distance',lambda p, param, d: self.dL_beta(p[param] + d)),}
    def get_fd_waveform_sequence(self, **params):
        # Wrapper for the pycbc.waveform.get_fd_waveform_sequence function.
        hp_py, hc_py = get_fd_waveform_sequence(sample_points = Array(self.frequencies),**params)
        return np.array(hp_py), np.array(hc_py)
    def get_antenna_pattern(self, det_name, ra, dec, pol):
        if det_name not in self.detectors:
            print("Detector not found in the network! Plz check if it is really needed.")
            det = pycbc.detector.Detector(det_name)
        else:
            det = self.pycbc_detectors[det_name]
        return det.antenna_pattern(ra, dec, pol, self.ref_time)
    def time_delay_from_earth_center(self, det_name, ra, dec):
        if det_name not in self.detectors:
            print("Detector not found in the network! Plz check if it is really needed.")
            det = pycbc.detector.Detector(det_name)
        else:
            det = self.pycbc_detectors[det_name]
        return det.time_delay_from_earth_center(ra, dec, self.ref_time)
    
    def Lambda1_Lambda2_from_Lambda_tilde_delta_Lambda_tilde(self, Lambda_tilde, delta_Lambda_tilde, mass1, mass2):
        delta_m = (mass1-mass2)/(mass1+mass2)
        eta = mass1*mass2/(mass1+mass2)**2
        a1, a2 = 24*(1+7*eta-31*eta**2), 24*(1+9*eta-11*eta**2)*delta_m
        b1, b2 = np.sqrt(1-4*eta)*(1-13272/1319*eta+8944/1319*eta**2), (1-15910/1319*eta+32850/1319*eta**2+3380/1319*eta**3)
        lambda_s = (b2*Lambda_tilde-a2*delta_Lambda_tilde)/(b2*a1-b1*a2)
        lambda_a = (b1*Lambda_tilde-a1*delta_Lambda_tilde)/(b1*a2-b2*a1)
        return (lambda_s+lambda_a), (lambda_s-lambda_a)

    def q_tilde_delta_q_tilde_from_q1q2(self, Q1, Q2, spin1z, spin2z, mass1, mass2):
        m_sum = mass1 + mass2
        q1 = (Q1-1)*(mass1*spin1z/m_sum)**2
        q2 = (Q2-1)*(mass2*spin2z/m_sum)**2
        q_tilde = 50*(q1+q2)
        delta_q_tilde = (5/84)*(9407+8218*mass1/m_sum-2016*(mass1/m_sum)**2)*q1+(5/84)*(9407+8218*mass2/m_sum-2016*(mass2/m_sum)**2)*q2
        return q_tilde, delta_q_tilde  # 3/128/x**2.5/params['eta']*(q_tilde*x**2+delta_q_tilde*x**3)

    def q1q2_from_q_tilde_delta_q_tilde(self, q_tilde, delta_q_tilde, spin1z, spin2z, mass1, mass2):
        m_sum = mass1 + mass2
        a = (5/84)*(9407+8218*mass1/m_sum-2016*(mass1/m_sum)**2)
        b = (5/84)*(9407+8218*mass2/m_sum-2016*(mass2/m_sum)**2)
        q1 = (b*q_tilde + delta_q_tilde*50)/(50*a-50*b)
        q2 = (a*q_tilde + delta_q_tilde*50)/(50*b-50*a)
        q1 = q1*(m_sum/mass1/spin1z)**2 + 1
        q2 = q2*(m_sum/mass2/spin2z)**2 + 1
        return q1, q2
    
    def get_strain(self, network=None, **params):
        if network is None:
            network = self.detectors
        
        params['coa_phase'] = params['phi_c']/2
        params['lambda1'], params['lambda2'] = self.Lambda1_Lambda2_from_Lambda_tilde_delta_Lambda_tilde(params['Lambda_tilde'], params['delta_Lambda_tilde'], params['mass1'], params['mass2'])
        params['quadrupole1'], params['quadrupole2'] = self.q1q2_from_q_tilde_delta_q_tilde(params['q_tilde'], params['delta_q_tilde'], params['spin1z'], params['spin2z'], params['mass1'], params['mass2'])
        params['dquad_mon1'], params['dquad_mon2'] = params['quadrupole1']-1, params['quadrupole2']-1
        # params['q_tilde'], params['delta_q_tilde'] = self.q_tilde_delta_q_tilde_from_q1q2(params['quadrupole1'], params['quadrupole2'], params['spin1z'], params['spin2z'], params['mass1'], params['mass2'])
        hp, hc = get_fd_waveform_sequence(sample_points = Array(self.frequencies),**params)
        strain = {}
        if set(network) > set(self.detectors):
            raise Exception('Detector not found in current network!')

        #phase = np.array([0.0 for f in self.frequencies])
        #x_list = [(pi*G*Ms*(params['mass1']+params['mass2'])*f/(c**3))**(2/3) for f in self.frequencies]
        #phase += np.array([-3/128/x**2.5/params['eta']*(params['Lambda_tilde']*x**5 + (3115/1248*params['Lambda_tilde']-6595/7098*np.sqrt(1-4*params['eta'])*params['delta_Lambda_tilde'])*x**6) for x in x_list])
        #phase += np.array([3/128/x**2.5/params['eta']*(params['q_tilde']*x**2 + params['delta_q_tilde']*x**3) for x in x_list])

        for det_name in network:
            # In this code, we don't consider the time-dependent antenna pattern functions. 
            # Therefore, the location and orientation of the detector are fixed at the reference time.
            # In this case, the merger time (Denoted as 'delta_tc') of the GW signal only affects the phase as a linear term in the frequency domain.
            det = self.pycbc_detectors[det_name]
            dt = det.time_delay_from_earth_center(params['ra'], params['dec'], self.ref_time) # Time delay from the earth center
            fp, fc = det.antenna_pattern(params['ra'], params['dec'], params['pol'], self.ref_time) # Antenna pattern functions
            # The projected GW strain signal 
            h = fp * hp + fc * hc
            # Phase correction due to the merger time
            h*=np.exp(-2j*np.pi*self.frequencies*(params['delta_tc']+dt))
            #h*=np.exp(1j*phase)
            strain[det_name] = h
        return strain

    def network_inner_product(self, A, B, network = None):
        if network is None:
            network = self.detectors
        if type(A) is dict and type(B) is dict:
            if not set(A.keys()) <= set(network):
                raise Exception("Detectors in strain not found in PSD data! Strain: %s, PSD: %s" % (A.keys(), network))
            if set(A.keys()) != set(B.keys()):
                raise Exception("Different detectors in strains!")
            inner_product = 0
            for det in network:
                inner_product+=self.single_inner_product(A[det],B[det],self.PSDs[det])
            return inner_product
        else:
            raise Exception("Wrong input format! Plz input the strain as a dictionary.")
    
    def single_inner_product(self,A,B,PSD):
        if len(A) != len(PSD):
           raise Exception("Different dimensions of A and Sn")
        elif B is None:
            return 4 * np.real(np.trapz(self.frequencies * np.conjugate(A)*A/PSD, 
                                    x=np.log(self.frequencies)))
        elif len(A) != len(B):
            raise Exception("Different dimensions of A, B")
        else:
            return 4 * np.real(np.trapz(self.frequencies * np.conjugate(A)*B/PSD, 
                                    x=np.log(self.frequencies)))
    def single_det_inner_product(self,A,B,det):
        if det not in self.detectors:
            raise Exception("Detector not found in the network!")
        PSD = self.PSDs[det]
        return self.single_inner_product(A,B,PSD)
        
    def gradient1_network(self, v_param, delta=None,network = None,**params):
        if network is None:
            network = self.detectors
        delta = delta if delta is not None else self.delta1_dic[v_param]
        params_p, params_m = self.params_perturbation(v_param, delta,network,**params)
        h_p = self.get_strain(network,**params_p)
        h_m = self.get_strain(network,**params_m)
        h_gradient1 = {}
        for det in network:
            h_gradient1[det] = (h_p[det]-h_m[det])/(2*delta)
        return h_gradient1

    def gradient2_network(self, v_param1, v_param2, delta1=None, delta2=None,network = None,**params):
        if network is None:
            network = self.detectors
        delta1 = delta1 if delta1 is not None else self.delta2_dic[v_param1]
        delta2 = delta2 if delta2 is not None else self.delta2_dic[v_param2]
        params_p, params_m = self.params_perturbation(v_param1, delta1,network,**params)
        params_pp, params_pm = self.params_perturbation(v_param2, delta2,network,**params_p)
        params_mp, params_mm = self.params_perturbation(v_param2, delta2,network,**params_m)
        h_pp = self.get_strain(network,**params_pp)
        h_pm = self.get_strain(network,**params_pm)
        h_mp = self.get_strain(network,**params_mp)
        h_mm = self.get_strain(network,**params_mm)
        h_gradient2 = {}
        for det in network:
            h_gradient2[det] = ((h_pp[det] - h_pm[det]) - (h_mp[det] - h_mm[det])) / ((2 * delta1) * (2 * delta2))
        return h_gradient2

    def gradient3_network(self, v_param1, v_param2, v_param3, delta1=None, delta2=None, delta3=None,network = None,**params):
        if network is None:
            network = self.detectors
        delta1 = delta1 if delta1 is not None else self.delta3_dic[v_param1]
        delta2 = delta2 if delta2 is not None else self.delta3_dic[v_param2]
        delta3 = delta3 if delta3 is not None else self.delta3_dic[v_param3]
        params_p, params_m = self.params_perturbation(v_param1, delta1,network,**params)
        params_pp, params_pm = self.params_perturbation(v_param2, delta2,network,**params_p)
        params_mp, params_mm = self.params_perturbation(v_param2, delta2,network,**params_m)
        params_ppp, params_ppm = self.params_perturbation(v_param3, delta3,network,**params_pp)
        params_pmp, params_pmm = self.params_perturbation(v_param3, delta3,network,**params_pm)
        params_mpp, params_mpm = self.params_perturbation(v_param3, delta3,network,**params_mp)
        params_mmp, params_mmm = self.params_perturbation(v_param3, delta3,network,**params_mm)
        h_ppp = self.get_strain(network,**params_ppp)
        h_ppm = self.get_strain(network,**params_ppm)
        h_pmp = self.get_strain(network,**params_pmp)
        h_pmm = self.get_strain(network,**params_pmm)
        h_mpp = self.get_strain(network,**params_mpp)
        h_mpm = self.get_strain(network,**params_mpm)
        h_mmp = self.get_strain(network,**params_mmp)
        h_mmm = self.get_strain(network,**params_mmm)
        h_gradient3 = {}
        for det in network:
            h_gradient3[det] = (((h_ppp[det] - h_ppm[det]) - (h_pmp[det] - h_pmm[det])) - (
                    (h_mpp[det] - h_mpm[det]) - (h_mmp[det] - h_mmm[det]))) / ((2 * delta1) * (2 * delta2) * (2 * delta3))
        return h_gradient3


    def fisher_matrix(self, v_param_lis,network = None,delta1_dic = None,
                      use_mp = False, mpdps = 50, warning = True,
                      **params): #Refactoring the 'fisher' function
        if network is None:
            network = self.detectors
        if delta1_dic is None:
            delta1_dic = self.delta1_dic
        strain_grad1 = {alpha:self.gradient1_network(alpha, delta1_dic[alpha],network, **params) for alpha in v_param_lis}
        F_dic = {}
        F_matrix = [
            [
                F_dic.setdefault(tuple(sorted([alpha, beta])),
                self.network_inner_product(strain_grad1[alpha],strain_grad1[beta],network))
                for beta in v_param_lis
            ]
            for alpha in v_param_lis
        ]
        F_matrix = np.array(F_matrix)
        if use_mp:
            if self.mpdps is None:
                self.set_mpdps(mpdps)
            FM_mp = mp.matrix(F_matrix)
            CM_mp = mp.inverse(FM_mp)
            max_diff = np.max(np.abs(np.array((FM_mp @ CM_mp - mp.eye(FM_mp.rows)).tolist(),dtype=float)))
            if max_diff > 1e-8 and warning:
                print(f"WARNING: The precision of the matrix inversion may be not enough!\nThe maximum difference between M*M^(-1) and I is {max_diff:.2e}.")
            CM = np.array(CM_mp.tolist(),dtype=np.float64)
        else:
            FM_cond = np.linalg.cond(F_matrix)
            if FM_cond > 1e16 and warning:
                print(f"WARNING: The condition number of the Fisher matrix is too large!\nThe matrix inversion may be inaccurate. The condition number is {FM_cond:.2e}.")
            CM_ = np.linalg.inv(F_matrix)
            max_diff = np.max(np.abs(F_matrix @ CM_ - np.eye(F_matrix.shape[0])))
            if max_diff > 1e-8 and warning:
                print(f"WARNING: The precision of the matrix inversion is not enough!\nThe maximum difference between M*M^(-1) and I is {max_diff:.2e}.")
            CM = CM_
        return F_matrix, CM
    def set_mpdps(self,mpdps):
        mp.mp.dps = mpdps

    def normalize_covariance_matrix(self,CovM):
        sigma = np.sqrt(np.diag(CovM))
        CorrM = CovM / np.outer(sigma, sigma)
        return sigma, CorrM
        

    def doublet_DALI(self, v_param_lis, network=None, delta1_dic=None, delta2_dic=None, **params):
        if network is None:
            network = self.detectors
        if delta1_dic is None:
            delta1_dic = self.delta1_dic
        if delta2_dic is None:
            delta2_dic = self.delta2_dic
        strain_grad1 = {alpha: self.gradient1_network(alpha, delta1_dic[alpha], network, **params) for alpha in
                        v_param_lis}
        strain_grad2_dic = {}
        strain_grad2 = {
            alpha: {
                beta: strain_grad2_dic.setdefault(tuple(sorted([alpha, beta])),
                                                  self.gradient2_network(alpha, beta, delta2_dic[alpha],
                                                                         delta2_dic[beta], network, **params))
                for beta in v_param_lis
            }
            for alpha in v_param_lis
        }
        doublet_DALIa_dic = {}
        doublet_DALIa = [
            [
                [
                    doublet_DALIa_dic.setdefault(tuple(sorted([tuple(sorted([alpha, beta])), tuple(sorted([gamma]))])),
                                                  self.network_inner_product(strain_grad2[alpha][beta],
                                                                              strain_grad1[gamma], network))
                    for gamma in v_param_lis
                ]
                for beta in v_param_lis
            ]
            for alpha in v_param_lis
        ]
        doublet_DALIb_dic = {}
        doublet_DALIb = [
            [
                [
                    [
                        doublet_DALIb_dic.setdefault(
                            tuple(sorted([tuple(sorted([alpha, beta])), tuple(sorted([gamma, delta]))])),
                            self.network_inner_product(strain_grad2[alpha][beta], strain_grad2[gamma][delta], network))
                        for delta in v_param_lis
                    ]
                    for gamma in v_param_lis
                ]
                for beta in v_param_lis
            ]
            for alpha in v_param_lis
        ]
        return np.array(doublet_DALIa), np.array(doublet_DALIb)

    def triplet_DALI(self, v_param_lis, network=None, delta1_dic=None, delta2_dic=None, delta3_dic=None, **params):
        if network is None:
            network = self.detectors
        if delta1_dic is None:
            delta1_dic = self.delta1_dic
        if delta2_dic is None:
            delta2_dic = self.delta2_dic
        if delta3_dic is None:
            delta3_dic = self.delta3_dic
        strain_grad1 = {alpha: self.gradient1_network(alpha, delta1_dic[alpha], network, **params) for alpha in
                        v_param_lis}
        strain_grad2_dic = {}
        strain_grad2 = {
            alpha: {
                beta: strain_grad2_dic.setdefault(tuple(sorted([alpha, beta])),
                                                  self.gradient2_network(alpha, beta, delta2_dic[alpha],
                                                                         delta2_dic[beta], network, **params))
                for beta in v_param_lis
            }
            for alpha in v_param_lis
        }
        strain_grad3_dic = {}
        strain_grad3 = {
            alpha: {
                beta: {
                    gamma: strain_grad3_dic.setdefault(tuple(sorted([alpha, beta, gamma])),
                                                         self.gradient3_network(alpha, beta, gamma, delta3_dic[alpha],
                                                                                delta3_dic[beta], delta3_dic[gamma],
                                                                                network, **params))
                    for gamma in v_param_lis
                }
                for beta in v_param_lis
            }
            for alpha in v_param_lis
        }

        triplet_DALIa_dic = {}
        triplet_DALIa = [
            [
                [
                    [
                        triplet_DALIa_dic.setdefault(
                            tuple(sorted([tuple(sorted([alpha])), tuple(sorted([beta, gamma, delta]))])),
                            self.network_inner_product(strain_grad1[alpha], strain_grad3[beta][gamma][delta], network))
                        for delta in v_param_lis
                    ]
                    for gamma in v_param_lis
                ]
                for beta in v_param_lis
            ]
            for alpha in v_param_lis
        ]

        triplet_DALIb_dic = {}
        triplet_DALIb = [
            [
                [
                    [
                        [
                            triplet_DALIb_dic.setdefault(
                                tuple(sorted([tuple(sorted([alpha, beta, gamma])), tuple(sorted([delta, tau]))])),
                                self.network_inner_product(strain_grad3[alpha][beta][gamma],
                                                           strain_grad2[delta][tau], network))
                            for tau in v_param_lis
                        ]
                        for delta in v_param_lis
                    ]
                    for gamma in v_param_lis
                ]
                for beta in v_param_lis
            ]
            for alpha in v_param_lis
        ]

        triplet_DALIc_dic = {}
        triplet_DALIc = [
            [
                [
                    [
                        [
                            [
                                triplet_DALIc_dic.setdefault(
                                    tuple(sorted([tuple(sorted([alpha, beta, gamma])), tuple(sorted([delta, tau, sigma]))])),
                                    self.network_inner_product(strain_grad3[alpha][beta][gamma],
                                                               strain_grad3[delta][tau][sigma], network))
                                for sigma in v_param_lis
                            ]
                            for tau in v_param_lis
                        ]
                        for delta in v_param_lis
                    ]
                    for gamma in v_param_lis
                ]
                for beta in v_param_lis
            ]
            for alpha in v_param_lis
        ]
        return np.array(triplet_DALIa), np.array(triplet_DALIb), np.array(triplet_DALIc)

    def params_perturbation(self, v_param, delta,network = None,**params):
        if network is None:
            network = self.detectors
        if v_param not in params.keys():
            raise Exception("Parameter not found!")
        params_p, params_m = deepcopy(params), deepcopy(params)
        if v_param in self.masses_perturbation_functions.keys():
            params_p['mass1'], params_p['mass2'] = self.masses_perturbation_functions[v_param](params, v_param,  delta)
            params_m['mass1'], params_m['mass2'] = self.masses_perturbation_functions[v_param](params, v_param, -delta)
        if v_param in self.other_perturbation_functions.keys():
            params_p[self.other_perturbation_functions[v_param][0]] = self.other_perturbation_functions[v_param][1](params, v_param,  delta)
            params_m[self.other_perturbation_functions[v_param][0]] = self.other_perturbation_functions[v_param][1](params, v_param, -delta)
        params_p[v_param] = params_p[v_param] + delta
        params_m[v_param] = params_m[v_param] - delta
        return params_p, params_m

    def get_network_SNR(self, network = None,**params):
        if network is None:
            network = self.detectors
        strain = self.get_strain(network,**params)
        return np.sqrt(self.network_inner_product(strain,strain,network))

    def get_aa_SNR(self, **params):
        h_aa = self.get_aa_strain(**params)
        return np.sqrt(self.aa_inner_product(h_aa,h_aa))

    def get_aa_strain(self,**params):
        # Average the inner product over the sky location, polarization angle, and inclination angle.
        # When doing the angular averaging, the location of the detectors and source are not important.
        params_aa = deepcopy(params)
        params_aa['inclination'] = 0
        hp,hc = get_fd_waveform_sequence(sample_points = Array(self.frequencies),**params_aa)
        h_aa = hp*0.4*np.exp(-2j*np.pi*self.frequencies*(params['delta_tc']))
        return h_aa
    
    def aa_inner_product(self,A,B):
        if len(A) != len(self.aa_PSD):
            raise Exception("Different dimensions of strain and angle-averaged PSD!")
        if len(A) != len(B):
            raise Exception("Different lengths of strains!")
        else:
            return 4 * np.real(np.trapz(self.frequencies * np.conjugate(A)*B/self.aa_PSD, 
                                    x=np.log(self.frequencies)))

    def set_frequencies(self, f_lis):
        self.frequencies = f_lis
        self.set_PSD(self.frequencies)

    def set_PSD(self, f_lis):
        self.PSDs = {det: self.detector_handler[det](
            f_lis) for det in self.detectors}
        sum_tmp = np.zeros_like(f_lis)
        for det in self.detectors:
            sum_tmp += form_factor[det]**2/self.PSDs[det]
        self.aa_PSD = 1/sum_tmp

    def set_detectors(self, ref_time):
        if ref_time is None:
            raise Exception("The reference time is not set!")
        # reference_time is the GPS time of the first detection of GW150914.
        # It is used mainly for the calculation of gmst (Greenwich mean sidereal time).
        # Therefore, the injected merger time t_c of the GW signal should be close to this time for accurate calculation.
        self.pycbc_detectors = {det: pycbc.detector.Detector(pycbc_name[det],
                                reference_time=ref_time)
                                for det in self.detectors}

    def get_detectors(self):
        return self.detectors

    # Calculate the component masses from chirp mass and symmetric mass ratio.
    def m1m2_mceta(self, chirp_mass, eta):
        if eta > 0.25:
            raise ValueError(
                "Symmetric mass ratio is too large. Should be <= 0.25.")
        total_mass = chirp_mass * eta ** (-0.6)
        delta = np.sqrt(1 - 4 * eta)
        mass1 = (1 + delta) * total_mass / 2
        mass2 = (1 - delta) * total_mass / 2
        return mass1, mass2

    # Calculate the component masses from chirp mass and mass ratio.
    def m1m2_mcq(self, chirp_mass, mass_ratio):
        if mass_ratio > 1:
            raise ValueError("Mass ratio is too large. Should be <= 1.")
        mass1 = chirp_mass * (mass_ratio / (1 + mass_ratio)) ** 0.2
        mass2 = chirp_mass * (1 / (1 + mass_ratio)) ** 0.2
        return mass1, mass2
    
    def dL_beta(self,beta):
        return 1/beta

    def set_deltas(self, delta1_dic=None, delta2_dic=None, delta3_dic=None):
        if delta1_dic is not None:
            self.delta1_dic = delta1_dic
        if delta2_dic is not None:
            self.delta2_dic = delta2_dic
        if delta3_dic is not None:
            self.delta3_dic = delta3_dic

    def get_deltas(self, order=1):
        if order == 1:
            return self.delta1_dic
        elif order == 2:
            return self.delta2_dic
        elif order == 3:
            return self.delta3_dic
        else:
            raise Exception("Don't have the order %d derivatives!" % order)
