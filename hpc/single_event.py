import os
os.environ['PYCBC_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from numpy import sin as sin
from numpy import cos as cos
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as sp_interp1d
import bilby
import corner
from copy import deepcopy
import multiprocessing
multiprocessing.set_start_method('fork')
from bilby.core.prior import DirichletPriorDict, PriorDict
import pandas as pd
from astropy import constants as const
from astropy import units as u
from nessai.flowsampler import FlowSampler
from nessai_bilby.model import BilbyModel
from nessai.utils import setup_logger
import argparse
from basic_code_pycbc_waveform import GWtool
from generate_priors import calculate_bounds, calculate_priors
from dL_z_relation import D_c_to_z

# customize the plot 
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.labelpad'] = 8.0
plt.rcParams['figure.constrained_layout.h_pad'] = 0
# plt.rcParams['text.usetex'] = False
# plt.rc('text', usetex=False)
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.tick_params(axis='both', which='minor', labelsize=18)
import matplotlib.ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# introduce constants that I use
G=const.G.cgs.value
c=const.c.cgs.value
Ms=const.M_sun.cgs.value
hbar=const.hbar.cgs.value
m_n=const.m_n.cgs.value
km=10**5
yr=(1.0*u.yr).cgs.value
plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--nlive", type=int, default=1500)
parser.add_argument('--dlogz', type=float, default=0.1)
parser.add_argument('--npool', type=int, default=16)
parser.add_argument('--log_dir', type=str, default='')
args = parser.parse_args()
print('The command line arguments are:\n', args)

pi = np.pi

global apx
apx = 'IMRPhenomXAS_NRTidalv3'

eos_name = 'AP4'
script_dir = os.path.dirname(os.path.abspath(__file__))
I_Love_Q_path = os.path.join(script_dir, "I_Love_Q", eos_name+'.txt')  

def Lambda_from_mass(m):
    loaded_data = np.loadtxt(I_Love_Q_path, delimiter='\t')
    m_list = loaded_data[:, 0]
    lg_Lambda_list = loaded_data[:, 2]
    Lambda_interp = sp_interp1d(m_list, lg_Lambda_list)
    return 10**(Lambda_interp(m))

def Q_from_mass(m):
    loaded_data = np.loadtxt(I_Love_Q_path, delimiter='\t')
    m_list = loaded_data[:, 0]
    lg_Q_list = loaded_data[:, 3]
    Q_interp = sp_interp1d(m_list, lg_Q_list)
    return 10**(Q_interp(m))

class GW_likelihood(bilby.Likelihood):
    def __init__(self, GWtool_instance, data, param_names):
        if not isinstance(GWtool_instance, GWtool):
            raise TypeError("GWtool_instance must be an instance of GWtool")
        self.GWtool_instance = GWtool_instance
        if set(GWtool_instance.detectors) != set(data.keys()):
            raise ValueError("Detectors in GWtool_instance and data must be the same")
        super().__init__(parameters=dict.fromkeys(param_names))
        self.param_names = param_names
        self.parameters = dict.fromkeys(param_names)
        self.data=data
        self.detector = GWtool_instance.detectors


    def log_likelihood(self):
        params1 = deepcopy(self.parameters)
        params1['mass1'],params1['mass2'] = m1m2_from_mceta(self.parameters['chirp_mass'],self.parameters['eta'])
        params1['lambda1'],params1['lambda2'] = Lambda1_Lambda2_from_Lambda_tilde_delta_Lambda_tilde(self.parameters['Lambda_tilde'], self.parameters['delta_Lambda_tilde'], params1['mass1'], params1['mass2'])
        params1['quadrupole1'], params1['quadrupole2'] = q1q2_from_q_tilde_delta_q_tilde(self.parameters['q_tilde'], self.parameters['delta_q_tilde'], self.parameters['spin1z'], self.parameters['spin2z'], params1['mass1'], params1['mass2'])
        params1['dquad_mon1'], params1['dquad_mon2'] = params1['quadrupole1'] - 1, params1['quadrupole2'] - 1
        if params1['dquad_mon1'] < 0 or params1['dquad_mon2'] < 0:
            return -np.inf
        if params1['lambda1'] < 0 or params1['lambda2'] < 0:
            return -np.inf
        params1['approximant'] = apx
        params1['coa_phase'] = params1['phi_c']/2 # Orbital phase is half the signal phase
        model = self.GWtool_instance.get_strain(**params1)
        noise = {}
        for det in self.detector:
            noise[det] = self.data[det] - model[det]
        return -0.5*self.GWtool_instance.network_inner_product(noise,noise)

def m1m2_from_mceta(chirp_mass, eta):  # 给定啾质量和对称质量比求双星质量
        M = chirp_mass * (eta ** (-0.6))
        delta = np.sqrt(1 - 4 * eta)
        return (1 + delta) * M / 2, (1 - delta) * M / 2

def q_tilde_delta_q_tilde_from_q1q2(Q1, Q2, spin1z, spin2z, mass1, mass2):
    m_sum = mass1 + mass2
    q1 = (Q1-1)*(mass1*spin1z/m_sum)**2
    q2 = (Q2-1)*(mass2*spin2z/m_sum)**2
    q_tilde = -50*(q1+q2)
    delta_q_tilde = (5/84)*(9407+8218*mass1/m_sum-2016*(mass1/m_sum)**2)*q1+(5/84)*(9407+8218*mass2/m_sum-2016*(mass2/m_sum)**2)*q2
    return q_tilde, delta_q_tilde  # 3/128/x**2.5/params['eta']*(q_tilde*x**2+delta_q_tilde*x**3)

def q1q2_from_q_tilde_delta_q_tilde(q_tilde, delta_q_tilde, spin1z, spin2z, mass1, mass2):
    m_sum = mass1 + mass2
    a = (5/84)*(9407+8218*mass1/m_sum-2016*(mass1/m_sum)**2)
    b = (5/84)*(9407+8218*mass2/m_sum-2016*(mass2/m_sum)**2)
    q1 = (b*q_tilde + delta_q_tilde*50)/(50*a-50*b)
    q2 = (a*q_tilde + delta_q_tilde*50)/(50*b-50*a)
    q1 = q1*(m_sum/mass1/spin1z)**2 + 1
    q2 = q2*(m_sum/mass2/spin2z)**2 + 1
    return q1, q2

def Lambda_tilde_delta_Lambda_tilde_from_Lambda1_Lambda2(Lambda1, Lambda2, mass1, mass2):
    lambda_s = 0.5*(Lambda1+Lambda2)
    lambda_a = 0.5*(Lambda1-Lambda2)
    delta_m = (mass1-mass2)/(mass1+mass2)
    eta = mass1*mass2/(mass1+mass2)**2
    Lambda_tilde = 24*((1+7*eta-31*eta**2)*lambda_s + (1+9*eta-11*eta**2)*lambda_a*delta_m)
    delta_Lambda_tilde = np.sqrt(1-4*eta)*(1-13272/1319*eta+8944/1319*eta**2)*lambda_s + (1-15910/1319*eta+32850/1319*eta**2+3380/1319*eta**3)*lambda_a
    return Lambda_tilde, delta_Lambda_tilde # -3/128/x**2.5/params['eta']*(Lambda_tilde*x**5 + (3115/1248*Lambda_tilde-6595/7098*np.sqrt(1-4*params['eta'])*delta_Lambda_tilde)*x**6)

def Lambda1_Lambda2_from_Lambda_tilde_delta_Lambda_tilde(Lambda_tilde, delta_Lambda_tilde, mass1, mass2):
    delta_m = (mass1-mass2)/(mass1+mass2)
    eta = mass1*mass2/(mass1+mass2)**2
    a1, a2 = 24*(1+7*eta-31*eta**2), 24*(1+9*eta-11*eta**2)*delta_m
    b1, b2 = delta_m*(1-13272/1319*eta+8944/1319*eta**2), (1-15910/1319*eta+32850/1319*eta**2+3380/1319*eta**3)
    lambda_s = (b2*Lambda_tilde-a2*delta_Lambda_tilde)/(b2*a1-b1*a2)
    lambda_a = (b1*Lambda_tilde-a1*delta_Lambda_tilde)/(b1*a2-b2*a1)
    return (lambda_s+lambda_a), (lambda_s-lambda_a)

def mceta_from_m1m2(m1, m2):
    chirp_mass = (m1*m2)**0.6/(m1+m2)**0.2
    eta = (m1*m2)/(m1+m2)**2
    return chirp_mass, eta

network = ['CE2_H1', 'CE2_L1', 'ETD_E1', 'ETD_E2', 'ETD_E3']
GWtool_instance = GWtool(network)

# Events generator
size = 100
mu1, sigma1, weight1 = 1.34, 0.02, 0.68 # 第一个高斯分布（均值、标准差、权重）
mu2, sigma2, weight2 = 1.47, 0.15, 0.32   # 第二个高斯分布
choices = np.random.choice([0, 1], size=size, p=[weight1, weight2])
mass1_samples = np.where(choices == 0, 
                   np.random.normal(mu1, sigma1, size), 
                   np.random.normal(mu2, sigma2, size))
mass2_samples = np.random.uniform(low=1.14, high=1.46, size=size)
spin1z_samples, spin2z_samples = np.random.uniform(low=-0.1, high=0.1, size=size), np.random.uniform(low=-0.1, high=0.1, size=size)
inclination_samples = np.arccos(1-2*np.random.uniform(low=0, high=1, size=size))
ra_samples = np.random.uniform(low=0, high=2*pi, size=size)
dec_samples = pi/2 - np.arccos(1-2*np.random.uniform(low=0, high=1, size=size))
pol_samples = np.random.uniform(low=0, high=pi, size=size)
phi_c_samples = np.random.uniform(low=0, high=2*pi, size=size)

R_max, R_min = 150, 50  # Mpc
u = np.random.rand(size)
D_c_samples = ((R_max**3 - R_min**3) * u + R_min**3)**(1/3) 
z_samples = np.array([D_c_to_z(Dc) for Dc in D_c_samples])
distance_samples = np.array([(1+z)*D_c for z, D_c in zip(z_samples, D_c_samples)])

snr_list = []
event_data_list = []
true_params_list = []
global m_min 
global m_max
m_min= 0.25
m_max = 2.00

for i in range(size):
    # Injection parameters and GW strain
    if mass1_samples[i] < mass2_samples[i]:
        mass1_samples[i], mass2_samples[i] = mass2_samples[i], mass1_samples[i]
        spin1z_samples[i], spin2z_samples[i] = spin2z_samples[i], spin1z_samples[i]
        
    true_params = {'mass1': mass1_samples[i], 'mass2': mass2_samples[i],
                   'distance': distance_samples[i], 'delta_tc': 0.0, 
                   'approximant': apx, 
                   'ra': ra_samples[i], 'dec': dec_samples[i], 'pol': pol_samples[i], 'inclination': inclination_samples[i], 
                   'phi_c': phi_c_samples[i], 'coa_phase': phi_c_samples[i]/2, 
                   'spin1z': spin1z_samples[i], 'spin2z': spin2z_samples[i], 
                   }
    true_params['chirp_mass'], true_params['eta'] = mceta_from_m1m2(true_params['mass1'], true_params['mass2'])
    true_params['lambda1'], true_params['lambda2'] = Lambda_from_mass(true_params['mass1']), Lambda_from_mass(true_params['mass2'])
    true_params['quadrupole1'], true_params['quadrupole2'] = Q_from_mass(true_params['mass1']), Q_from_mass(true_params['mass2'])
    true_params['dquad_mon1'], true_params['dquad_mon2'] = true_params['quadrupole1'] - 1, true_params['quadrupole2'] - 1    
    
    true_params['Lambda_tilde'], true_params['delta_Lambda_tilde'] = Lambda_tilde_delta_Lambda_tilde_from_Lambda1_Lambda2(true_params['lambda1'], true_params['lambda2'], true_params['mass1'], true_params['mass2'])
    true_params['q_tilde'], true_params['delta_q_tilde'] = q_tilde_delta_q_tilde_from_q1q2(true_params['quadrupole1'], true_params['quadrupole2'], true_params['spin1z'], true_params['spin2z'], true_params['mass1'], true_params['mass2'])
    
    data_zero_noise = GWtool_instance.get_strain(**true_params)        
    event_data_list.append(data_zero_noise)
    true_params_list.append(true_params)
    snr = np.sqrt(GWtool_instance.network_inner_product(data_zero_noise, data_zero_noise, network))
    snr_list.append(snr)

import heapq
n_top_snr = 1
indexed = [(value, idx) for idx, value in enumerate(snr_list)]
top_n = heapq.nlargest(n_top_snr, indexed)
top_n_index = [idx for value, idx in top_n]
print(top_n_index)
for i in top_n_index:
    print(snr_list[i])
    print(true_params_list[i])

index = 0
quasi_likelihood = []
for i in top_n_index:
    index += 1
    # Parameters for PE
    full_param_lis = ['chirp_mass', 'eta', 
                      'Lambda_tilde', 'delta_Lambda_tilde', 
                      'spin1z','spin2z','distance','delta_tc','ra','dec','pol', 'inclination','phi_c',
                      'q_tilde','delta_q_tilde'] #'quadrupole1', 'quadrupole2']
    v_param_lis = full_param_lis#['chirp_mass','eta','distance','Lambda_tilde','delta_Lambda_tilde','q_tilde','delta_q_tilde'] #,'delta_tc','spin1z','spin2z']
    likelihood = GW_likelihood(GWtool_instance, data=event_data_list[i], param_names=full_param_lis)

    true_params = true_params_list[i]
    bounds_dic = calculate_bounds(GWtool_instance,true_params,v_param_lis)
    priors, _ = calculate_priors(bounds_dic)
    for param in full_param_lis:
        if param not in v_param_lis:
            priors[param] = true_params[param]
    priors = PriorDict(priors)

    label = 'pycbc_waveform'
    sampler = 'nessai'
    outdir = f"./Test_Results/{label}_{sampler}_{len(v_param_lis)}d"
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)
    model = BilbyModel(
        priors=priors,
        likelihood=likelihood,
        use_ratio=False   # Whether to use the log-likelihood ratio
    )
    fs = FlowSampler(model,
                    nlive=args.nlive,
                    stopping=args.dlogz,
                    parallelise_prior=True,
                    output=outdir,
                    resume=True,
                    n_pool=args.npool,
                    flow_config=dict(
                        n_blocks=4,
                        n_layers=2,
                        n_neurons=16,
                        linear_transform='lu'
                    ),
                    training_config=dict(
                        lr=3e-3,
                        batch_size=1000,
                        max_epochs=500,
                        patience=20
                    ))

    logger = setup_logger(output=outdir)
    fs.run()

    posterior_samples = pd.DataFrame(fs.posterior_samples)
    posterior_samples = posterior_samples.iloc[:, :-1]
  
    posterior_samples.to_csv(outdir + '/posterior_data.csv', index=False)
    posterior_samples = posterior_samples.iloc[:, :len(v_param_lis)]
    samples = posterior_samples.values

    import matplotlib.pyplot as plt
    from corner import corner

    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'axes.labelsize': 28,
        'axes.titlesize': 32,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 18,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'figure.autolayout': False
    })
    '''
    labels_dic = {'chirp_mass':"${\cal{M}}_c\, [M_{\odot}]$",'eta':r'$\eta$',
                    'distance':r"$d_L\, [{\rm Mpc}]$",'delta_tc':r"$\Delta t_c\, [{\rm s}]$",
                    'ra':r"$\alpha$",'dec':"$\delta$",
                    'pol':"$\psi$",'inclination':"$\iota$",
                    'phi_c':"$\phi_c$",'Lambda_tilde':r'$\widetilde{\Lambda}$','delta_Lambda_tilde':r'$\delta \widetilde{\Lambda}$',
                    'q_tilde':r"$\widetilde{Q}$",'delta_q_tilde':r"$\delta \widetilde{Q}$",
                    'spin1z':"$\chi_{1}$",'spin2z':"$\chi_{2}$",
                    'quadrupole1':"$Q_{1}$",'quadrupole2':"$Q_{2}$",
                    'lambda1':"$\Lambda_{1}$",'lambda2':"$\Lambda_{2}$",
                    }
    '''
    config = dict(
        show_titles=True,
        title_kwargs={'pad': 10, 'fontsize': 24},
        title_fmt=".3f",
        #  use_math_text = True,
        bins=30,
        smooth=1,
        color='#0072C1',
        truths=[true_params[key] for key in v_param_lis],
        truth_color='tab:orange',
        quantiles=[0.16, 0.5,0.84],
        levels=[0.393, 0.675, 0.864],
        labelpad=0.1,
        plot_density=False,
        plot_datapoints=True,
        fill_contours=True,
        max_n_ticks=3,
        hist_kwargs={'density': True, 'lw': 1.},
    )
    labels = v_param_lis
    plot_fig = corner(samples, labels=labels, **config)
    plot_fig.savefig(outdir+'/My_corner.png',bbox_inches='tight')

    if 'Lambda_tilde' in v_param_lis and 'delta_Lambda_tilde' in v_param_lis:
        selected_params = ['lambda1', 'lambda2', 'quadrupole1', 'quadrupole2']
        labels = selected_params
        m1_samples, m2_samples = m1m2_from_mceta(posterior_samples['chirp_mass'].values, posterior_samples['eta'].values)
        lambda1_samples, lambda2_samples = Lambda1_Lambda2_from_Lambda_tilde_delta_Lambda_tilde(
            posterior_samples['Lambda_tilde'].values, posterior_samples['delta_Lambda_tilde'].values,
            m1_samples, m2_samples)
        q1_samples, q2_samples = q1q2_from_q_tilde_delta_q_tilde(posterior_samples['q_tilde'].values, posterior_samples['delta_q_tilde'].values, 
                                                                posterior_samples['spin1z'].values, posterior_samples['spin2z'].values,
                                                                m1_samples, m2_samples)
            
        marginalized_samples = np.vstack([lambda1_samples, lambda2_samples, q1_samples, q2_samples])

        cordinate_trans_config = dict(
            show_titles=True,
            title_kwargs={'pad': 10, 'fontsize': 24},
            title_fmt=".1f",
            # use_math_text = True,
            bins=30,
            smooth=1,
            color='#0072C1',
            truths=[true_params[key] for key in selected_params],
            truth_color='tab:orange',
            quantiles=[0.16, 0.5,0.84],
            levels=[0.393, 0.675, 0.864],
            labelpad=0.1,
            plot_density=False,
            plot_datapoints=True,
            fill_contours=True,
            max_n_ticks=3,
            hist_kwargs={'density': True, 'lw': 1.},
        )
        plot_fig = corner(marginalized_samples.T, labels=labels, **cordinate_trans_config)
        plot_fig.savefig(outdir+'/l1l2_q1q2_corner.png',bbox_inches='tight')


