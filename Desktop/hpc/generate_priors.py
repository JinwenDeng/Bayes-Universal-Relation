import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import bilby
from basic_code_pycbc_waveform import GWtool
from dL_z_relation import D_l_to_z, z_to_D_l
from copy import deepcopy
pi = np.pi
'''
labels = {'mass1':'$m_1$  $(M_\odot)$', 'mass2':'$m_2$  $(M_\odot)$',
'lambda1':'$\Lambda_1$','lambda2':'$\Lambda_2$','chirp_mass':"${\cal{M}}_c\, (M_{\odot})$",
'q':'$q$','eta':r'$\eta$','spin1z':'$\chi_{1}$', 
'comoving_volume':'$V_{comoving}$','spin2z':'$chi_{2}$','ra':r'$\alpha$', 
'dec':'$\delta$','sd':'$\sin \delta$' ,'sin_dec':'sin$\,\delta$','tc':'$t_c$',
'pol':'$\psi$', 'inclination':'$\iota$','cos_iota':'cos$\,\iota$',
'delta_tc':r"$\Delta t_c\,(\rm{s})$",'distance':r"$d_L\, (\rm{Mpc})$",'log_posterior':'log$\,P$',
'Lambda_tilde':r'$\widetilde{\Lambda}$','delta_Lambda_tilde':r'$\delta \widetilde{\Lambda}$',
'quadrupole1':'$Q_1$','quadrupole2':'$Q_2$','q_tilde':r"$\widetilde{Q}$",'delta_q_tilde':r"$\delta \widetilde{Q}$",
}
'''
# GWtool_instance =  GWtool(['CE2_H1','CE2_L1','ETD_E1'],f_num_factor=200)
def calculate_bounds(GWtool_instance,true_params_det,full_param_lis):
    FM,CM = GWtool_instance.fisher_matrix(v_param_lis=full_param_lis,use_mp=True,**true_params_det)
    stds = np.diag(CM)**0.5
    bounds_dic = {}
    abs_len = {'chirp_mass': 0.1,'eta':0.25,'distance':10,'delta_tc':1,
                    'ra':0.2,'dec':0.2,'pol':0.2,'inclination':0.2,
                    'lambda1':10,'lambda2':10,'Lambda_tilde':500,'delta_Lambda_tilde':100,
                    'quadrupole1':1,'quadrupole2':1,'q_tilde':5,'delta_q_tilde':20,
                    'spin1z':0.2,'spin2z':0.2,'phi_c':0.2
                    }
    for i, param in enumerate(full_param_lis):
        n_sigma = 5
        if param == 'q_tilde' or param == 'delta_q_tilde':
            n_sigma = 15
        std_i = stds[i]
        len_ = min(abs_len[param],n_sigma*std_i)
        bounds_dic[param] = [true_params_det[param] - len_, true_params_det[param] + len_]
    # modify the bounds range for some parameters
    bounds_phy = {'chirp_mass': [0,np.infty],'eta':[0,0.25],'distance':[0,np.infty],'delta_tc':[-10,10],
                    'ra':[0,2*pi],'dec':[-pi/2,pi/2],'pol':[0,pi],'inclination':[0,pi],
                    'lambda1':[0,5000],'lambda2':[0,5000],'Lambda_tilde':[0,np.infty],'delta_Lambda_tilde':[-np.infty,np.infty],
                    'quadrupole1':[1,100],'quadrupole2':[1,100],'q_tilde':[-1000,1000],'delta_q_tilde':[-1000,1000],
                    'spin1z':[-1,1],'spin2z':[-1,1],'phi_c':[-np.infty,np.infty]
                    }
    for key in bounds_phy:
        if key in bounds_dic:
            bounds_dic[key] = [max(bounds_dic[key][0],bounds_phy[key][0]),min(bounds_dic[key][1],bounds_phy[key][1])]
    return bounds_dic

def calculate_priors(bounds_dic):
    z_range = D_l_to_z(bounds_dic['distance'])
    bilby_z_prior = bilby.gw.prior.UniformSourceFrame(name='redshift', minimum=z_range[0], maximum=z_range[1])
    Mc_range = np.array([bounds_dic['chirp_mass'][0]/(1+z_range[1]),bounds_dic['chirp_mass'][1]/(1+z_range[0])]) 
    bilby_Mc_prior = bilby.gw.prior.Uniform(name='chirp_mass', minimum=Mc_range[0], maximum=Mc_range[1]) # note that this is in the source frame
    Mc_det_range = Mc_range*(1+z_range)


    def cal_Mc_det_prior_margin(Mc_det):
        z_max_inte = Mc_det/Mc_range[0]-1
        z_min_inte = Mc_det/Mc_range[1]-1
        z_max_inte = min(z_max_inte,z_range[1])
        z_min_inte = max(z_min_inte,z_range[0])
        if z_max_inte<=z_min_inte:
            return 0
        zs_inte = np.linspace(z_min_inte,z_max_inte,40000)
        pdf_inte = bilby_z_prior.prob(zs_inte)/(1+zs_inte)*bilby_Mc_prior.prob(Mc_det/(1+zs_inte))
        return np.trapz(pdf_inte,zs_inte)
    Mc_dets = np.linspace(Mc_det_range[0],Mc_det_range[1],10000)
    pdf_Mc_det = np.array([cal_Mc_det_prior_margin(Mc_det) for Mc_det in Mc_dets])
    bilby_Mc_det_prior = bilby.core.prior.interpolated.Interped(Mc_dets,pdf_Mc_det,
                                                                minimum=Mc_det_range[0],
                                                                maximum=Mc_det_range[1],name='chirp_mass_det',latex_label='M_c^det')
    def z_prior_give_Mc_det(z_,Mc_det_):
        if bilby_Mc_det_prior.prob(Mc_det_) <=0:
            return np.zeros_like(z_)
        else:
            return bilby_z_prior.prob(z_)/(1+z_)*bilby_Mc_prior.prob(Mc_det_/(1+z_))/bilby_Mc_det_prior.prob(Mc_det_)

    # priors = {key: bilby.core.prior.Uniform(bounds_dic[key][0],bounds_dic[key][1],name=key) for key in bounds_dic}
    priors = {}
    for key in bounds_dic:
        priors.update({key: bilby.core.prior.Uniform(bounds_dic[key][0],bounds_dic[key][1],name=key)})
        if key == 'chirp_mass':
            priors.update({'chirp_mass': bilby_Mc_det_prior})
        if key == 'dec':
            priors.update({'dec': bilby.core.prior.Cosine(name='dec',minimum=bounds_dic['dec'][0], maximum=bounds_dic['dec'][1])})
        if key == 'inclination':
            priors.update({'inclination': bilby.core.prior.Sine(name='inclination',minimum=bounds_dic['inclination'][0], maximum=bounds_dic['inclination'][1])})
        if key == 'distance':
            priors.update({'distance': bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=bounds_dic['distance'][0], maximum=bounds_dic['distance'][1])})

    return priors, z_prior_give_Mc_det # prior for unmarginalized parameters, and conditional prior for marginalized parameters

'''
import matplotlib.pyplot as plt
from corner import corner
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'axes.labelsize': 28,
    'axes.titlesize': 32,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 18,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'figure.autolayout': False
})

def q_tilde_delta_q_tilde_from_q1q2(Q1, Q2, spin1z, spin2z, mass1, mass2):
    m_sum = mass1 + mass2
    q1 = - Q1*mass1*spin1z**2/m_sum**2
    q2 = - Q2*mass2*spin2z**2/m_sum**2
    q_tilde = 50*(q1+q2)
    delta_q_tilde = (-5/84)*(9470+8218*mass1/m_sum-2016*(mass1/m_sum)**2)*q1+(-5/84)*(9470+8218*mass2/m_sum-2016*(mass2/m_sum)**2)*q2
    return q_tilde, delta_q_tilde  # 3/128/x**2.5/params['eta']*(q_tilde*x**2+delta_q_tilde*x**3)

network = ['CE2_H1', 'CE2_L1', 'ETD_E1', 'ETD_E2', 'ETD_E3']
GWtool_instance = GWtool(network)

true_params = {'mass1': 1.8326471290456863, 'mass2': 1.3642847316666025,
                   'distance': 54.282797422027684, 'delta_tc': 0.0, 
                   'approximant': 'IMRPhenomXAS_NRTidalv3', 
                   'chirp_mass': 1.37354738306275, 'eta': 0.24463416529986998, 
                   'ra': 3.7853115079332635, 'dec': 1.261825289496651, 'pol': 0.13862391294645482, 'inclination': 0.5370049455662562, 
                   'phi_c': 0.7486919446937721, 'coa_phase': 0.37434597234688605, 
                   'spin1z': 0.5477174121612478, 'spin2z': 0.4677750791308994, 
                   #'lambda1': 37.85317850961079, 'lambda2': 315.3531261584679, 
                   'quadrupole1': 2.6495834217405902, 'quadrupole2': 5.098349691052437, 
                   'Lambda_tilde': 2392.4759354405173, 'delta_Lambda_tilde': 31.351951262962114, 
                   }
true_params['q_tilde'], true_params['delta_q_tilde'] = q_tilde_delta_q_tilde_from_q1q2(true_params['quadrupole1'], true_params['quadrupole2'], true_params['spin1z'], true_params['spin2z'], true_params['mass1'], true_params['mass2'])

full_param_lis = ['chirp_mass', 'eta', 
                    'Lambda_tilde', 'delta_Lambda_tilde', 
                    'spin1z','spin2z','distance','delta_tc','ra','dec','pol', 'inclination','phi_c',
                    'q_tilde','delta_q_tilde',] # 'quadrupole1', 'quadrupole2']
v_param_lis = ['chirp_mass','eta','distance','delta_tc','spin1z','spin2z','Lambda_tilde','delta_Lambda_tilde','q_tilde','delta_q_tilde',]
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
outdir = f"./Test_Results/prior_{len(v_param_lis)}d"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

bounds_dic = calculate_bounds(GWtool_instance,true_params,v_param_lis)
priors, _ = calculate_priors(bounds_dic)

prior_samples = []
for key in v_param_lis:
    prior_samples.append(priors[key].sample(size=10000))

prior_samples = np.array(prior_samples)
prior_labels = [labels_dic[key] for key in v_param_lis]
prior_config = dict(
    show_titles=True,
    title_kwargs={'pad': 10, 'fontsize': 24},
    title_fmt=".3f",
    # use_math_text = True,
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
plot_fig = corner(prior_samples.T,labels=prior_labels, **prior_config)
plot_fig.savefig(outdir+'/prior_corner.png',bbox_inches='tight')

for param in full_param_lis:
    if param not in v_param_lis:
        priors[param] = true_params[param]

'''


