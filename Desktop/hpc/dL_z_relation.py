import h5py
import numpy as np
import os
from scipy.interpolate import interp1d

from astropy.cosmology import LambdaCDM

# 定义宇宙学模型
cosmo = LambdaCDM(H0=67.4, Om0=0.315, Ode0=0.685)

# 定义红移和视亮距离范围
z_lis_linear = np.linspace(0, 20, 200000)
z_lis_log = np.logspace(-9, 1, 200000)
z_total = sorted(np.concatenate((z_lis_linear, z_lis_log)))
dl = cosmo.luminosity_distance(z_total).value
dc = np.array([D_l/(1+z) for z, D_l in zip(z_total, dl)])


with h5py.File('dl_z_data.h5', 'w') as h5file:
    h5file.create_dataset('z_total', data=z_total, ) 
    h5file.create_dataset('dl_total', data=dl, )
    h5file.create_dataset('dc_total', data=dc, )

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'dl_z_data.h5')  
# 打开 HDF5 文件并读取数据
with h5py.File(path, 'r') as h5file:
    z_total = h5file['z_total'][:]
    dl_total = h5file['dl_total'][:]
    dc_total = h5file['dc_total'][:]

# 创建插值函数
redshift_from_luminosity_distance_interp = interp1d(dl_total, z_total, kind=1, bounds_error=True)
luminosity_distance_from_redshift_interp = interp1d(z_total, dl_total, kind=1, bounds_error=True)
redshift_from_comoving_distance_interp = interp1d(dc_total, z_total, kind=1, bounds_error=True)
comoving_distance_from_redshift_interp = interp1d(z_total, dc_total, kind=1, bounds_error=True)

def D_l_to_z(D_l):
    return redshift_from_luminosity_distance_interp(D_l)

def z_to_D_l(z):
    return luminosity_distance_from_redshift_interp(z)

def D_c_to_z(D_c):
    return redshift_from_comoving_distance_interp(D_c)

def z_to_D_c(z):
    return comoving_distance_from_redshift_interp(z)