import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import log10 as lg
from numpy import pi as pi
from numpy import sin as sin
from numpy import cos as cos
from scipy.interpolate import interp1d as sp_interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.integrate import ode
from scipy.integrate import dblquad
import warnings
import timeit
import scipy.optimize as opt
from matplotlib import cm
from astropy import constants as const
from astropy import units as u
from scipy.special import lqmn as qmn
import scipy.special as sc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                              AutoMinorLocator)
from sympy import var, plot_implicit
import math
from math import radians as radian

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

eos_name = 'SLy4'
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "EOS", eos_name+'.txt')  
data = np.genfromtxt(file_path)
rho, p, e = data[:,0], data[:,1], data[:,2]
#提取某一个物态 分别为密度 压强 能量

#对状态方程做插值，在对数空间
min_e=min(e)
max_e=max(e)
min_p=min(p)
max_p=max(p)

lge, lgp = lg(e), lg(p)
lgp2lge = sp_interp1d(lgp, lge)
lge2lgp = sp_interp1d(lge, lgp)
#在对数空间做插值函数，转换成实数空间

def e2p(e):
    if e < min_e or e > max_e:
        return 0.0
    else:
        return 10**(lge2lgp(lg(e)))
    
def p2e(p):
    if p < min_p or p > max_p:   
        return 0.0
    else:
        return 10.0**(lgp2lge(lg(p)))
# y = x2y(x)

def dedp(p, rel_dp=1e-6):
    dp = p*rel_dp
    if p < min_p or p > max_p:   
        return 0.0
    else:
        return (p2e(p+dp)-p2e(p-dp))/(2*dp)

# dydx表示y对x的导数
def tov(r, y):
    p, m = y
    e = p2e(p)
    dmdr=4*pi * r**2 * e
    dpdr=-G*(e+p/c**2)*(m+4*pi*r**3*p/c**2)/r**2/(1-2*G*m/c**2/r)
    dydr = [dpdr, dmdr]
    return dydr

# 求解结束条件
def hit_ground(r, y): return y[0]-min_p   # 压强达到状态方程提供的最小值
hit_ground.terminal = True
hit_ground.direction = -1   # 终止时压强导数应该为负，即压强在减小

# r0以内设为密度均匀的中子星核心
r0=1
r1=2*10**7
#半径一定不会跑出r1，hit_ground之后就会找到中子星半径

#要把[0,r0]的质量计算进去

ec_list = np.linspace(10**14, 3*10**15, 1000)
R_list = []
M_list = []
for ec in ec_list:
    pc=e2p(ec)
    mc=4*pi/3*r0**3*ec
    sol = solve_ivp(tov, [r0, r1], [pc, mc],events=hit_ground, max_step=(r1-r0)/1000)
    #sol.t是半径 sol.y[0]是压强 sol.y[1]是质量
    R_list.append(sol.t[-1])
    M_list.append(sol.y[1][-1])

R_list = np.array(R_list)
M_list = np.array(M_list)

def R_from_M(m):    # 以太阳质量为单位
    m = m*Ms
    r_from_m = sp_interp1d(M_list, R_list)
    return r_from_m(m)
    
def ec_from_m(m):
    m = m*Ms
    ec_from_m = sp_interp1d(M_list, ec_list)
    return ec_from_m(m)

def I_Love_Q_from_mass(m):
    ec=ec_from_m(m)
    pc=e2p(ec)
    mc=4*pi/3*r0**3*ec
    sol = solve_ivp(tov, [r0, r1], [pc, mc],events=hit_ground, max_step=(r1-r0)/1000)
    #sol.t是半径 sol.y[0]是压强 sol.y[1]是质量
    R = sol.t[-1]
    M = sol.y[1][-1]
    C = G*M/R/c**2
    # 对p，m插值
    beg_m=mc
    end_m=M
    beg_p=min_p
    end_p=max(sol.y[0])

    sol_lgp, sol_lgm, sol_lgr = lg(sol.y[0]), lg(sol.y[1]), lg(sol.t)
    sol_lgr2lgm = sp_interp1d(sol_lgr, sol_lgm)
    sol_lgr2lgp = sp_interp1d(sol_lgr, sol_lgp)
    #在对数空间做插值函数，转换成实数空间

    def sol_p(r):
        if r < r0:
            return pc
        elif r > R:
            return min_p
        else:
            return 10**(sol_lgr2lgp(lg(r)))
        
    def sol_m(r):
        if r < r0:
            return 4*pi/3*r**3*ec
        elif r > R:   
            return M
        else:
            return 10.0**(sol_lgr2lgm(lg(r)))
    def sol_e(r):
        if r < r0:
            return ec
        elif r > R:   
            return min_e
        else:
            return p2e(sol_p(r))

    # 求解nu
    nu_r=np.log(1-2*G*M/R/c**2)
    def nu_solving(r, nu):
        dnudr=2*G/c**2*(sol_m(r)+4*pi*r**3*sol_p(r)/c**2)/(r*(r-2*G*sol_m(r)/c**2))
        return dnudr

    sol1 = solve_ivp(nu_solving, [R, r0], [nu_r], max_step=(R-r0)/1000)    

    sol_lgr1 = lg(sol1.t)
    sol_lgr2nu = sp_interp1d(sol_lgr1, sol1.y[0])

    def sol_nu(r):
        if r < r0 or r > R:
            return 0.0
        else:
            return sol_lgr2nu(lg(r))

    def exp_lambda(r):    
        return 1/(1-2*G*sol_m(r)/r/c**2)
        
    def dnudr(r, dr):
        if r < r0 or r > R:
            return 0
        else:
            return (sol_nu(r+dr)-sol_nu(r-dr))/(2*dr)

    def h_solver(r, y):
        h, h_aux = y
        dhdr = h_aux
        dh_auxdr = -(2/r+G/c**2*(2*sol_m(r)/r**2+4*pi*r*(sol_p(r)/c**2-sol_e(r)))*exp_lambda(r))*h_aux+(6*exp_lambda(r)/r**2-4*pi*G/c**2*(5*sol_e(r)+9*sol_p(r)/c**2+c**2*(sol_e(r)+sol_p(r)/c**2)*dedp(sol_p(r)))*exp_lambda(r)+(dnudr(r, dr))**2)*h
        dydr = [dhdr, dh_auxdr]
        return dydr

    # lambda and love number
    B = 100
    dr = R * 10**(-7)
    sol_h = solve_ivp(h_solver, [r0, R], [B*(1.5*r0)**2, 3*B*r0])

    sol_lgr2 = lg(sol_h.t)
    sol_lgr2h = sp_interp1d(sol_lgr2, sol_h.y[0])
    sol_lgr2h_aux = sp_interp1d(sol_lgr2, sol_h.y[1])

    y = R * sol_lgr2h_aux(lg(R-1.5*dr)) / sol_lgr2h(lg(R-1.5*dr))
    C = G * M / c**2 / R

    k_2 = 1.6*(C**5)*((1-2*C)**2)*(2+2*C*(y-1)-y)/(2*C*(6-3*y+3*C*(5*y-8))+4*(C**3)*(13-11*y+C*(3*y-2)+2*(C**2)*(1+y))+3*((1-2*C)**2)*(2-y+2*C*(y-1))*np.log(1-2*C))
    Lambda = 2/3*k_2/C**5

    '''
    from bilby.gw.eos import EOSFamily, TabularEOS
    eos = TabularEOS(eos_name.upper())
    fam = EOSFamily(eos)
    print(fam.radius_from_mass(M/Ms),fam.k2_from_mass(M/Ms),fam.lambda_from_mass(M/Ms))
    '''

    # I
    def omega_solver(r, y):
        omega,omega_aux = y
        domegadr = omega_aux
        domega_auxdr = -4/r*(1-(G/c**2)*pi*r**2*(sol_e(r)+sol_p(r)/c**2)*exp_lambda(r))*omega_aux + 16*(G/c**2)*pi*(sol_e(r)+sol_p(r)/c**2)*exp_lambda(r)*omega
        dydr = [domegadr, domega_auxdr]
        return dydr

    chi = 0.1
    S = chi * M**2 * (G/c**2) * c

    dr = R * 10**(-6)
    solving_omega = solve_ivp(omega_solver, [r0, R], [1+8*G/c**2*pi/5*(ec+pc/c**2)*r0**2, 16*G/c**2*pi/5*(ec+pc/c**2)*r0], max_step=(R-r0)/1000)
    sol_lgr = lg(solving_omega.t)
    sol_lgr2omega = sp_interp1d(sol_lgr, solving_omega.y[0])

    def sol_omega1(r):
        if r < r0 or r > R:
            return 0.0
        else:
            return sol_lgr2omega(lg(r))
    omega_c = G/c**2 * 6*S/R**4/((sol_omega1(R)-sol_omega1(R-dr))/dr)
    Omega = G/c**2 * 2*S/R**3 + omega_c*sol_omega1(R)
    I = S / Omega
    I_bar = I/(M**3*(G/c**2)**2)

    # Q
    omega_c = 1e20
    # u ~ chi * r**3, chi ~ t**-1, u ~ r**2 * c
    def chi_u_solver(r, y):
        e = G/c**2 * sol_e(r) # r**-2
        p = G/c**4 * sol_p(r) # r**-2
        m = G/c**2 * sol_m(r) # r
        chi, u = y
        dchidr = u/r**4 - 4*pi*r**2 *(e+p)/(r-2*m)*chi
        dudr = 16*pi*r**5*(e+p)/(r-2*m)*chi
        dydr = [dchidr, dudr]
        return dydr

    j0 = np.exp(-sol_nu(r0)/2)*np.sqrt(1-2*G*sol_m(r0)/c**2/r0)
    solving_chi_u = solve_ivp(chi_u_solver, [r0, R], [j0*omega_c, 0.0], max_step=(R-r0)/1000)
    sol_lgr = lg(solving_chi_u.t)
    sol_lgr2chi = sp_interp1d(sol_lgr, solving_chi_u.y[0])
    sol_lgr2u = sp_interp1d(sol_lgr, solving_chi_u.y[1])

    J = solving_chi_u.y[1][-1]/6 / (G/c**2)
    Omega = solving_chi_u.y[0][-1] + 2*J/R**3/(G/c**2)

    def sol_chi(r):
        if r < r0:
            return j0*omega_c
        elif r > R:
            return Omega - 2*J/r**3/(G/c**2)
        else:
            return sol_lgr2chi(lg(r))
        
    def sol_u(r):
        if r < r0:
            return 0.0
        elif r > R:
            return 6*J/(G/c**2)
        else:
            return sol_lgr2u(lg(r))
        
    def v_h_solver(r, y):
        e = G/c**2 * sol_e(r) # r**-2
        p = G/c**4 * sol_p(r) # r**-2
        m = G/c**2 * sol_m(r) # r
        chi = sol_chi(r)
        u = sol_u(r)
        dnudr = 2*(m+4*pi*r**3*p)/(r*(r-2*m))
        v, h = y
        dvdr = -dnudr*h + c**(-2)*(1/r+0.5*dnudr)*(8*pi*r**5*(e+p)*chi**2/3/(r-2*m)+u**2/6/r**4)
        dhdr = (-dnudr+r/(r-2*m)/dnudr*(8*pi*(e+p)-4*m/r**3))*h - 4*v/(r-2*m)/r/dnudr + u**2/6/r**5*(0.5*dnudr*r - 1/(r-2*m)/dnudr)/c**2 + 8*pi*r**5*(e+p)*chi**2/3/(r-2*m)/r*(0.5*dnudr*r+1/(r-2*m)/dnudr)/c**2
        dydr = [dvdr, dhdr]
        return dydr

    def v_h_solver_homo(r, y):  # 齐次方程
        e = G/c**2 * sol_e(r) # r**-2
        p = G/c**4 * sol_p(r) # r**-2
        m = G/c**2 * sol_m(r) # r
        chi = 0.0
        u = 0.0
        dnudr = 2*(m+4*pi*r**3*p)/(r*(r-2*m))
        v, h = y
        dvdr = -dnudr*h + c**(-2)*(1/r+0.5*dnudr)*(8*pi*r**5*(e+p)*chi**2/3/(r-2*m)+u**2/6/r**4)
        dhdr = (-dnudr+r/(r-2*m)/dnudr*(8*pi*(e+p)-4*m/r**3))*h - 4*v/(r-2*m)/r/dnudr + u**2/6/r**5*(0.5*dnudr*r - 1/(r-2*m)/dnudr)/c**2 + 8*pi*r**5*(e+p)*chi**2/3/(r-2*m)/r*(0.5*dnudr*r+1/(r-2*m)/dnudr)/c**2
        dydr = [dvdr, dhdr]
        return dydr

    def B_from_A(a):
        # a ~ r**-2, b ~ r**-4
        return 2/3*pi*G/c**2*(ec+pc/c**2)*(j0*omega_c/c)**2 - 2*pi*G/c**2*(ec/3+pc/c**2)*a

    A = 1
    solving_v_h_part = solve_ivp(v_h_solver, [r0, R], [B_from_A(A)*r0**4, A*r0**2], max_step=(R-r0)/1000)
    solving_v_h_homo = solve_ivp(v_h_solver_homo, [r0, R], [-2*pi*G/c**2*(ec/3+pc/c**2)*r0**4, r0**2], max_step=(R-r0)/1000)

    v_R_part = solving_v_h_part.y[0][-1]
    v_R_homo = solving_v_h_homo.y[0][-1]
    h_R_part = solving_v_h_part.y[1][-1]
    h_R_homo = solving_v_h_homo.y[1][-1]

    xi = R/(G*M/c**2)-1 
    Q_22 = 1.5*(xi**2-1)*np.log((xi+1)/(xi-1)) - (3*xi**3-5*xi)/(xi**2-1)
    Q_12 = np.sqrt(xi**2-1)*((3*xi**2-2)/(xi**2-1)-1.5*xi*np.log((xi+1)/(xi-1)))

    K = (((G*J/c**3)**2*(1/(G*M*R**3/c**2)+1/R**4)-h_R_part)/h_R_homo + ((J*G/c**3/R**2)**2+v_R_part)/v_R_homo) / (2*G*M/c**2/np.sqrt(R*(R-2*G*M/c**2))*Q_12/v_R_homo - Q_22/h_R_homo)
    Q_bar = 1 + 1.6*K*(M**2*c*G/c**2/J)**2
    # print(Q_bar, M/Ms)
    # print(h_R_homo, h_R_part, v_R_homo, v_R_part)

    return {'I': I_bar, 'Lambda': Lambda, 'Q': Q_bar }

m_list = np.linspace(0.25, 2.00, 500)
I_list, Lambda_list, Q_list = [], [], []
for m in m_list:
    I_Love_Q = I_Love_Q_from_mass(m)
    I_list.append(I_Love_Q['I'])
    Lambda_list.append(I_Love_Q['Lambda'])
    Q_list.append(I_Love_Q['Q'])

lg_I_list = np.array([lg(i) for i in I_list])
lg_Lambda_list = np.array([lg(l) for l in Lambda_list])
lg_Q_list = np.array([lg(q) for q in Q_list])

I_Love_Q_path = os.path.join(script_dir, "I Love Q", eos_name+'.txt')
data_to_save = np.column_stack((m_list, lg_I_list, lg_Lambda_list, lg_Q_list))

np.savetxt(I_Love_Q_path, data_to_save, 
           fmt=['%.10f', '%.10f', '%.10f', '%.10f'],  
           delimiter='\t',            # 制表符分隔
           )          

def I_from_mass(m):
    loaded_data = np.loadtxt(I_Love_Q_path, delimiter='\t')
    m_list = loaded_data[:, 0]
    lg_I_list = loaded_data[:, 1]
    I_interp = sp_interp1d(m_list, lg_I_list)
    return 10**(I_interp(m))

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

