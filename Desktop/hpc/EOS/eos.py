import os
import numpy as np
from numpy import log10 as lg
from scipy.interpolate import interp1d as sp_interp1d
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

DATA_DIR = os.path.join(os.path.dirname(__file__), "./")

EOS_NAMES_All = ['AP4']


class EOS:

    def __init__(self, name='AP4'):

        self.name = name
        self.mB = 1.660538921e-24  # g

        dat = np.genfromtxt(DATA_DIR + name + '.txt')

        self.rho, self.p, self.e = dat[:, 0], dat[:, 1], dat[:, 2]
        self.min_rho = np.min(self.rho)
        self.max_rho = np.max(self.rho)
        self.min_p = np.min(self.p)
        self.max_p = np.max(self.p)
        self.min_e = np.min(self.e)
        self.max_e = np.max(self.e)
        self.min_n = self.rho2n(self.min_rho)
        self.max_n = self.rho2n(self.max_rho)

        lgrho, lgp, lge = lg(self.rho), lg(self.p), lg(self.e)
        self.lgrho2lgp = sp_interp1d(lgrho, lgp)
        self.lgrho2lge = sp_interp1d(lgrho, lge)
        self.lgp2lgrho = sp_interp1d(lgp, lgrho)
        self.lgp2lge = sp_interp1d(lgp, lge)
        self.lge2lgp = sp_interp1d(lge, lgp)
        self.lge2lgrho = sp_interp1d(lge, lgrho)

    def rho2p(self, rho):
        return 10.0**self.lgrho2lgp(lg(rho))

    def rho2n(self, rho):
        return rho / self.mB

    def rho2e(self, rho):
        return 10.0**self.lgrho2lge(lg(rho))

    def p2rho(self, p):
        return 10.0**self.lgp2lgrho(lg(p))

    def p2n(self, p):
        return self.p2rho(p) / self.mB

    def p2e(self, p):
        if p < self.min_p:
            return 0.0
        return 10.0**self.lgp2lge(lg(p))

    def n2rho(self, n):
        return n * self.mB

    def n2p(self, n):
        return self.rho2p(n * self.mB)

    def n2e(self, n):
        return self.rho2e(n * self.mB)

    def e2p(self, e):
        return 10.0**self.lge2lgp(lg(e))

    def e2n(self, e):
        return self.e2rho(e) / self.mB

    def e2rho(self, e):
        return 10.0**self.lge2lgrho(lg(e))
        