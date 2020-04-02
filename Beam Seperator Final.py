"""
Beam Separator
=====

Provides
  Functions for modelling the performance of a simulated beam separator
  proposed for use at the NA62 experiment at CERN, as used in UoB Particle
  Physics Y3 Group Study 2020.
"""


import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import ss2
import shlex
database = ss2.ParticleDatabase()
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

def make_patch_spines_invisible(ax):
    """
    Makes patch spines of ax object invisible.
    
    Parameters
    ----------
    ax : matplotlib.axis
        Axis to act on
        
    Returns
    -------
    None
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return


def multiple_formatter(denominator=12, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

def multiple_formatter_non_int(denominator=12, number=np.pi, latex='\pi'):
    def _multiple_formatter_non_int(x, pos):
        den = denominator
        num = den*x/number
        if num == 0:
            return r'$0$'
        if num==1:
            return r'$%s$'%latex
        elif num==-1:
            return r'$-%s$'%latex
        else:
            return r'$%s%s$'%(num,latex)
    return _multiple_formatter_non_int

class Multiple:
    def __init__(self, denominator=12, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

c = 299792458
alpha = 0.00729735257
e = np.sqrt(4*np.pi*alpha)
kaon = database['K+']
pion = database['pi+']
proton = database['p+']

def beta(p, m):
    """
    Calculates beta factor of given particle.
    
    Parameters
    ----------
    p : float
        Momentum of particle.
    m : float
        Mass of particle.
    
    Returns
    -------
    beta : float
        Beta factor of particle.
    """
    return p/np.sqrt(p**2 + m**2)

def gamma(p, m):
    """
    Calculates gamma factor of given particle.
    
    Parameters
    ----------
    p : float
        Momentum of particle.
    m : float
        Mass of particle.
    
    Returns
    -------
    gamma : float
        Gamma factor of particle.
    """
    return 1/np.sqrt(1-beta(p, m)**2)

def comp_beta(m1=database['pi+'].mass, m2=database['K+'].mass, p=75):
    """
    Compares the beta factors of two given particles.
    
    Parameters
    ----------
    m1 : float
        Mass of first particle.
    m2 : float
        Mass of second particle.
    p : float
        Momentum of particle.
    
    Returns
    -------
    dBeta : float
        Difference between beta factors of particles.
    dt : float
        Difference per metre in travel time for the particles.
    rt : float
        Ratio between per metre travel times for the particles.
    """
    b1 = beta(p, m1)
    b2 = beta(p, m2)
    db = abs(b1 - b2)
    if np.log10(db) < -2:
        disp_db = db*10**(-round(np.log10(db)))
        text1 = 'dBeta = {0:.3f}e-0{1:.0f}, '.format(disp_db, -round(np.log10(db)))
    else:
        text1 = 'dBeta = {0:.3f}, '.format(db)
    t1_per_m = 1/(b1*c)
    t2_per_m = 1/(b2*c)
    if t1_per_m > t2_per_m:
        rt = 1 - (t2_per_m/t1_per_m)
    else:
        rt = 1 - (t1_per_m/t2_per_m)
    if np.log10(rt) < -2:
        disp_rt = rt*10**(-round(np.log10(rt)))
        text3 = ', rt = {0:.3f}e-0{1:.0f}'.format(disp_rt, -round(np.log10(rt)))
    else:
        text3 = ', rt = {0:.3f}'.format(rt)
    dt_per_m = abs(t1_per_m-t2_per_m)
    if np.log10(dt_per_m) >= -1:
        text2 = 'dt = {0:.3f} s'.format(dt_per_m)
    if np.log10(dt_per_m) < -1 and np.log10(dt_per_m) >= -4:
        text2 = 'dt = {0:.3f} ms'.format(dt_per_m*1e3)
    if np.log10(dt_per_m) < -4 and np.log10(dt_per_m) >= -7:
        text2 = 'dt = {0:.3f} microns'.format(dt_per_m*1e6)
    if np.log10(dt_per_m) < -7 and np.log10(dt_per_m) >= -10:
        text2 = 'dt = {0:.3f} ns'.format(dt_per_m*1e9)
    if np.log10(dt_per_m) < -10:
        text2 = 'dt = {0:.3f} ps'.format(dt_per_m*1e12)
    return text1+text2+text3

def comp_beta_plot(p1=database['K+'], p2=database['pi+'], pmax=0.5):
    """
    Compares the beta factors of two given particles graphically.
    
    Parameters
    ----------
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    pmax : float
        Momentum to plot up to.
    
    Returns
    -------
    graph
    """
    db = []
    mom = np.linspace(0, pmax, 1000)
    m1 = p1.mass
    m2 = p2.mass
    for p in mom:
        b1 = beta(p, m1)
        b2 = beta(p, m2)
        db.append(abs(b1-b2))
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(mom, db, 'b')
    ax.set_xlim(0, pmax)
    ax.set_ylim(0)
    ax.set_xlabel('p / GeV', fontsize=20)
    ax.set_ylabel(r'$\Delta\beta$', fontsize=20)
    title = f'{p1.name} to {p2.name} '
    title += r'$\Delta\beta$ dependancy on particle momenta'
    ax.set_title(title, fontsize=20)
    ax.text(0.30, 0.10, r'Maximum $\Delta\beta$ at p={0:.3f} GeV'.format(mom[np.argmax(db)]),
            transform=ax.transAxes, fontsize=20)
    plt.show()
    return

def comp_time_plot(p1=database['K+'], p2=database['pi+'], pmax=80, plot=True):
    """
    Compares the travel time per metre for two given particles graphically.
    
    Parameters
    ----------
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    pmax : float
        Momentum to plot up to.
    plot : True/False
        States whether a graph is to be plotted.
    
    Returns
    -------
    dt(12.5) : float
        dt at p = 12.5 GeV.
    dt(75) : float
        dt at p = 75 GeV.
    ratio : float
        ratio of dt(12.5):dt(75)>
    graph
    """
    dt = []
    p_range = np.linspace(10, pmax, 1000)
    m1 = p1.mass
    m2 = p2.mass
    for p in p_range:
        t1_per_m = 76.273/(beta(p, m1)*gamma(p, m1)*c)
        t2_per_m = 76.273/(beta(p, m2)*gamma(p, m2)*c)
        dt.append(abs(t1_per_m - t2_per_m)*1e12)
    dt_12_5 = dt[np.argmin(abs(p_range-12.5))]
    dt_75 = dt[np.argmin(abs(p_range-75))]
    ratio = dt_12_5/dt_75
    if plot==True:
        fig = plt.figure(figsize=[10, 5])
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(p_range, dt, 'b', label=r'$\Delta t$')
        ax.axvline(12.5, color='r', label='p=12.5 GeV')
        ax.axvline(75, color='g', label='p=75 GeV')
        ax.set_xlim(10, pmax)
        ax.set_ylim(0)
        ax.set_xlabel('p / GeV', fontsize=20)
#        ax.set_yscale('log')
        ax.set_ylabel(r'$\Delta t$ / ps', fontsize=20)
        title = f'{p1.name} to {p2.name} '
        title += r'$\Delta t$ dependancy on particle momenta'
        ax.set_title(title, fontsize=20)
        ax.legend(fontsize=20)
        text = 'dt(12.5) = {0:.2f} ps, '.format(dt_12_5)
        text += 'dt(75) = {0:.2f} ps, '.format(dt_75)
        text += 'ratio = {0:.3f}'.format(ratio)
        plt.show()
        print(text)
    return [dt_12_5, dt_75, ratio]

def comp_vel(p1=database['K+'], p2=database['pi+'], p3=database['p+'], pmin=0, pmax=80):
    """
    Plots the velocities of three given particles graphically.
    
    Parameters
    ----------
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    pmin : float
        Lower limit of plotted momentum.
    pmax : float
        Upper limit of plotted momentum.
    
    Returns
    -------
    graph
    """
    p_range = np.linspace(pmin, pmax, 1000)
    m1 = p1.mass
    m2 = p2.mass
    m3 = p3.mass
    v1, v2, v3 = [], [], []
    for p in p_range:
        v1.append(c*beta(p, m1))
        v2.append(c*beta(p, m2))
        v3.append(c*beta(p, m3))
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(1, 1, 1)
    p1_name = r'K$^+$'
    p2_name = r'$\pi^+$'
    p3_name = r'p$^+$'
    ax.plot(p_range, v1, 'r', label=p1_name)
    ax.plot(p_range, v2, 'b', label=p2_name)
    ax.plot(p_range, v3, 'g', label=p3_name)
    ax.set_xlabel('p / GeV', fontsize=20)
    ax.set_ylabel(r'v / $ms^{-1}$', fontsize=20)
    ax.axvline(75, color='k', label='p = 75 GeV')
    ax.set_xticks(np.arange(pmin, pmax+1, 1))
    ax.set_xticklabels(np.arange(pmin, pmax+1, 1))
    ax.grid()
    ax.minorticks_on()
    ax.set_xlim(pmin, pmax)
#    ax.set_ylim(np.min(v1+v2))
    ax.legend(fontsize=20)
    plt.show
    return

def delta_vel(p1=database['K+'], p2=database['pi+'], p3=database['p+'], pmin=0, pmax=80):
    """
    Compares the velocities of three given particles graphically.
    
    Parameters
    ----------
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    pmin : float
        Lower limit of plotted momentum.
    pmax : float
        Upper limit of plotted momentum.
    
    Returns
    -------
    graph
    """
    p_range = np.linspace(pmin, pmax, 1000)
    m1 = p1.mass
    m2 = p2.mass
    m3 = p3.mass
    dv2, dv3 = [], []
    for p in p_range:
        v1 = c*beta(p, m1)
        v2 = c*beta(p, m2)
        v3 = c*beta(p, m3)
        dv2.append(abs(v1-v2))
        dv3.append(abs(v1-v3))
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(1, 1, 1)
#    p1_name = r'K$^+$'
#    p2_name = r'$\pi^+$'
#    p3_name = r'p$^+$'
    ax.plot(p_range, dv2, 'r', label=r'$\left|v_{K^+}-v_{\pi^+}\right|$')
    ax.plot(p_range, dv3, 'b', label=r'$\left|v_{K^+}-v_{p^+}\right|$')
    ax.set_xlabel('p / GeV', fontsize=20)
    ax.set_ylabel(r'$\left|\Delta v\right|$ / $ms^{-1}$', fontsize=20)
    ax.axvline(75, color='k', label='p = 75 GeV')
    ax.set_xticks(np.arange(pmin, pmax+1, 1))
    ax.set_xticklabels(np.arange(pmin, pmax+1, 1))
    ax.grid()
    ax.minorticks_on()
    ax.set_xlim(pmin, pmax)
#    ax.set_ylim(np.min(v1+v2))
    ax.legend(fontsize=20, loc=[0.65, 0.2])
    plt.show
    return

def delta_beta(p1=database['K+'], p2=database['pi+'], p3=database['p+'], pmin=0, pmax=80):
    """
    Compares the beta factors of three given particles graphically.
    
    Parameters
    ----------
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    pmin : float
        Lower limit of plotted momentum.
    pmax : float
        Upper limit of plotted momentum.
    
    Returns
    -------
    graph
    """
    p_range = np.linspace(pmin, pmax, 1000)
    m1 = p1.mass
    m2 = p2.mass
    m3 = p3.mass
    db2, db3 = [], []
    for p in p_range:
        b1 = beta(p, m1)
        b2 = beta(p, m2)
        b3 = beta(p, m3)
        db2.append(abs(b1-b2))
        db3.append(abs(b1-b3))
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(1, 1, 1)
#    p1_name = r'K$^+$'
#    p2_name = r'$\pi^+$'
#    p3_name = r'p$^+$'
    ax.plot(p_range, db2, 'r', label=r'$\left|\beta_{K^+}-\beta_{\pi^+}\right|$')
    ax.plot(p_range, db3, 'b', label=r'$\left|\beta_{K^+}-\beta_{p^+}\right|$')
    ax.set_xlabel('p / GeV', fontsize=20)
    ax.set_ylabel(r'$\left|\Delta \beta\right|$', fontsize=20)
    ax.set_title(r'$\left|\Delta \beta\right|$ between $K^+$ and background particles as a functon of $p$', fontsize=20)
    ax.axvline(75, color='k', label='p = 75 GeV')
    ax.set_xticks(np.arange(pmin, pmax+1, 1))
    ax.set_xticklabels(np.arange(pmin, pmax+1, 1))
    ax.grid()
    ax.minorticks_on()
    ax.set_xlim(pmin, pmax)
#    ax.set_ylim(np.min(v1+v2))
    ax.legend(fontsize=20, loc=[0.65, 0.2])
    plt.show
    fig.savefig("Delta_Beta.pdf", bbox_inches='tight')
    return

def ang_deflection(p=75, f=1e11, p1=database['K+'], p2=database['pi+'],
                   L_t=79.6, l=2.74, E=1e6,delta_p=1.6e-2):
    """
    Calculates the angular deflection of particle p2 when passed through a
    beam separator with the specified parameters, tuned to cancel the
    deflection of particle p1.
    
    Parameters
    ----------
    p : float
        Momentum of particle beam.
    f : float
        Frequency of electric field.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    L_t : float
        Total length of beam separator.
    l : float
        Cavity length of beam separator.
    E : float
        Electric field strength
    delta_p : float
        Momentum spread of particle beam
    
    Returns
    -------
    ang_def : float
        Angular deflection from the longitudinal direction in mrad.
    """
    m1 = p1.mass
    m2 = p2.mass
    L = L_t - (2*l)
    A = (e*E*l)/(p*c)
    tau_21 = (2*np.pi*((L*f)/(c))*((1/beta(p, m2))-(1/beta(p, m1))))*(1-(2*delta_p))
#    tau_21 = ((np.pi*L*f)/c)*((m2**2-m1**2)/p**2)
    return abs(np.arctan(2*A*np.sin(tau_21/2))*1e3)

def ang_deflection_plot(pmax=100, pmin=5, f=1e11, p1=database['K+'],
                        p2=database['pi+'], p3=database['p+'], L_t=79.6,
                        l=2.74, E=1e6, delta_p=1.6e-2, highlight_p=None):
    """
    Compares the angular deflection of particles p2 and p3 when passed through
    a beam separator with the specified parameters, tuned to cancel the
    deflection of particle p1.
    
    Parameters
    ----------
    pmax : float
        Upper limit of plotted momentum of particle beam.
    pmin : float
        Lower limit of plotted momentum of particle beam.
    f : float
        Frequency of electric field.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    L_t : float
        Total length of beam separator.
    l : float
        Cavity length of beam separator.
    E : float
        Electric field strength
    delta_p : float
        Momentum spread of particle beam
    highlight_p : float
        Specific momentum to highlight and print deflections at.
        
    Returns
    -------
    ang_def : float
        Angular deflection from the longitudinal direction in mrad at
        p = highlight_p if not None.
    graph : graph
        Graph showing angular deflections.
    """
    p_range = np.linspace(pmin, pmax, 10000)
    defl2 = []
    defl3 = []
    for p in p_range:
        defl2.append(ang_deflection(p, f, p1, p2, L_t, l, E, delta_p))
        defl3.append(ang_deflection(p, f, p1, p3, L_t, l, E, delta_p))
    fig = plt.figure(figsize=[12, 4])
    ax = fig.add_subplot(1, 1, 1)
    #ax.plot(p_range, defl2, 'r', lw=1.5, label=p2.name, alpha=0.5)
    #ax.plot(p_range, defl3, 'b', lw=1.5, label=p3.name, alpha=0.5)
    ax.plot(p_range, defl2, 'r', lw=1.5, label=r'$\pi^+$', alpha=0.5)
    ax.plot(p_range, defl3, 'b', lw=1.5, label=r'$p^+$', alpha=0.5)
    ax.set_xlim(pmin, pmax)
    ax.set_ylim(0, 1.03*np.max(defl2+defl3))
#    ax.set_ylim(0, 1)
    ax.set_xlabel('p / GeV', fontsize=20)
    ax.set_xticks(np.arange(pmin, pmax+1, 10))
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
#    text = r'$D_{'
#    text += p2.name + '/' + p3.name
#    text += '}$ / mrad'
    text = r'$D_{\pi^+ / p^+}$ / mrad'
    ax.set_ylabel(text, fontsize=20)
    ax.set_title('Amplitude of overall angular deflection of unwanted particles', fontsize=20)
    ax.minorticks_on()
    ax.grid()
    if highlight_p != None:
        ax.axvline(highlight_p, color='k', lw=1, label=f'{highlight_p} GeV')
        text1 = 'D{'
        text1 += p2.name
        text1 += '} = '
        text1 += str(defl2[np.argmin(abs(p_range-highlight_p))])
        text1 += ' mrad and '
        text2 = 'D{'
        text2 += p3.name
        text2 += '} = '
        text2 += str(defl3[np.argmin(abs(p_range-highlight_p))])
        text2 += ' mrad'
        ax.legend(fontsize=20)
        plt.show()
        fig.savefig('Ang_Def_Plot_Opt_75.pdf', bbox_inches='tight')
        return text1 + text2
    ax.legend(fontsize=20)
    plt.show()
    return

def abs_deflection(D, x):
    """
    Calculates the transverse displacement of a particle after a gap length x
    with an initial angular deflection D.
    
    Parameters
    ----------
    D : float
        Angular deflection of particle in mrad.
    x : float
        Gap length in m
        
    Returns
    -------
    abs_def : float
        Transverse displacement in mm.
    """
    return x*np.tan(D*1e-3)*1e3

def freq_defl_comp(fmax, p=75, p1=database['K+'], p2=database['pi+'],
                   p3=database['p+'], L_t=79.6, l=2.74, E=1e6, plot=True,
                   details=False, n=1000, delta_p=1.6e-2, fmin=0, just_pi=False):
    """
    Compares the angular deflection of particles as a function of electric
    field frequency.
    
    Parameters
    ----------
    fmax : float
        Upper limit of frequencies to try.
    p : float
        Momentum of particle beam.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    L_t : float
        Total length of beam separator.
    l : float
        Cavity length of beam separator.
    E : float
        Electric field strength
    plot : True/False
        States whether a graph is to be plotted.
    details : True/False
        States whether the individual particle deflections (True) or just
        the minimum at every frequency (False) is to be plotted
    n : int
        The number of frequencies to try
    delta_p : float
        Momentum spread of particle beam
    fmin : float
        Lower limit of frequencies to try
    just_pi : True/False
        States whether or not to consider only the pion deflection and
        ignore the protons.
        
    Returns
    -------
    opt_freq : float
        Optimum frequency at which  minimum deflection is maximised.
    graph : graph
        Graph showing angular deflections.
    """
    f_range = np.linspace(fmin, fmax, n)
    min_defl, d2, d3 = [], [], []
    for f in f_range:
        if just_pi == True:
            d2.append(ang_deflection(p, f, p1, p2, L_t, l, E, delta_p))
            min_defl.append(d2[-1])
        if just_pi == False:    
            d2.append(ang_deflection(p, f, p1, p2, L_t, l, E, delta_p))
            d3.append(ang_deflection(p, f, p1, p3, L_t, l, E, delta_p))
            min_defl.append(np.min([d2[-1], d3[-1]]))
    if plot == True:
        fig = plt.figure(figsize=[9, 5])
        ax = fig.add_subplot(1, 1, 1)
        if details == True:
            ax.plot(f_range, d2, 'r', alpha=0.5, label=f'{p2.name} deflection')
            ax.plot(f_range, d3, 'b', alpha=0.5, label=f'{p3.name} deflection')
        else:
            ax.plot(f_range, min_defl, color=[1, 0, 1], alpha=0.5, label='min deflection')
#        ax.set_xlim(fmin, fmax)
        ax.set_ylim(0)
        ax.set_xlabel('Frequency / Hz', fontsize=20)
        text = r'$D_{'
        text += p2.name + '/' + p3.name
        text += '}$ / mrad'
        ax.set_ylabel(text, fontsize=20)
        ax.set_xscale('log')
        ax.legend(fontsize=20)
        plt.show()
    opt_freq = f_range[np.argmax(min_defl)]
    d_opt = (fmax-fmin)/1000
    d_opt_perc = (d_opt/opt_freq)*100
    if details == True:
        return 'optimum frequency = {0:.0f} +/- {1:.5f}%'.format(opt_freq, d_opt_perc)
    return opt_freq

def cavity_length(lmax=79.6/2, L_t=79.6, p=75, fmax=1e12, p1=database['K+'],
                  p2=database['pi+'], p3=database['p+'], E=1e6,
                  plot=True, nf=200, delta_p=1.6e-2, n=100, just_pi=False,
                  set_freq=5.7e9):
    """
    Compares the angular deflection of particles as a function of cavity
    lengths.
    
    Parameters
    ----------
    lmax : float
        Upper limit of cavity lengths to try.
    L_t : float
        Total length of beam separator.
    p : float
        Momentum of particle beam.
    fmax : float
        Upper limit of frequencies to try if set_freq is None.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    E : float
        Electric field strength
    plot : True/False
        States whether a graph is to be plotted.
    nf : int
        The number of frequencies to try if set_freq is None.
    delta_p : float
        Momentum spread of particle beam
    n : int
        The number of cavity lengths to try.
    just_pi : True/False
        States whether or not to consider only the pion deflection and
        ignore the protons.
    set_freq : float
        If preset frequency is to be used then it is specified here, None
        is not.
        
    Returns
    -------
    opt_freq_l_defl : list
        [frequency, cavity length, deflection] at optimum values of f and l
    graph : graph
        Graph showing angular deflections.
    """
    l_range = np.linspace(0, lmax, n)[1:-1]
    min_defl, freq = [], []
    for l in l_range:
        if set_freq == None:
            f = freq_defl_comp(fmax, p, p1, p2, p3, L_t, l, E, plot=False,
                               details=False, n=nf, delta_p=delta_p, fmin=0,
                               just_pi=just_pi)
        else:
            f = set_freq
        freq.append(f)
        if just_pi == True:
            min_defl.append(ang_deflection(p, f, p1, p2, L_t, l, E, delta_p=delta_p))
        if just_pi == False:    
            d2 = ang_deflection(p, f, p1, p2, L_t, l, E, delta_p=delta_p)
            d3 = ang_deflection(p, f, p1, p3, L_t, l, E, delta_p=delta_p)
            min_defl.append(np.min([d2, d3]))
#    freq = np.array(freq)
    min_defl = np.array(min_defl)
#    ratio = min_defl/freq
#    ratio *= np.max(freq)/np.max(ratio)
    opt_freq_l_index = np.argmax(min_defl)
    opt_freq_l_defl = [freq[opt_freq_l_index], l_range[opt_freq_l_index], min_defl[opt_freq_l_index]]
#    opt_l = l_range[np.argmax(min_defl)]
#    d_opt = (L_t-(2*l))/1000
#    d_opt_perc = (d_opt/opt_l)*100
    if plot == True:    
        fig = plt.figure(figsize=[9, 5])
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(l_range, min_defl, 'b', alpha=0.5, label=f'minimum deflection')
#        ax2 = ax1.twinx()
#        line2 = ax2.plot(l_range, freq, 'b', alpha=0.5, label=f'optimum frequency')
#        line3 = ax2.plot(l_range, ratio, 'g', alpha=0.5, label=f'ratio')
        ax1.set_xlabel('Cavity Length / m', fontsize=20)
        ax1.set_xlim(0, lmax)
#        text = r'Minimum $D_{'
#        text += p2.name[:-1] + '/' + p3.name[:-1]
#        text += '}$ / mrad'
        ax1.set_ylabel('Minimum $D_{\pi^+/p^+}$ / mrad', fontsize=20, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
#        ax2.set_ylabel('Frequency / Hz', fontsize=20, color='b', alpha=1)
#        ax2.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(0)
#        ax2.set_ylim(0)
#        leg = line1 + line2 + line3
#        labs = [l.get_label() for l in leg]
#        ax1.legend(leg, labs, loc=0, fontsize=12)
#        ax1.axvline(l_range[opt_freq_l_index], lw=2)
#        output = 'optimum gap length = {0:.2f} m +/- {1:.3f}%\n'.format(opt_l, d_opt_perc)
#        output += 'optimum frequency/gap length = {0:.1f} GHz / {1:.1f} m'.format(opt_freq_l_defl[0]*1e-9,
#                                                                                  opt_freq_l_defl[1])
        ax1.minorticks_on()
        ax1.grid()
        fig.tight_layout()
        plt.show()
    return opt_freq_l_defl
    
    
def gap_length(L_t=79.6, p=75, fmax=1e12, p1=database['K+'],
               p2=database['pi+'], p3=database['p+'], l=2.74,
               E=1e6, plot=True, nf=200, delta_p=1.6e-2, n=100,
               just_pi=False, set_freq=5.7e9):
    """
    Compares the angular deflection of particles as a function of gap
    length.
    
    Parameters
    ----------
    L_t : float
        Total length of beam separator.
    p : float
        Momentum of particle beam.
    fmax : float
        Upper limit of frequencies to try if set_freq is None.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    l : float
        Cavity length of beam separator.
    E : float
        Electric field strength
    plot : True/False
        States whether a graph is to be plotted.
    nf : int
        The number of frequencies to try if set_freq is None.
    delta_p : float
        Momentum spread of particle beam
    n : int
        The number of gap lengths to try.
    just_pi : True/False
        States whether or not to consider only the pion deflection and
        ignore the protons.
    set_freq : float
        If preset frequency is to be used then it is specified here, None
        is not.
        
    Returns
    -------
    opt_freq_gap_disp : list
        [frequency, gap length, transverse displacement] at optimum values of f and g
    graph : graph
        Graph showing transverse displacements.
    """
    gap_length = np.linspace(0, L_t-(2*l), n)[:-1]
    min_disp, freq = [], []
    for g in gap_length:
        if set_freq == None:
            f = freq_defl_comp(fmax, p, p1, p2, p3, L_t, l, E, plot=False,
                               details=False, n=nf, delta_p=delta_p, fmin=0,
                               just_pi=just_pi)
        else:
            f = set_freq
        freq.append(f)
        if just_pi == True:
            d2 = ang_deflection(p, f, p1, p2, L_t-g, l, E, delta_p=delta_p)
            min_disp.append(abs_deflection(d2, g))
        if just_pi == False:    
            d2 = ang_deflection(p, f, p1, p2, L_t-g, l, E, delta_p=delta_p)
            d3 = ang_deflection(p, f, p1, p3, L_t-g, l, E, delta_p=delta_p)
            disp_2 = abs_deflection(d2, g)
            disp_3 = abs_deflection(d3, g)
            min_disp.append(np.min([disp_2, disp_3]))
    freq = np.array(freq)
    min_disp = np.array(min_disp)
    ratio = min_disp/freq
    ratio *= np.max(freq)/np.max(ratio)
    opt_freq_gap_index = np.argmax(ratio)
    opt_freq_gap_disp = [freq[opt_freq_gap_index], gap_length[opt_freq_gap_index], min_disp[opt_freq_gap_index]]
    if plot == True:    
        fig = plt.figure(figsize=[9, 5])
        ax1 = fig.add_subplot(1, 1, 1)
        line1 = ax1.plot(gap_length, min_disp, 'r', alpha=0.5, label=f'minimum displacement')
        ax2 = ax1.twinx()
        line2 = ax2.plot(gap_length, freq, 'b', alpha=0.5, label=f'optimum frequency')
        line3 = ax2.plot(gap_length, ratio, 'g', alpha=0.5, label=f'ratio')
        ax1.set_xlabel('Gap Length / m', fontsize=20)
        ax1.set_xlim(0, L_t-(2*l))
        text = r'Minimum $D_{'
        text += p2.name[:-1] + '/' + p3.name[:-1]
        text += '}$ / mm'
        ax1.set_ylabel(text, fontsize=20, color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax2.set_ylabel('Frequency / Hz', fontsize=20, color='b', alpha=1)
        ax2.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(0)
        ax2.set_ylim(0)
        leg = line1 + line2 + line3
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs, loc=0, fontsize=12)
        fig.tight_layout()
        plt.show()
    return opt_freq_gap_disp

def cavity_gap_comp(L_t=79.6, p=75, fmax=1e12, p1=database['K+'],
                    p2=database['pi+'], p3=database['p+'], E=1e6,
                    plot=True, delta_p=1.6e-2, n=100, nf=200, nl=100,
                    just_pi=False, count=True, set_freq=5.7e9):
    """
    Compares the angular deflection of particles as a function of gap
    length for optimised cavity length.
    
    Parameters
    ----------
    L_t : float
        Total length of beam separator.
    p : float
        Momentum of particle beam.
    fmax : float
        Upper limit of frequencies to try if set_freq is None.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    E : float
        Electric field strength
    plot : True/False
        States whether a graph is to be plotted.
    delta_p : float
        Momentum spread of particle beam
    n : int
        The number of gap lengths to try.
    nf : int
        The number of frequencies to try if set_freq is None.
    nl : int
        The number of cavity lengths to try.
    just_pi : True/False
        States whether or not to consider only the pion deflection and
        ignore the protons.
    count : True/False
        If True will print each value of g as it is tested to show progress.
    set_freq : float
        If preset frequency is to be used then it is specified here, None
        is not.
        
    Returns
    -------
    opt_freq_gap_disp : list
        [frequency, gap length, cavity length, transverse displacement] at
        optimum values of f, l and g
    graph : graph
        Graph showing transverse displacements.
    """
    gap_length = np.linspace(0, L_t, n)[:-1]
    min_disp, freq, cavity = [], [], []
    for g in gap_length:
        cav = cavity_length((L_t-g)/2, L_t-g, p, fmax, p1, p2, p3, E, False,
                            nf, delta_p, nl, just_pi, set_freq)
        l = cav[1]
        f = cav[0]
        cavity.append(l)
        freq.append(f)
        if just_pi == True:
            d2 = ang_deflection(p, f, p1, p2, L_t-g, l, E, delta_p=delta_p)
            min_disp.append(abs_deflection(d2, g))
            if count == True:
                print(g)
        if just_pi == False:
            d2 = ang_deflection(p, f, p1, p2, L_t-g, l, E, delta_p=delta_p)
            d3 = ang_deflection(p, f, p1, p3, L_t-g, l, E, delta_p=delta_p)
            disp_2 = abs_deflection(d2, g)
            disp_3 = abs_deflection(d3, g)
            min_disp.append(np.min([disp_2, disp_3]))
            if count == True:
                print(g)
    freq = np.array(freq)
    min_disp = np.array(min_disp)
    ratio = min_disp/freq
    ratio *= np.max(freq)/np.max(ratio)
    opt_freq_gap_index = np.argmax(ratio)
    opt_freq_gap_disp = [freq[opt_freq_gap_index], gap_length[opt_freq_gap_index],
                         cavity[opt_freq_gap_index], min_disp[opt_freq_gap_index]]
    opt_gap = gap_length[np.argmax(min_disp)]
    d_opt = (L_t-(2*l))/1000
    d_opt_perc = (d_opt/opt_gap)*100
    if plot == True:    
        fig = plt.figure(figsize=[12, 4])
        ax1 = fig.add_subplot(1, 1, 1)
#        line1 = ax1.plot(gap_length, min_disp, 'r', alpha=0.5, label='minimum displacement')
        ax1.plot(gap_length, min_disp, 'b', label=f'p = {p} GeV')
#        ax2 = ax1.twinx()
#        line2 = ax2.plot(gap_length, freq, 'b', alpha=0.5, label=f'optimum frequency')
#        line3 = ax2.plot(gap_length, ratio, 'g', alpha=0.5, label=f'ratio')
        ax1.set_xlabel('Gap Length / m', fontsize=20)
        ax1.set_xlim(0, L_t-(2*l))
#        text = r'Minimum $D_{'
#        text += p2.name[:-1] + '/' + p3.name[:-1]
#        text += '}$ / mm'
#        text = r'Minimum $D_{\pi^+ / p^+}$ / mm'
        text = 'Minimum Displacement / mm'
        ax1.set_ylabel(text, fontsize=15, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
#        ax2.set_ylabel('Frequency / Hz', fontsize=20, color='b', alpha=1)
#        ax2.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(0)
#        ax2.set_ylim(0)
#        leg = line1
#        leg = line1 + line2 + line3
#        labs = [l.get_label() for l in leg]
#        ax1.legend(leg, labs, loc=0, fontsize=12)
#        ax1.axvline(gap_length[opt_freq_gap_index], lw=2)
#        if set_freq != None:
#            g_opt = L_t/3
#            l_opt = L_t/6
#            d2_opt = ang_deflection(p, set_freq, p1, p2, L_t-g_opt, l_opt, E, delta_p=delta_p)
#            d3_opt = ang_deflection(p, set_freq, p1, p3, L_t-g_opt, l_opt, E, delta_p=delta_p)
#            disp_2_opt = abs_deflection(d2_opt, g_opt)
#            disp_3_opt = abs_deflection(d3_opt, g_opt)
#            min_disp_opt = np.min([disp_2_opt, disp_3_opt])
#            print(min_disp_opt)
#            ax1.scatter(g_opt, min_disp_opt, s=20, color='k')
#        ax1.legend(fontsize=20)
        output = 'optimum gap length = {0:.2f} m +/- {1:.3f}%\n'.format(opt_gap, d_opt_perc)
        output += 'optimum frequency/gap length = {0:.1f} GHz / {1:.1f} m'.format(opt_freq_gap_disp[0]*1e-9,
                                                                                  opt_freq_gap_disp[1])
        ax1.set_title('Minimum  unwanted  particle  displacement  as  a  function  of  gap  length', fontsize=18)
        fig.tight_layout()
        ax1.minorticks_on()
        ax1.grid()
        plt.show()
        print(output)
        fig.savefig('Opt_Gap_Cav_Comp_75.pdf', bbox_inches='tight')
    return opt_freq_gap_disp

def decay_proportion(L=102.4, p1=database['K+'], p=75, target_rate=53957518.001):
    """
    Calculates the proportion of particles p1 that will decay over a distance
    L.
    
    Parameters
    ----------
    L : float
        Distance travelled.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p : float
        Momentum of particle beam.
    target_rate : float
        Rate of particles at the start of the distance.
        
    Returns
    -------
    dec_prop : float
        If target_rate is None this returns the decimal proportion of the
        initial particles, otherwise returns the absolute rate at a distance
        L for initial rate target_rate.
    """
    tau = p1.tau*1e-3/c
    if target_rate == None:
        return np.exp(-(((L*p1.mass)/(p*c))/tau))
    return target_rate*np.exp(-(((L*p1.mass)/(p*c))/tau))

def decay_proportion_plot(Lmax=1000, p1=database['K+'], p=75, target_rate=53957518.001):
    """
    Plots the proportion of particles p1 that will decay over a distance
    L as a function of L
    
    Parameters
    ----------
    Lmax : float
        Upper limit of plotted distance travelled.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p : float
        Momentum of particle beam.
    target_rate : float
        Rate of particles at the start of the distance.
        
    Returns
    -------
    graph : graph
        Graph showing decay proportions.
    """
    L_range = np.linspace(0, 1000, 10000)
    prop = []
    for L in L_range:
        prop.append(decay_proportion(L, p1, p, target_rate))
#    charac_L = p*c*(p1.tau*1e-3/c)/p1.mass
    fig = plt.figure(figsize=[12, 3])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(L_range, prop, 'r', lw=2)
    ax.set_xlim(0, Lmax)
    ax.set_ylim(0)
    ax.set_xlabel('Target Distance', fontsize=20)
    ax.set_ylabel(r'$K^+$ flux', fontsize=20)
#    ax.xaxis.set_major_locator(plt.MultipleLocator(charac_L/4))
#    ax.xaxis.set_minor_locator(plt.MultipleLocator(charac_L/20))
#    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter_non_int(1, charac_L, 'L_{K^+}')))
    ax.set_xticks([0])
    ax.set_yticks([target_rate])
    ax.yaxis.set_major_locator(plt.MultipleLocator(target_rate/1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(target_rate/1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter_non_int(1, target_rate, 'R_t')))
    ax.legend(fontsize=20)
    ax.minorticks_off()
#    ax.grid()
    plt.show()
    return

def decay_rate(L=102.4, p1=database['K+'], p=75, target_rate=53957518.001,
               decay_region=65):
    """
    Calculates the decay rate of particles p1 over a distance decay_region
    at a distance L from a point at which their rate is target_rate.
    
    Parameters
    ----------
    L : float
        Distance to decay region.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p : float
        Momentum of particle beam.
    target_rate : float
        Rate of particles at the start of the distance.
    decay_region : float
        Length of decay region.
        
    Returns
    -------
    dec_rate : float
        Decay rate in the decay region.
    """
    tau = p1.tau*1e-3/c
    decay_region_time = (decay_region*p1.mass)/(p*c)
    dec_reg_start_rate = target_rate*(np.exp(-(((L*p1.mass)/(p*c))/tau)))
    return dec_reg_start_rate*(1-np.exp(-(decay_region_time/tau)))

def target_rate(L=102.4, p1=database['K+'], p=75, dec_reg_rate=19591295,
                decay_region=65):
    """
    Calculates the required target rate of particles p1 such that the decay
    rate over a distance decay_region at a distance L from the target is
    equal to dec_reg_rate.
    
    Parameters
    ----------
    L : float
        Distance to decay region.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p : float
        Momentum of particle beam.
    dec_reg_rate : float
        Required decay rate of particles in the decay region.
    decay_region : float
        Length of decay region.
        
    Returns
    -------
    target_rate : float
        Required particle rate at the target.
    """
    tau = p1.tau*1e-3/c
    decay_region_time = (decay_region*p1.mass)/(p*c)
    dec_reg_start_rate = dec_reg_rate/(1-np.exp(-(decay_region_time/tau)))
    return dec_reg_start_rate/(np.exp(-(((L*p1.mass)/(p*c))/tau)))

def decay_rate_vs_p(pmax=100, L=102.4, p1=database['K+'],
                    target_rate=53957518.001, decay_region=65):
    """
    Plots the particle decay rate in the decay region as a function of
    momentum.
    
    Parameters
    ----------
    pmax : float
        Upper limit on plotted momentum of particle beam.
    L : float
        Distance to decay region.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    target_rate : float
        Particle rate at the start of the distance.
    decay_region : float
        Length of decay region.
        
    Returns
    -------
    output : string
        States momentum at which maximum decay rate occurs and compares to
        decay rate at p = 75 GeV.
    graph : graph
        Graph showing decay rate.
    """
    p_range = np.linspace(1, 100, 1000)
    rate = []
    for p in p_range:
        rate.append(decay_rate(L, p1, p, target_rate, decay_region)*1e-6)
    fig = plt.figure(figsize=[12, 4])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p_range, rate, 'r', lw=2)
    ax.set_xlim(0, pmax)
    ax.set_ylim(0)
    ax.set_xlabel('p / GeV', fontsize=20)
    text = f'{p1.name} decays in decay region / MHz'
    ax.set_ylabel(text, fontsize=15)
    ax.axvline(75, color='k', lw=2, label='current setup')
    ax.legend(fontsize=20)
    ax.minorticks_on()
    ax.grid()
    plt.show()
    output = 'Maximum decays at p = {0:.1f} GeV --> decay rate = {1:.2f} MHz \n'.format(p_range[np.argmax(rate)], np.max(rate))
    output += 'Current decays at p = 75 GeV --> decay rate = {0:.2f} MHz'.format(decay_rate(p=75)*1e-6)
    print(output)
    return

def decay_rate_plot(Lmax=1000, p1=database['K+'], p=75,
                    target_rate=53957518.001, decay_region=65):
    """
    Plots the particle decay rate in the decay region as a function of
    distance.
    
    Parameters
    ----------
    Lmax : float
        Upper limit on plotted momentum of particle beam.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p : float
        Momentum of particle beam.
    target_rate : float
        Particle rate at the start of the distance.
    decay_region : float
        Length of decay region.
        
    Returns
    -------
    graph : graph
        Graph showing decay rate at specified target_rate and 4 times that
        intensity.
    """
    L_range = np.linspace(0, Lmax, 1000)
    rates, new_rates = [], []
    for L in L_range:
        rates.append(decay_rate(L, p1, p, target_rate, decay_region)*1e-6)
        new_rates.append(decay_rate(L, p1, p, target_rate*4, decay_region)*1e-6)
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(L_range, rates, 'g', lw=2, label='current beam intensity')
    ax.plot(L_range, new_rates, 'b', lw=2, label='proposed beam intensity')
    ax.set_xlim(0, Lmax)
    ax.set_ylim(0)
    ax.set_xlabel('Distance between target and decay region / m', fontsize=20)
    text = f'{p1.name} decays in decay region / MHz'
    ax.set_ylabel(text, fontsize=20)
    ax.axvline(102.4, color='k', lw=2, label='current setup')
    ax.legend(fontsize=20)
    plt.show()
    return

def length_vs_rate_2(Lmax=200, Lmin=102.4, p=75, fmax=1e12, p1=database['K+'],
                   p2=database['pi+'], p3=database['p+'], E=1e6, n=20,
                   ng=50, nl=50, nf=100, delta_p=1.6e-2, just_pi=False,
                   set_freq=5.7e9, count=True):
    """
    Plots transverse displacement and decay rate of particle p1 in the decay
    region as a function of the target to decay region distance for optimised
    g, l and f.
    
    Parameters
    ----------
    Lmax : float
        Upper limit on target to decay region distance.
    Lmin : float
        Lower limit on target to decay region distance.
    p : float
        Momentum of particle beam.
    fmax : float
        Upper limit of frequencies to try if set_freq is None.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    E : float
        Electric field strength
    n : int
        The number of target to decay region distances to try.
    ng : int
        The number of gap lengths to try.
    nl : int
        The number of cavity lengths to try.
    nf : int
        The number of frequencies to try if set_freq is None.
    delta_p : float
        Momentum spread of particle beam
    just_pi : True/False
        States whether or not to consider only the pion deflection and
        ignore the protons.
    set_freq : float
        If preset frequency is to be used then it is specified here, None
        is not.
    count : True/False
        If True will print each value of L as it is tested to show progress.
        
    Returns
    -------
    opt_disp_L_gap_cav_freq_rate : list
        [Transverse displacement, target distance, gap length, cavity length,
        frequency, decay rate] at which displacement is maximised.
    graph : graph
        Graph showing transverse displacement and decay rate.
    """
    if set_freq == None:
        print('Estimated time: {0}'.format(timing(n*ng*nl*nf*3.84e-05)))
    else:
        print('Estimated time: {0}'.format(timing(n*ng*nl*3.84e-05)))
    i = 0
    while i < 1e7:
        i += 1
    start = time.time()
    L_range = np.linspace(Lmin, Lmax, n)
    opt_gap, opt_freq, opt_cav, opt_disp, rate = [], [], [], [], []
    for L in L_range:
        if count == True:
            print(f'L={L}')
        opt = cavity_gap_comp(L_t=L-22.8, p=p, fmax=fmax, p1=p1, p2=p2, p3=p3,
                              E=E, plot=False, delta_p=delta_p, n=ng, nf=nf,
                              nl=nl, just_pi=just_pi, set_freq=set_freq,
                              count=count)
        opt_freq.append(opt[0])
        opt_gap.append(opt[1])
        opt_cav.append(opt[2])
        opt_disp.append(opt[3])
        rate.append(decay_rate(L, p1, p)*1e-6)
    opt_freq, opt_disp, rate = np.array(opt_freq), np.array(opt_disp), np.array(rate)
#    ratio = (opt_disp*rate)/(opt_freq*L)
#    ratio *= (np.max(opt_freq)-np.min(opt_freq))/(np.max(ratio)-np.min(ratio))
#    ratio += ((np.min(opt_freq))-np.min(ratio))
    fig = plt.figure(figsize=[14, 4])
    host = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(right=0.75)
    par1 = host.twinx()
#    par2 = host.twinx()
#    par2.spines['right'].set_position(('axes', 1.2))
#    make_patch_spines_invisible(par2)
#    par2.spines['right'].set_visible(True)    
    p1, = host.plot(L_range, rate, 'r', alpha=0.8, lw=2, label='decay rate in decay region')
    p2, = par1.plot(L_range, opt_disp, 'b', alpha=0.8, lw=2, label='min disp')
#    p3, = par2.plot(L_range, opt_freq, 'b', alpha=0.8, lw=2, label='frequency')
#    p4, = par2.plot(L_range, ratio, 'g', alpha=0.8, lw=2, label='ratio')
    host.set_xlim(Lmin, Lmax)
    host.set_xlabel('Distance between Target and Decay Region / m', fontsize=15)
    host.set_ylabel('Decay Rate in Decay Region / MHz', fontsize=15)
    par1.set_ylabel('Minimum Displacement / mm', fontsize=15)
#    par2.set_ylabel('Frequency / Hz', fontsize=15)
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
#    par2.yaxis.label.set_color(p3.get_color())
    host.tick_params(axis='y', colors=p1.get_color())
    par1.tick_params(axis='y', colors=p2.get_color())
#    par2.tick_params(axis='y', colors=p3.get_color())
#    lines = [p1, p2, p3, p4]
#    lines = [p1, p2]
#    host.legend(lines, [l.get_label() for l in lines], loc=[0.45, 0.8], fontsize=20)
    host.minorticks_on()
    par1.minorticks_on()
    host.set_ylim(0)
    par1.set_ylim(0)
    host.grid()
    host_yticks = np.arange(0, int(np.max(rate))+2, 1)
    par1_yticks = np.arange(0, (int(np.max(opt_disp)/100)+1)*100, 100)
    host.set_yticks(host_yticks)
    host.set_yticklabels(host_yticks)
    par1.set_yticks(par1_yticks)
    par1.set_yticklabels(par1_yticks)
    host.set_title(r'Unwanted particle displacement and observed $K^+$'+' decay rate\nas a function of target distance', fontsize=15)
    plt.show()
    fig.savefig('Length_vs_Rate_75.pdf', bbox_inches='tight')
    index = np.argmax(opt_disp)
#    index = np.argmax(ratio)
    print(f'{timing(time.time()-start)}')
    return [opt_disp[index], L_range[index], opt_gap[index], opt_cav[index], opt_freq[index], rate[index]]

def efficiency(w, L_t=79.6, p=75, fmax=1e12, p1=database['K+'],
               p2=database['pi+'], p3=database['p+'], E=1e6, delta_p=1.6e-2,
               n=10000, ng=50, nl=50, nf=100, plot=True, set_freq=5.7e9):
    """
    Plots transverse displacements of particles p2 and p3 as a function of
    phase with respect to the electric field and calculates the outgoing beam
    purity.
    
    Parameters
    ----------
    w : float
        Radius of collimator hole.
    L_t : float
        Total length of beam separator.
    p : float
        Momentum of particle beam.
    fmax : float
        Upper limit of frequencies to try if set_freq is None.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    E : float
        Electric field strength
    delta_p : float
        Momentum spread of particle beam
    n : int
        The number of phases to plot across.
    ng : int
        The number of gap lengths to try.
    nl : int
        The number of cavity lengths to try.
    nf : int
        The number of frequencies to try if set_freq is None.
    plot : True/False
        States whether a graph is to be plotted.
    set_freq : float
        If preset frequency is to be used then it is specified here, None
        is not.
    
    Returns
    -------
    purity : float
        Outgoing beam purity (kaon percentage) as a percentage.
    graph : graph
        Graph showing transverse displacement as a function of phase for
        particles p2 and p3.
    """
    if L_t == 79.6 and p == 75 and fmax == 1e12:
        opt = [296.5e9, 25.73, 13.33, 9.21]
    if L_t == 641.4-22.8 and p == 75 and fmax == 5.7e9:
        opt = [5.7e9, 206.2, 104.1, 134.8]
    else:
        opt = cavity_gap_comp(L_t, p, fmax, p1, p2, p3, E, False, delta_p, ng, nl,
                              nf, False, False, set_freq)
    d_2 = abs_deflection(ang_deflection(p, opt[0], p1, p2, L_t-opt[1], opt[2], E, delta_p), opt[1])
    d_3 = abs_deflection(ang_deflection(p, opt[0], p1, p3, L_t-opt[1], opt[2], E, delta_p), opt[1])
    phase_range = np.linspace(0, 2*np.pi, n)
    disp_2, disp_3, count_2, count_3 = [], [], 0, 0
    for tau in phase_range:
        disp_2.append(-d_2*np.cos(tau))
        disp_3.append(-d_3*np.cos(tau))
        if abs(disp_2[-1]) > w:
            count_2 += 1
        if abs(disp_3[-1]) > w:
            count_3 += 1      
    num_K = 6*decay_proportion(L_t, p1, p, None)/Gaussian_integral(w/26.4, plot=False)
    purity = (num_K/(num_K+((70*(n-count_2))/n)+((23*(n-count_3))/n)))*100
    if plot == True:
        fig = plt.figure(figsize=[12, 4])
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(phase_range, disp_2, 'r', lw=2, label=r'$\pi^+$')
        ax.plot(phase_range, disp_3, 'b', lw=2, label=r'$p^+$')
        blank = np.zeros(n)
        ymin, ymax = np.min(disp_2 + disp_3)*1.1, np.max(disp_2 + disp_3)*1.1
        ax.fill_between(phase_range, blank+w, blank+ymax, color='k', alpha=0.5)
        ax.fill_between(phase_range, blank-w, blank+ymin, color='k', alpha=0.5)
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Phase', fontsize=20)
        ax.set_ylabel('Displacement', fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Phase dependence of unwanted particle absorbance in the beam stopper', fontsize=17)
        ax.legend(fontsize=20)
        plt.show()
        fig.savefig('Efficiency_Blank.pdf', bbox_inches='tight')
        print('{0} efficiency = {1:.1f}%\n{2} efficiency = {3:.1f}%\nBeam purity = {4:.1f}%'.format(p2.name, (count_2/n)*100, p3.name, (count_3/n)*100, purity))
    return purity

def efficiency_vs_length(w=50, Lmax=1000, Lmin=102.4, p=75, fmax=5.7e9,
                         p1=database['K+'], p2=database['pi+'],
                         p3=database['p+'], E=1e6, delta_p=1.6e-2, n=1000,
                         ng=50, nl=50, nf=100, resolution=0.01, plot=True,
                         set_freq=5.7e9, present=False, count_L=True):
    """
    Calculates required and outgoing beam purities as functions of target
    distance and calculates to a given resolution the minimum target distance
    at which they intersect. Also plots required proton intentsity as a
    function of target distance.
    
    Parameters
    ----------
    w : float
        Radius of collimator hole.
    Lmax : float
        Upper limit of plotted target to decay rate distance.
    Lmin : float
        Lower limit of plotted target to decay rate distance.
    p : float
        Momentum of particle beam.
    fmax : float
        Upper limit of frequencies to try if set_freq is None.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    E : float
        Electric field strength
    delta_p : float
        Momentum spread of particle beam
    n : int
        The number of target to decay region distances to try.
    ng : int
        The number of gap lengths to try.
    nl : int
        The number of cavity lengths to try.
    nf : int
        The number of frequencies to try if set_freq is None.
    resolution : float
        The minimum required resolution of the intersect point between
        required and outgoing beam purity.
    plot : True/False
        States whether a graph is to be plotted.
    set_freq : float
        If preset frequency is to be used then it is specified here, None
        is not.
    present : True/False
        If True will plot as a function of L with 10 times more data points.
    count_L : True/False
        If True will print each L that is tested to show progress.    
    
    Returns
    -------
    output : list
        [target distance, outgoing purity, required proton intensity]
    graph : graph
        Graph showing purities and intensity.
    """
    current_target_rate = 53957518
    if count_L == True:
        if resolution == None:
            if set_freq == None:
                est = timing(n*ng*nl*nf*4e-05)
            else:
                est = timing(n*ng*nl*4e-05)
            print('Estimated time: {0}'.format(est))
        else:
            num = int(round(np.log10(Lmax-Lmin))-1)-np.log10(resolution)+1
            if set_freq == None:
                est = timing(10*num*ng*nl*nf*4e-05)
            else:
                est = timing(10*num*ng*nl*4e-05)
            print('Estimated maximum time: {0}'.format(est))
        start = time.time()
    if resolution == None:
        L_range = np.linspace(Lmin, Lmax, n)
        eff, purity = [], []
        for L in L_range:
            if count_L == True:
                print(L)
            eff.append(efficiency(w, L-22.8, p, fmax, p1, p2, p3, E, delta_p,
                                  10000, ng, nl, nf, False, set_freq))
#            purity.append((target_rate(L, p1, p)/current_target_rate)*(6/Gaussian_integral(w/26.4, mu=0, sigma=26.4, A=1, y0=0, plot=False)))
            purity.append(100*decay_proportion(L-22.8, p1, p, target_rate=target_rate(L, p1, p))/(750e6*Gaussian_integral(w/26.4, plot=False)))
        i = 0
        while eff[i] < purity[i] and i < np.size(L_range)-1:
            i += 1
        if i >= np.size(L_range)-1:
            return None
        output = [L_range[i], eff[i], purity[i]/6]
    else:
        if present == True:
            n = 10**(int(round(np.log10(Lmax-Lmin))-2))
            n1 = 10**(int(round(np.log10(Lmax-Lmin))-2))
        else:
            n = 10**(int(round(np.log10(Lmax-Lmin))-1))
            n1 = 10**(int(round(np.log10(Lmax-Lmin))-1))
        Lmin = Lmax - ((int(((Lmax-Lmin)/n))+1)*n)
        Lmax = Lmin + ((int(((Lmax-Lmin)/n))+1)*n)
        while n >= resolution:
            if n == n1 and plot==True:
                graph_Lmin = Lmin
                graph_Lmax = Lmax - n
                graph_L_range = np.arange(Lmin, Lmax, n)
                graph_eff, graph_purity, graph_intensity = [], [], []
                for L in graph_L_range:
                    if count_L == True:
                        print(L)
                    graph_eff.append(efficiency(w, L-22.8, p, fmax, p1, p2,
                                                p3, E, delta_p, 10000, ng, nl,
                                                nf, False, set_freq))
#                    graph_purity.append((target_rate(L, p1, p)/current_target_rate)*(6/Gaussian_integral(w/26.4, mu=0, sigma=26.4, A=1, y0=0, plot=False)))
                    graph_purity.append(100*decay_proportion(L-22.8, p1, p, target_rate=target_rate(L, p1, p))/(750e6*Gaussian_integral(w/26.4, plot=False)))
                    graph_intensity.append((target_rate(L, p1, p)/current_target_rate)/Gaussian_integral(w/26.4, plot=False))
                i = 0
                while graph_eff[i] < graph_purity[i] and i < np.size(graph_L_range)-1:
                    i += 1
                if i >= np.size(graph_L_range)-1:
                    output = None
                    break
                Lmin = graph_L_range[i-1]
                n = 10**(int(np.log10(n)-1))
                Lmax = Lmin + ((int(((Lmax-Lmin)/n))+1)*n)
            else:
                L_range = np.arange(Lmin, Lmax, n)
                eff, purity, intensity = 0, 0, 0
                i = 0
                while eff <= purity and i < np.size(L_range)-1:
                    i += 1
                    L= L_range[i]
                    if count_L == True:
                        print(L)
                    eff = efficiency(w, L-22.8, p, fmax, p1, p2, p3, E, delta_p,
                                     10000, ng, nl, nf, False, set_freq)
#                    purity = (target_rate(L, p1, p)/current_target_rate)*(6/Gaussian_integral(w/26.4, mu=0, sigma=26.4, A=1, y0=0, plot=False))
                    purity = 100*decay_proportion(L-22.8, p1, p, target_rate=target_rate(L, p1, p))/(750e6*Gaussian_integral(w/26.4, plot=False))
                    intensity = (target_rate(L, p1, p)/current_target_rate)/Gaussian_integral(w/26.4, plot=False)
                if i >= np.size(L_range)-1:
                    return None
                output = [L_range[i], eff, intensity]
                Lmin = L_range[i-1]
                n = 10**(int(np.log10(n)-1))
                Lmax = Lmin + ((int(((Lmax-Lmin)/n))+1)*n)
    if plot == True:
        fig = plt.figure(figsize=[10, 5])
        ax = fig.add_subplot(1, 1, 1)
        ax2 = ax.twinx()
        eff, purity, intensity, L_range = graph_eff, graph_purity, graph_intensity, graph_L_range
        Lmax, Lmin = graph_Lmax, graph_Lmin
        line1, = ax.plot(L_range, eff, 'r', lw=2, label='Beam Purity', alpha=0.6)
        line2, = ax.plot(L_range, purity, 'b', lw=2, label='Required Purity', alpha=0.6)
        line3, = ax2.plot(L_range, intensity, 'g', lw=2, label='Req. Intensity', alpha=0.6)
#        ax.axhline(24, color='k', label='Kaon percentage = 24%')
        ax.set_xlim(Lmin, Lmax)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Distance between target and decay region / m', fontsize=20)
        ax.set_ylabel('Beam purity / %', fontsize=20)
        ax2.set_ylabel(r'Intensity / I$_0$', fontsize=20)
        ax2.set_ylim(0)
        ax2.yaxis.label.set_color(line3.get_color())
        ax2.tick_params(axis='y', colors=line3.get_color())
#        index = np.argmin(np.abs(eff-24))
#            if eff[index] >= 24:
#                min_L = L_range[index]
#            else:
#                min_L = L_range[index+1]
#        print(f'{timing(time.time()-start)}')
#        print('Minimum target distance required = {0:.1f} m'.format(min_L))
#        return min_L
        if output != None:
            line4 = ax.axvline(output[0], color='k', label='Intercept', dashes=[6, 2])
            lines = [line1, line2, line3, line4]
        else:
            lines = [line1, line2, line3]
        ax2.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=15)
        ax.minorticks_on()
        ax2.minorticks_on()
        ax.grid()
        ax.set_title('Observed and required beam purities as a function of\ntarget distance'+f' for E = {int(E*1e-6)} MV/m', fontsize=16)
        plt.show()
        fig.savefig(f'Purity_vs_Required_p_75_E_1e{int(round(np.log10(E)))}_with_Intensity.pdf', bbox_inches='tight')
    if count_L == True:
        print(f'{timing(time.time()-start)}')
    return output

def efficiency_vs_w(wmax, wmin=10, Lmax=1000, Lmin=102.4, p=75, fmax=5.7e9,
                    p1=database['K+'], p2=database['pi+'], p3=database['p+'],
                    E=1e6, delta_p=1.6e-2, nw=10, nL=10, ng=50, nl=50, nf=20,
                    resolution=0.01, w_resolution=0.01, plot=True,
                    set_freq=5.7e9, count_L=False, count_w=True):
    """
    Plots minimum target distance, required proton intensity and the
    reciprocal of the product of the two as a function of the collimator width
    w. Calculates the value of w at which L is minimised, the value of w at
    which required proton intensity is minimised and the value of w at which
    the reciprocal product is maximised.
    
    Parameters
    ----------
    wmax : float
        Upper limit of plotted radius of collimator hole.
    wmin : float
        Lower limit of plotted radius of collimator hole.
    Lmax : float
        Upper limit of target to decay rate distance to try.
    Lmin : float
        Lower limit of target to decay rate distance to try.
    p : float
        Momentum of particle beam.
    fmax : float
        Upper limit of frequencies to try if set_freq is None.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    E : float
        Electric field strength
    delta_p : float
        Momentum spread of particle beam
    nw : int
        The number of collimator widths to try.
    nL : int
        The number of target to decay region distances to try.
    ng : int
        The number of gap lengths to try.
    nl : int
        The number of cavity lengths to try.
    nf : int
        The number of frequencies to try if set_freq is None.
    resolution : float
        The minimum required resoluton of the intersect point between
        required and outgoing beam purity.
    w_resolution : float
        The minimum required resoluton of the value of w at which the
        reciprocal product is maximised.
    plot : True/False
        States whether a graph is to be plotted.
    set_freq : float
        If preset frequency is to be used then it is specified here, None
        is not.
    count_L : True/False
        If True will print each L that is tested to show progress.
    count_w : True/False
        If True will print each w that is tested to show progress.
    
    Returns
    -------
    output : list
        [collimator width, target distance, proton intensity] at which the
        reciprocal product is maximised.
    graph : graph
        Graph showing target distance and intensity.
    """
    if resolution == None:
        if set_freq == None:
            est = timing(nw*nL*ng*nl*nf*4e-05)
        else:
            est = timing(nw*nL*ng*nl*4e-05)
        print('Estimated total time: {0}'.format(est))
    else:
        num = int(round(np.log10(Lmax-Lmin))-1)-np.log10(resolution)+1
        if set_freq == None:
            est = timing(nw*10*num*ng*nl*nf*4e-05)
        else:
            est = timing(nw*10*num*ng*nl*4e-05)
        print('Estimated maximum total time: {0}'.format(est))
    start = time.time()
    if w_resolution == None:
        w_range = np.linspace(wmin, wmax, int(nw))
        plot_w, length, intensity = [], [], []
        for w in w_range:
            if count_w == True:
                if count_L == True:
                    print(f'w = {w} mm')
                else:
                    print(w)
            eff = efficiency_vs_length(w, Lmax, Lmin, p, fmax,p1, p2, p3, E,
                                       delta_p, nL, ng, nl, nf, resolution,
                                       False, set_freq, False, count_L)
            if eff != None:
                plot_w.append(w)
                length.append(eff[0])
                intensity.append(eff[2])
                file = open(f'Save_Data_efficiency_vs_w_{wmin}_{wmax}.txt','a')
                file.write(f'{[plot_w[-1], length[-1], intensity[-1]]}\n')
                file.close()
        length, intensity = np.array(length), np.array(intensity)
        ratio = 1/(length*intensity)
        ratio *= (np.max(intensity)-np.min(intensity))/(np.max(ratio)-np.min(ratio))
        ratio += ((np.min(intensity))-np.min(ratio))
        output = [plot_w[np.argmax(ratio)], plot_w[np.argmin(length)], plot_w[np.argmin(intensity)]]
    else:
        wmax = 10*(int(wmax/10)+2)
        wmin = 10*int(wmin/10)
        n = 10
        while n >= w_resolution:
            if n == 10 and plot == True:
                w_range = np.arange(wmin, wmax, n)
                graph_w, graph_length, graph_intensity, graph_ratio = [], [], [], []
                for w in w_range:
                    if count_w == True:
                        if count_L == True:
                            print(f'w = {w} mm')
                        else:
                            print(w)
                    eff = efficiency_vs_length(w, Lmax, Lmin, p, fmax,p1, p2, p3, E,
                                               delta_p, nL, ng, nl, nf, resolution,
                                               False, set_freq, False, count_L)
                    if eff != None:
                        graph_w.append(w)
                        graph_length.append(eff[0])
                        graph_intensity.append(eff[2])
                        graph_ratio.append(1/(eff[0]*eff[2]))
                i, d_ratio = 1, 0
                while d_ratio >= 0 and i < np.size(graph_ratio):
                    d_ratio = graph_ratio[i] - graph_ratio[i-1]
                    i += 1
                if i >= np.size(graph_ratio):
                    output = None
                    break
                graph_ratio = np.array(graph_ratio)
                graph_ratio *= (np.max(graph_intensity)-np.min(graph_intensity))/(np.max(graph_ratio)-np.min(graph_ratio))
                graph_ratio += ((np.min(graph_intensity))-np.min(graph_ratio))
                d = -int(np.log10(n))
                output = [round(graph_w[np.argmax(graph_ratio)], d),
                          graph_length[np.argmax(graph_ratio)],
                          graph_intensity[np.argmax(graph_ratio)]]
                wmax = graph_w[i-1]
                wmin = graph_w[i-3]
                n /= 10
            else:
                w_range = np.arange(wmin, wmax+n, n)
                plot_w, length, intensity, ratio = [], [], [], []
                i, d_ratio = 0, 0
                d = -int(np.log10(n))
                while d_ratio >= 0 and i < np.size(w_range):
                    if count_w == True:
                        if count_L == True:
                            print(f'w = {round(w_range[i], d)} mm')
                        else:
                            print(round(w_range[i], d))
                    eff = efficiency_vs_length(w_range[i], Lmax, Lmin, p, fmax,p1,
                                               p2, p3, E, delta_p, nL, ng, nl, nf,
                                               resolution, False, set_freq, False,
                                               count_L)
                    if eff != None:
                        plot_w.append(w_range[i])
                        length.append(eff[0])
                        intensity.append(eff[2])
                        ratio.append(1/(eff[0]*eff[2]))
                        if np.size(ratio) == 1:
                            d_ratio = ratio[-1]
                        else:
                            d_ratio = ratio[-1] - ratio[-2]
                    i += 1
                if i >= np.size(w_range):
                    output = None
                    break
                output = [round(plot_w[np.argmax(ratio)], d),
                          length[np.argmax(ratio)],
                          intensity[np.argmax(ratio)]]
                wmax = w_range[i-1]
                wmin = w_range[i-3]
                n /= 10
    if plot == True:
        if w_resolution != None:
            plot_w, length, intensity, ratio = graph_w, graph_length, graph_intensity, graph_ratio
        fig = plt.figure(figsize=[12, 4])
        ax = fig.add_subplot(1, 1, 1)
        line1, = ax.plot(plot_w, length, 'r', lw=2, alpha=0.6, label='Target Distance')
        ax2 = ax.twinx()
        line2, = ax2.plot(plot_w, intensity, 'g', lw=2, alpha=0.6, label='Intensity Required')
        line3, = ax2.plot(plot_w, ratio, 'b', lw=2, alpha=0.6, label='Reciprocal Product')
        ax.set_xlabel('Collimator Width / mm', fontsize=20)
        ax.set_ylabel('Target Distance / m', fontsize=20, color='r')
        ax2.set_ylabel(r'Intensity / I$_0$', fontsize=20, color='g')
        ax.set_xlim(np.min(plot_w), np.max(plot_w))
        ax.tick_params(axis='y', colors=line1.get_color())
        ax2.tick_params(axis='y', colors=line2.get_color())
#        ax.set_ylim(0)
        lines = [line1, line2, line3]
        ax.legend(lines, [l.get_label() for l in lines], loc=[0.70, 0.12], fontsize=15)
        ax.minorticks_on()
        ax2.minorticks_on()
        ax.grid()
        ax.set_title(' Minimum required target distance and proton intensity as a function of beam stopper inner radius', fontsize=16)
        plt.show()
    print(f'Total time: {timing(time.time()-start)}')
    return output

def plot_w_from_file(filename='Save_Data_efficiency_vs_w_8_80.txt'):
    """
    Plots efficiency_vs_w() from saved data in a text file.
    
    Parameters
    ----------
    filename : txt file
        File to import data from.
    
    Returns
    -------
    output : list
        [collimator width, target distance, proton intensity] at which the
        reciprocal product is maximised.
    graph : graph
        Graph showing target distance and intensity.
    """
    txt = open(filename)
    data = []
    for line in txt:
        line = line.strip()
        line = shlex.split(line)
        if len(line) > 0:
            data.append(line)
    plot_w, length, intensity = [], [], []
    for d in data:
        plot_w.append(float(d[0]))
        length.append(float(d[1]))
        intensity.append(float(d[2]))
    length, intensity = np.array(length), np.array(intensity)
    ratio = 1/(length*intensity)
    ratio *= (np.max(intensity)-np.min(intensity))/(np.max(ratio)-np.min(ratio))
    ratio += ((np.min(intensity))-np.min(ratio))
    fig = plt.figure(figsize=[12, 4])
    ax = fig.add_subplot(1, 1, 1)
    line1, = ax.plot(plot_w, length, 'r', lw=2, alpha=0.6, label='Target Distance')
    ax2 = ax.twinx()
    line2, = ax2.plot(plot_w, intensity, 'g', lw=2, alpha=0.6, label='Intensity Required')
    line3, = ax2.plot(plot_w, ratio, 'b', lw=2, alpha=0.6, label='Reciprocal Product (no scale)')
    ax.set_xlabel('Collimator Width / mm', fontsize=20)
    ax.set_ylabel('Target Distance / m', fontsize=20, color='r')
    ax2.set_ylabel(r'Intensity / I$_0$', fontsize=20, color='g')
    ax.set_xlim(np.min(plot_w), np.max(plot_w))
    ax.tick_params(axis='y', colors=line1.get_color())
    ax2.tick_params(axis='y', colors=line2.get_color())
#    ax.set_ylim(0)
    lines = [line1, line2, line3]
    ax2.legend(lines, [l.get_label() for l in lines], loc=[0.50, 0.39], fontsize=15)
    ax.minorticks_on()
    ax2.minorticks_on()
    ax.grid()
    ax.set_title('Minimum required target distance and proton intensity\nas a function of beam stopper inner radius', fontsize=16)
    plt.show()
    fig.savefig(f'Width_vs_Length_Intensity_p_75_E_1e7_Thin_Finer.pdf', bbox_inches='tight')
    return

def E_vs_length(Emax, Emin, wmax=90, wmin=10, Lmax=1000, Lmin=102.4, p=75,
                fmax=5.7e9, p1=database['K+'], p2=database['pi+'],
                p3=database['p+'], delta_p=1.6e-2, nE=10, nw=10, nL=10, ng=50,
                nl=50, nf=20, L_resolution=0.01, w_resolution=0.01, plot=True,
                set_freq=5.7e9, count_L=False, count_w=True, count_E=True):
    """
    Plots minimum target distance, required proton intensity and the optimum
    collimator width as a function of the electric field strength E.
    
    Parameters
    ----------
    Emax : float
        Upper limit of plotted electric field strength.
    Emin : float
        Lower limit of plotted electric field strength.
    wmax : float
        Upper limit of collimator hole to try.
    wmin : float
        Lower limit of collimator hole to try.
    Lmax : float
        Upper limit of target to decay rate distance to try.
    Lmin : float
        Lower limit of target to decay rate distance to try.
    p : float
        Momentum of particle beam.
    fmax : float
        Upper limit of frequencies to try if set_freq is None.
    p1 : ss2.ParticleData
        Particle Data of first particle.
    p2 : ss2.ParticleData
        Particle Data of second particle.
    p3 : ss2.ParticleData
        Particle Data of third particle.
    delta_p : float
        Momentum spread of particle beam
    nE : int
        The number of electric field strengths to try.
    nw : int
        The number of collimator widths to try.
    nL : int
        The number of target to decay region distances to try.
    ng : int
        The number of gap lengths to try.
    nl : int
        The number of cavity lengths to try.
    nf : int
        The number of frequencies to try if set_freq is None.
    L_resolution : float
        The minimum required resoluton of the intersect point between
        required and outgoing beam purity.
    w_resolution : float
        The minimum required resoluton of the value of w at which the
        reciprocal product is maximised.
    plot : True/False
        States whether a graph is to be plotted.
    set_freq : float
        If preset frequency is to be used then it is specified here, None
        is not.
    count_L : True/False
        If True will print each L that is tested to show progress.
    count_w : True/False
        If True will print each w that is tested to show progress.
    count_E : True/False
        If True will print each E that is tested to show progress.
    
    Returns
    -------
    output : list
        [collimator width, target distance, proton intensity] for every tested
        value of E
    graph : graph
        Graph showing target distance, intensity and collimator width.
    """
    E_range = np.logspace(np.log10(Emin), np.log10(Emax), int(nE))
    plot_E, w, length, intensity = [], [], [], []
    for E in E_range:
        if count_E == True:
            if count_w == True:
                print(f'E = {round(E, -3)} MV/m')
            else:
                print(E)
        output = efficiency_vs_w(wmax, wmin, Lmax, Lmin, p, fmax, p1, p2, p3,
                                 E, delta_p, nw, nL, ng, nl, nf, L_resolution,
                                 w_resolution, False, set_freq, count_L, count_w)
        if output != None:
            plot_E.append(E*1e-6)
            w.append(output[0])
            length.append(output[1])
            intensity.append(output[2])
            file = open(f'Save_Data_{Emin}_{Emax}.txt','a')
            file.write(f'{[plot_E[-1], w[-1], length[-1], intensity[-1]]}\n')
            file.close()
    if plot == True:
        fig = plt.figure(figsize=[10, 5])
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(right=0.75)
        line1, = ax.plot(plot_E, length, 'r', lw=2, label='Target Distance')
        ax2 = ax.twinx()
        line2, = ax2.plot(plot_E, intensity, 'g', lw=2, label='Intensity Required')
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.2))
        make_patch_spines_invisible(ax3)
        ax3.spines['right'].set_visible(True)
        line3, = ax3.plot(plot_E, w, 'b', lw=2, label='Collimator Width')
        ax.set_xlabel(r'Electric Field Strength / MVm$^{-1}$', fontsize=20)
        ax.set_xlim(np.min(plot_E), np.max(plot_E))
        ax.set_ylabel('Target Distance / m', fontsize=20, color=line1.get_color())
        ax2.set_ylabel(r'Intensity / I$_0$', fontsize=20, color=line2.get_color())
        ax3.set_ylabel('Collimator Width / mm', fontsize=20, color=line3.get_color())
        ax.tick_params(axis='y', colors=line1.get_color())
        ax2.tick_params(axis='y', colors=line2.get_color())
        ax3.tick_params(axis='y', colors=line3.get_color())
        lines = [line1, line2, line3]
        ax.legend(lines, [l.get_label() for l in lines], loc='upper center', fontsize=15)
        ax.set_xscale('log')
        ax.minorticks_on()
        ax2.minorticks_on()
        ax3.minorticks_on()
        ax.grid()
        plt.show()
    return [w, length, intensity]

def plot_from_file(filename='Save_Data.txt'):
    """
    Plots E_vs_length() from saved data in a text file.
    
    Parameters
    ----------
    filename : txt file
        File to import data from.
    
    Returns
    -------
    output : list
        [electric field strength, target distance, proton intensity] at which
        the reciprocal product of the three is maximised.
    graph : graph
        Graph showing target distance, intensity and collimator width.
    """
    txt = open(filename)
    data = []
    for line in txt:
        line = line.strip()
        line = shlex.split(line)
        if len(line) > 0:
            data.append(line)
    plot_E, w, length, intensity, reciprocal = [], [], [], [], []
    for d in data:
        plot_E.append(float(d[0])*1e-6)
        w.append(float(d[1]))
        length.append(float(d[2]))
        intensity.append(float(d[3]))
        reciprocal.append(1/(plot_E[-1]*length[-1]*intensity[-1]))
    reciprocal = np.array(reciprocal)
    reciprocal *= (np.max(intensity)-np.min(intensity))/(np.max(reciprocal)-np.min(reciprocal))
    reciprocal += ((np.min(intensity))-np.min(reciprocal))
    fig = plt.figure(figsize=[14, 4])
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(right=0.75)
    line1, = ax.plot(plot_E, length, 'r', lw=2, label='Target Distance')
    ax2 = ax.twinx()
    line2, = ax2.plot(plot_E, intensity, 'g', lw=2, label='Intensity Required')
#    line4, = ax2.plot(plot_E, reciprocal, 'k', lw=2, label='Reciprocal Product')
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('axes', 1.15))
    make_patch_spines_invisible(ax3)
    ax3.spines['right'].set_visible(True)
    line3, = ax3.plot(plot_E, w, 'b', lw=2, label='Collimator Width')
    ax.set_xlabel(r'Electric Field Strength / MVm$^{-1}$', fontsize=20)
    ax.set_xlim(np.min(plot_E), np.max(plot_E))
    ax.set_ylabel('Target Distance / m', fontsize=20, color=line1.get_color())
    ax2.set_ylabel(r'Intensity / I$_0$', fontsize=20, color=line2.get_color())
    ax3.set_ylabel('Collimator Width / mm', fontsize=20, color=line3.get_color())
    ax.tick_params(axis='y', colors=line1.get_color())
    ax2.tick_params(axis='y', colors=line2.get_color())
    ax3.tick_params(axis='y', colors=line3.get_color())
    lines = [line1, line2, line3]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper center', fontsize=15)
    ax.set_xscale('log')
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels([10, 100, 1000])
    ax.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax.grid()
    ax.set_title('Minimum required target distance and proton intensity and\noptimum collimator width as a function of electric field strength', fontsize=16)
    plt.show()
    fig.savefig(f'E_vs_Length_and_Intensity_Wide.pdf', bbox_inches='tight')
    index = np.argmax(reciprocal)
    return [plot_E[index], length[index], intensity[index]]

def Gaussian(x, mu=0, sigma=26.4, A=1, y0=0):
    """
    Calculates the Gaussian function of input x.
    
    Parameters
    ----------
    x : float or ndarray
        Input data to the function
    mu : float
        Mean value of Gaussian distribution
    sigma : float
        Standard deviation of Gaussian distribution
    A : float
        Amplitude of function
    y0 : float
        y-intercept of function
        
    Returns
    -------
    y : float or ndarray
        Output of the Gaussian function
    """
    #width = sigma*(2*np.sqrt(2*np.log(2)))
    b = 1/(sigma*np.sqrt(2*np.pi))
    f = b*np.power(np.e, -(((x-mu)**2)/(2*sigma**2)))
    return A*f + y0

def Gaussian_integral(n_sigma=1, mu=0, sigma=26.4, A=1, y0=0, plot=True):
    """
    Integrates the Gaussian function within a given number of standard devations.
    
    Parameters
    ----------
    n_sigma : float
        Number of standard devaitions to integrate across
    mu : float
        Mean value of Gaussian distribution
    sigma : float
        Standard deviation of Gaussian distribution
    A : float
        Amplitude of function
    y0 : float
        y-intercept of function
    Plot : True/False
        States whether a graph will be plotted
        
    Returns
    -------
    integral : float
        Integral between set ranges of Gaussian function
    """
    def Gaussian(x, mu=mu, sigma=sigma, A=A, y0=y0):
        b = 1/(sigma*np.sqrt(2*np.pi))
        f = b*np.power(np.e, -(((x-mu)**2)/(2*sigma**2)))
        return A*f + y0
    if plot == True:
        if n_sigma - int(n_sigma) >= 0.5:
            plot_n_sigma = int(n_sigma) + 1
        if n_sigma - int(n_sigma) < 0.5:
            plot_n_sigma = int(n_sigma)
        x = np.linspace(-(plot_n_sigma+2)*sigma, (plot_n_sigma+2)*sigma, 1000)
        y = Gaussian(x, mu, sigma, A, y0)
    int_range = np.linspace(-n_sigma*sigma, n_sigma*sigma, 1000)
    int_y = Gaussian(int_range, mu, sigma, A, y0)
    integral = scipy.integrate.quad(Gaussian, -n_sigma*sigma, n_sigma*sigma)
    if plot == True:
        fig = plt.figure(figsize=[12, 3])
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, 'r')
        ax.fill_between(int_range, 0, int_y, color='r', alpha=0.5)
        ax.set_xlabel('x', fontsize=20)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(0)
        ticks = np.arange(-(plot_n_sigma+2)*sigma, (plot_n_sigma+3)*sigma, sigma)
        tick_nums = np.arange(-(plot_n_sigma+2), plot_n_sigma+3, 1)
        ax.set_xticks(ticks)
        labels = []
        for i, label in enumerate(ax.get_xticklabels()):
            labels.append(f'{int(tick_nums[i])}' + r'$\sigma$')
        ax.set_xticklabels(labels)
        ax.set_yticks([])
        ax.spines['left'].set_position(('axes', 0.5))
        make_patch_spines_invisible(ax)
        ax.spines['bottom'].set_visible(True)
#        ax.spines['left'].set_visible(True)
#        plt.arrow(0, 0, 0, 0.015)
        plt.show()
    return integral[0]

def timing(t):
    """
    Formats a time in second into units up to years
    
    Parameters
    ----------
    t : float
        Number of seconds
    
    Returns
    -------
    formatted_time : string
        Formatted time.
    """
    if t < 1:
        power = int(np.log10(t))-1
        num = t/10**power
        if abs(power) < 10:
            return '{0:.1f}e-0{1:.0f}s'.format(num, abs(power))
        return '{0:.1f}e-{1:.0f}s'.format(num, abs(power))
    if t >= 1 and t < 60:
        return '{0:.1f}s'.format(t)
    if t >= 60 and t < 3600:
        minutes = int(t/60)
        seconds = t-(60*minutes)
        return '{0:.0f}m, {1:.1f}s'.format(minutes, seconds)
    if t >= 3600 and t < 86400:
        hours = int(t/3600)
        minutes = int((t-(3600*hours))/60)
        seconds = t-(3600*hours + 60*minutes)
        return '{0:.0f}h, {1:.0f}m, {2:.1f}s'.format(hours, minutes, seconds)
    if t >= 86400 and t < 31536000:
        days = int(t/86400)
        hours = int((t-(86400*days))/3600)
        minutes = int((t-(86400*days + 3600*hours))/60)
        seconds = t - (86400*days + 3600*hours + 60*minutes)
        return '{0:.0f}d, {1:.0f}h, {2:.0f}m, {3:.1f}s'.format(days, hours, minutes, seconds)
    if t >= 31536000:
        years = int(t/31536000)
        if years > 9999:
            years = t/31536000
            power = int(np.log10(years))
            num = years/10**power
            if abs(power) < 10:
                return '{0:.2f}e+0{1:.0f} years'.format(num, abs(power))
            return '{0:.2f}e+{1:.0f} years'.format(num, abs(power))
        days = int((t-(31536000*years))/86400)
        hours = int((t-(31536000*years + 86400*days))/3600)
        minutes = int((t-(31536000*years + 86400*days + 3600*hours))/60)
        seconds = t - (31536000*years + 86400*days + 3600*hours + 60*minutes)
        return '{0:.0f}y, {1:.0f}d, {2:.0f}h, {3:.0f}m, {4:.1f}s'.format(years, days, hours, minutes, seconds)
