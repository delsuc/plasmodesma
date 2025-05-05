#!/usr/bin/env python 
# encoding: utf-8

"""
Program to compute peak and bucket lists from a set of 1D and 2D

First version: Petar Markov & M-A Delsuc,  


v1: renamed to Plasmodesma, made it working   PM MAD
v2: added report(), many cosmetic improvements  MAD
v3: added DOSY processing, apmin, external parameters + many improvements   MAD
v4: added multiprocessing    MAD
v5: added bk_ftF1 and adapted to python3    MAD
v6: final version for publication           MAD
v6.3 : added parallelization of 1D and 2D Processing    MAD
v6.4 : corrected so that there might be no 1D or 2D to process    MAD
v7.0 : code for the Faraday 2019 paper - adapted to server - added reading from zip 
v7.1: Adapted to newer version of Spike, added 19F, changed user interface...
v7.2: local version, not published
v8.0 extension and clean-up (a little !) of the code

This code is associated to the publication:

"Automatic differential analysis of NMR experiments in complex samples."
Margueritte, L., Markov, P., Chiron, L., Starck, J.-P., Vonthron Sénécheau, C., Bourjot, M., & Delsuc, M.-A.
Magnetic Resonance in Chemistry, (2018) 80(5), 1387.
http://doi.org/10.1002/mrc.4683

and developped for the publication:
"Automatised pharmacophoric deconvolution of plant extracts application to cinchona bark crude extract"
Margueritte L., Duciel L., Bourjot M., Vonthron-Sénécheau C., Delsuc M-A.
Faraday Discussions (2019) 218, 441-458 
http://doi.org/10.1039/c8fd00242h

code is deposited in https://github.com/delsuc/plasmodesma

the prgm has been fully tested under

python 3.11
    with    numpy 1.11.1 - 1.26.0
            scipy 1.11.3 
"""

#---------------------------------------------------------------------------
#1. Imports; From global to specific modules in descending order
import sys
from glob import glob
import os.path as op
import os,json
import pprint
import tempfile
import zipfile as zip
import datetime
import re
try:
    import ConfigParser
except:
    import configparser as ConfigParser
import itertools
# added july 2018
try:
    from itertools import imap
except ImportError:
    # Python 3...
    imap = map
try:
    import copy_reg                 # python 2
    zip_longest = itertools.izip_longest
except:
    import copyreg as copy_reg      # python 3
    zip_longest = itertools.zip_longest
import types
import multiprocessing as mp

POOL = None      # will be overwritten by main()
import numpy as np
import matplotlib.pyplot as plt

VERSION = "8.0.4"

print ("**********************************************************************************")
print ("*                              PLASMODESMA program %5s                         *"%(VERSION,))
print ("*           - automatic advanced processing of NMR experiment series -           *")
print ("*                                                                                *")
print ("**********************************************************************************")

#---------------------------------------------------------------------------
#2. Parameters
# These can be changed to tune the program behavior
# These are default behaviours
# these values will be overloaded with the content of the RunConfig.json file if present

global Config
Config = {
    'NPROC' : 1,            # The default number of processors for calculation, if  value >1 will activate multiprocessing mode
                            # for best results keep it below your actual number of cores ! (MKL and hyperthreading !).
    'BC_ALGO' : 'Spline',   # baseline correction algo, either 'None', 'Coord', 'Spline' or 'Iterative'   
    'BC_ITER' : 5,          # Used by 'Iterative' baseline Correction; It is advisable to use a larger number for iterating, e.g. 5
    'BC_CHUNKSZ' : 1000,    # chunk size used by 'Iterative' baseline Correction,
    'BC_NPOINTS' : 8,       # number of pivot points used by automatic 'Spline' baseline Correction
    'BC_COORDS' : [],       # coordinates of pivot points in ppm used by 'Coord' baseline Correction
    'ROLLREM_N' : 6,        # control a baseline flattening applied on fid for 13C and 19F - not used if 0
    'TMS' : True,           # if true, TMS (or any 0 ppm reference) is supposed to be present and used for ppm calibration
    'LB_1H' : 1.0,          # exponential linebroadening in Hz used for 1D 1H 
    'LB_13C' : 3.0,         # exponential linebroadening in Hz used for 1D 13C
    'LB_19F' : 4.0,         # exponential linebroadening in Hz used for 1D 19F
    'MODUL_19F' : False,    # 19F are processed in modulus
    'SANERANK' : 20,        # used for denoising of 2D experiments, sane is an improved version of urQRd
                            # typically 10-50 form homo2D; 5-15 for HSQC, setting to 0 deactivates denoising
                            # takes time !  and time is proportional to SANERANK (hint more is not better !)
    'DOSY_LAZY' : False,    # if True, will not reprocess DOSY experiment if an already processed file is on the disk
    'PALMA_ITER' : 20000,   # used for processing of DOSY
    'BCK_1H_1D' : 0.01,     # bucket size for 1D 1H
    'BCK_1H_2D' : 0.03,     # bucket size for 2D 1H
    'BCK_1H_LIMITS' : [0.5, 9.5],   # limits of zone to  bucket and display in 1H
    'BCK_13C_LIMITS' : [-10, 150],  # limits of zone to  bucket and display in 13C
    'BCK_13C_1D' : 0.03,    # bucket size for 1D 13C
    'BCK_13C_2D' : 1.0,     # bucket size for 2D 13C
    'BCK_19F_LIMITS' : [-220, -40],  # limits of zone to  bucket and display in 19F
    'BCK_19F_1D' : 0.1,     # bucket size for 1D 19F
    'BCK_19F_2D' : 1.0,     # bucket size for 2D 19F
    'BCK_DOSY' : 0.1,       # bucket size for vertical axis of DOSY experiments
    'BCK_PP' : False,       # if True computes number of peaks per bucket (different from global peak-picking)
    'BCK_SK' : False,       # if True computes skewness and kurtosis over each bucket
    'TITLE': False,         # if true, the title file will be parsed for standard values (see documentation in Bruker_Report.py)
    'PNG': True,            # Figures of computed spectra are stored as PNG files
    'PDF': False,           # Figures of computed spectra are stored as PDF files
    'addpar': [],           # additional parameters for report.csv : eg ['D2', 'D12', 'P31']
    'add2Dpar': [],
    'addDOSYpar': []
}

global RunConfig
# RunConfig = {} | Config        # update internal RunConfig
RunConfig = {}
RunConfig.update(Config)        # update internal RunConfig

#---------------------------------------------------------------------------
#3. Utilities

def set_globalconfig(DIREC='.'):
    "prgm global parameters - loaded in Config{} "
    global Config
    print("Current directory: ", os.path.realpath(os.getcwd()))
    print("Working directory: ", DIREC)
    try:
        with open(op.join(DIREC,"RunConfig.json"),"r") as f:
            try:
                config = json.load(f)
            except:
                raise Exception('Error in reading Configuration file %s'%(op.join(DIREC,"RunConfig.json"),))
    except IOError:
        print('*** WARNING - no RunConfig.json file found - using default configuration')
    else:  # if no error
        for k in config.keys():
            if k not in Config.keys():
                print ("*** WARNING %s entry in RunConfig.json is not a standard entry"%k)
            Config[k] = config[k]
    #print('configuration:\n',Config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-D', '--Data', action='store', dest='DIREC', default=" .", help='DIRECTORY_with_NMR_experiments, default is "."')
    parser.add_argument('-N', action='store', dest='Nproc', default=Config['NPROC'], type=int, help='number of processors to use, default=%d'%Config['NPROC'])
    parser.add_argument('-n', '--dry',  action='store_true', help="list parameters, generate report.csv, but do not process")
    parser.add_argument('-T', '--template',  action='store_true',
                        help="Generate default config files templates (parameters.json RunConfig.json), implies --dry")
    args = parser.parse_args()

    set_globalconfig(args.DIREC)
    Config['NPROC'] = args.Nproc
    #RunConfig = {} | Config        # update internal RunConfig
    RunConfig = {}
    RunConfig.update(Config)
    print("params are:")
    #pprint.pprint(RunConfig)
    print(json.dumps(RunConfig, indent=4))

print ("Loading utilities ...")

import spike
import spike.NMR as npkd
import spike.File.BrukerNMR as bk
import spike.plugins.bcorr as bcorr
import spike.plugins.Peaks as Peaks
from spike.Algo.BC import correctbaseline # Necessary for the baseline correction
from spike.NPKData import as_cpx
from spike.util.signal_tools import findnoiselevel
from spike.Algo.Linpredic import baselinerollrem
from spike.v1 import Nucleus

import Bruker_Report



import ctypes
import platform
if platform.system() == 'Linux':
        mkl_rt = ctypes.CDLL('libmkl_rt.so')               # Linux !
        mkl_get_max_threads = mkl_rt.mkl_get_max_threads
elif platform.system() == 'Darwin':
        mkl_rt = ctypes.CDLL('libmkl_rt.dylib')               # MacOs !
        mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

def mkdir(f):
    "If a folder doesn't exist it is created"
    if not op.exists(f):
        os.makedirs(f)

def findnucleus(data):
    "preliminary"
    SF =  data.params['acqu']['$SFO1']
    if data.dim == 1:
        if ('564' in SF):
            nuc = '19F'
        elif ('600' in SF):
            nuc = '1H'
        elif ('150' in SF):
            nuc = '13C'
        else:
            raise Exception('Unknown nucleus')
    else:
        raise Exception('Reste à faire')
    return nuc

def findNuc(data):
    """
    estimates spin name and Bo from acqus file stored in parameters
    restrict to spin 1/2
    """
    try:
        acq = data.params['acqu']
    except:
        return (None, None)
    spinhalf = [k for (k,v) in Nucleus.table.items() if v[0] == 1]
    BFO1 = float(data.params['acqu']['$BF1'])     # acquisition spin
    F1H = 0                                     # search for Freq 1H
    for i in range(1,9):
        F1H = max(F1H,float(data.params['acqu']['$BF%d'%i]))   # assuming it is the largest of all freq 
    # search for closest freq
    mindist = F1H
    for s in spinhalf:
        Fs = Nucleus.freq(spin=s, H_freq=F1H)
        dist = abs(Fs-BFO1)
        if (dist<mindist):
            Spin = s
            mindist = dist
    # (_,  _, _, magnetogyricRatio, _, _) = Nucleus.table["1H"]
    # Bo = F1H *2* np.pi / (10* magnetogyricRatio)
    # return(Spin,Bo)
    return Spin
   
def FT1D(numb1, autoph=True):
    "Performs FT and corrections of experiment 'numb1' and returns data"
    def phase_from_param():
        "read the proc parame file, and returns phase parameters ok for spike"
        #print( proc['$PHC0'], proc['$PHC1'] )
        ph1 = -float( proc['$PHC1'] ) # First-order phase correction
        ph0 = -float( proc['$PHC0'] )+ph1/2 # Zero-order phase correction
        zero = -360*d.axis1.zerotime
        #print (ph0, ph1)
        return (ph0, ph1+zero)

    ph0 = RunConfig['ph0']
    ph1 = RunConfig['ph1']

    if numb1.endswith('fid'):
        d = bk.Import_1D(numb1)
    elif numb1.endswith('.gf1'):
        d = npkd.NMRData(name=numb1)
        acqu = bk.read_param( op.join(op.dirname(numb1),'acqus') )
        proc = bk.read_param( op.join(op.dirname(numb1),'pdata','1','procs') )
        d.axis1.zerotime = bk.zerotime(acqu)
        d.params = {"acqu": acqu, "proc": proc}   # add the parameters to the data-set
    else:
        print(f"**** WARNING, {numb1} file format unknow")
        raise Exception(f"**** WARNING, {numb1} file format unknow")

#    proc = d.params['procs']

    if findNuc(d) in ('13C'):   # First deal with  baseline roll for wide band spectra
        print("Experiment detected as 1D 13D")
        if d.axis1.specwidth > 20000:      # only if SW > 20kHz 
            d.center()
            if RunConfig['ROLLREM_N']>0:
                d = baselinerollrem(d, n=RunConfig['ROLLREM_N'])
            d.apod_em(RunConfig['LB_13C'],1).zf(2).ft_sim()
    if findNuc(d) == '19F':   # First 19F case
        if RunConfig['MODUL_19F']:      # modulus ? easy ! (but sloppy)
            print("Experiment detected as 1D 19F Modulus")
            d.center().apod_em(RunConfig['LB_19F'],1).zero_dsp(coeff=1.3).zf(2).ft_sim()
            d.modulus()
        else:
            if 'OPERA' in d.params['acqu']['$PULPROG']:    # OPERA acquisition is different
                print("Experiment detected as 1D 19F OPERA")
                d.center()
                if RunConfig['ROLLREM_N']>0:
                    d = baselinerollrem(d, n=RunConfig['ROLLREM_N'])
                d.apod_em(RunConfig['LB_19F'],1).zf(2).ft_sim()
                d.bk_pk().apmin()
            else:   # phasing non-opera is tricky !
                print("Experiment detected as 1D 19F")
                dd = d.copy().center().apod_em(RunConfig['LB_19F'],1).zero_dsp(coeff=1.3).zf(2).ft_sim()   # work on a copy
                dd.bk_pk()
                dd.apmin()
                p0,p1 = (dd.axis1.P0, dd.axis1.P1)    # copy apmin results
                d.center().apod_em(RunConfig['LB_19F'],1).zf(2).ft_sim().bk_pk().phase(p0,p1)
    else:
        print("Experiment detected as 1H")
        d.center().apod_em(RunConfig['LB_1H'],1).zf(2).ft_sim()
        if not autoph:
            p0,p1 = phase_from_param()
            d.phase( p0+ph0, p1+ph1 )   # Performs the stored phase correction
        else:
            d.bruker_corr().apmin()     # automatic phase correction
            d.phase( ph0, ph1 )   # Performs the additional phase correction from parameters.json
    d.unit = 'ppm'
    d.axis1.offset += RunConfig['ppm_offset']*d.axis1.frequency

    spec = np.real( d.get_buffer() )
    if RunConfig['BC_ALGO'] == 'Iterative':
        # the following is a bit convoluted baseline correction, 
        bl = correctbaseline(spec, iterations=RunConfig['BC_ITER'], nbchunks=d.size1//RunConfig['BC_CHUNKSZ'])
        dd = d.copy()
        dd.set_buffer(bl)
        dd.unit = 'ppm'
        d.real()
        d -= dd # Equal to d=d-dd; Used instead of (spec-bl)
    elif RunConfig['BC_ALGO'] == 'Coord':
        d.real().set_unit('ppm')
        BCcoors = RunConfig['BC_COORDS']
        if len(BCcoors) <4:
            d.bcorr(method='linear', nsmooth=200, xpunits='current', xpoints=BCcoors)
        else:
            d.bcorr(method='spline', nsmooth=200, xpunits='current', xpoints=BCcoors)
    elif RunConfig['BC_ALGO'] == 'Spline':
        d.real()
        d.bcorr(method='spline', xpoints=RunConfig['BC_NPOINTS'],  nsmooth=3)
    elif RunConfig['BC_ALGO'] != "None":
        raise Exception("Wrong BC_ALGO value, use either 'None', 'Coord', 'Spline', or 'Iterative'") 
    return d

def autozero(d, z1=(0.1,-0.1), z2=(0.1,-0.1),):
    """
    This function search for a peak around 0ppm, assumed to be the reference compound (TMS)
    and assign it to exactly 0
    z1 and z2 (z2 not used in 1D) are the zoom window in which the peak is searched
    """
    # peak pick TMS
    sc = 25                     # scaling for pp threshold
    try:
        d.absmax = np.nanmax( np.abs(d.buffer) )
    except AttributeError:
        d._absmax = np.nanmax( np.abs(d.buffer) )   # newest version of Spike
    d.peaks=[]          # initialize the loop
    while len(d.peaks)==0 and sc<400:
        sc *= 2.0
        if d.dim == 1:
            d.pp(zoom=z1, threshold=d.absmax/sc)   # peak-pick around zero
        elif d.dim == 2:
            d.pp(zoom=(z1,z2),  threshold=d.absmax/sc)   # peak-pick around zero
    # exit if nothing at max sc
    if len(d.peaks) == 0:
        print("**** autozero does not find the TMS peak ****")
        return d
    # then set to 0
    d.peaks.largest()       # sort largest first
    d.centroid()            # optimize the peak
    if d.dim == 1:
        d.axis1.offset -= d.axis1.itoh(d.peaks[0].pos)  # and do the correction
    elif d.dim == 2:
        d.axis2.offset -= d.axis2.itoh(d.peaks[0].posF2)
        d.axis1.offset -= d.axis1.itoh(d.peaks[0].posF1)
    del(d.peaks)
    return d
    
def get_localparameters(expname):
    """
    ( used to be get_config() ... and .cfg files)
    reads the parameters.json file that is located in the root of the processing
    it contains additional parameters applied only for the processing of this experiment
    [manip/#expno]
    ppm_offset = 1.2
    ph0 = 30
    ph1 = -60
    return a tuples with the values

    missing values are set to zero
    no effect if the file is absent
    """
    global RunConfig
    from pprint import pprint
    from collections import defaultdict
    # expname : Base/Manipe/Expno/fid
    fiddir =  op.dirname(expname)            # Base/Manipe/Expno
    basedir, fidname = op.split(fiddir)  # Base/Manipe Expno
    base, manip =  op.split(basedir)
    # print("base, manip, fidname, fiddir, expname")
    # print(base, manip, fidname, fiddir, expname)
    try:
        with open(op.join(base,"parameters.json"),"r") as f:
            try:
                localjson = json.load(f)
            except:
                raise Exception('Error in reading Configuration file %s'%(op.join(base,"parameters.json"),))
    except IOError:
        localjson = {} # no parameters in this file
        print(f'####### in {base}/parameters.json - {manip}/{fidname}  No param')
    try:
        res = localjson[ f"{manip}/{fidname}" ]
    except KeyError:
        res = {}      #
    # RunConfig = defaultdict(float) | Config | res         # update internal RunConfig
    RunConfig = defaultdict(float)
    RunConfig.update(Config)
    RunConfig.update(res)
    pprint(res)
    return res

#---------------------------------------------------------------------------
#4. Main code
def process_1D(xarg):
    "Performs all processing of exp, and produces the spectrum (with and without peaks) and the list files"
    exp, resdir = xarg
    # exp : Base/Manipe/Expno/fid
    fiddir =  op.dirname(exp)            # Base/Manipe/Expno
    basedir, fidname = op.split(fiddir)  # Base/Manipe Expno
    base, manip =  op.split(basedir)

    print (f"=================================================\n{manip}/{fidname} 1D\n")
    LocParam = get_localparameters(exp)

    d = FT1D(exp)
    if RunConfig['TMS']:
        d = autozero(d)
    d.save(op.join( fiddir,"processed.gs1") )
    
    analyze_1D(d, name=op.join(resdir, '1D', fidname), pplevel=50)
    return d

def analyze_1D(d, name, pplevel=50):
    "Computes peak and bucket lists and exports them as CSV files"
    noise = findnoiselevel( d.get_buffer() )
    d.pp(pplevel*noise)
    d.centroid()            # optimize the peaks
    
    pkout = open( name+'_peaklist.csv' , 'w') 
    d.peaks.report(f=d.axis1.itop, file=pkout)
    pkout.close()

    bkout = open( name+'_bucketlist.csv' , 'w')

    if (findNuc(d) == '19F'):
        d.bucket1d(file=bkout, zoom=RunConfig['BCK_19F_LIMITS'], bsize=RunConfig['BCK_19F_1D'], pp=RunConfig['BCK_PP'], sk=RunConfig['BCK_SK'])
    else:
        d.bucket1d(file=bkout, zoom=RunConfig['BCK_1H_LIMITS'], bsize=RunConfig['BCK_1H_1D'], pp=RunConfig['BCK_PP'], sk=RunConfig['BCK_SK'])
    bkout.close()
    return d

def plot_1D(d, exp, resdir):
    fiddir =  op.dirname(exp)
    basedir, fidname = op.split(fiddir)
    base, manip =  op.split(basedir)

    d.unit = 'ppm'
    try:
        d.display(label=f"{manip}/{fidname} {d.comment}")
    except AttributeError:
        d.display(label=f"{manip}/{fidname}")
    if RunConfig['PDF']:
	    plt.savefig( op.join(resdir, '1D', fidname+'.pdf') ) # Creates a PDF of the 1D spectrum without peaks
    if RunConfig['PNG']:
	    plt.savefig( op.join(resdir, '1D', fidname+'.png'), dpi=300 ) # and a PNG 
    d.display_peaks() # peaks.display(f=d.axis1.itop)
    if RunConfig['PDF']:
	    plt.savefig( op.join(resdir, '1D', fidname+'_pp.pdf') ) # Creates a PDF of the 1D spectrum with peaks
    if RunConfig['PNG']:
	    plt.savefig( op.join(resdir, '1D', fidname+'_pp.png'), dpi=300 ) # and a PNG
    plt.close()

def process_2D(xarg):
    "Performs all processing of experiment 'numb2' and produces the spectrum with and without peaks"
    numb2, resdir = xarg
    fiddir =  op.dirname(numb2)
    basedir, fidname = op.split(fiddir)
    base, manip =  op.split(basedir)
    pulprog = bk.read_param(bk.find_acqu( fiddir ) ) ['$PULPROG']
    exptype = pulprog[1:-1]  # removes the <...>
    if 'cosy' in exptype:
        exptype = 'COSY'
    elif 'dipsi' in exptype or 'mlev' in exptype or 'tocsy' in exptype:
        exptype = 'TOCSY'
    elif 'hsqc' in exptype:
        exptype = 'HSQC'
    elif 'hmbc' in exptype:
        exptype = 'HMBC'
    elif 'ste' in exptype or 'led' in exptype:
        exptype = "DOSY"
    else:
        exptype = 'UNKNOWN'
    print (f"=================================================\n{manip}/{fidname}\nExperiment detected as ", exptype)
    LocParam = get_localparameters(numb2)

    d = bk.Import_2D(numb2)
    NUS = d.params['acqu']['$FnTYPE']
    if NUS != "0":
        print("It seems this experiment is in NUS mode - NUS processing not implemented yet")
        return None


    d.unit = 'ppm'
    scale = 10.0 
    sanerank = RunConfig['SANERANK']

    #1. If TOCSY  
    if exptype == "TOCSY":
        if 'etgp' in pulprog :
            d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim()
            if sanerank != 0:
                d.sane(rank=sanerank, axis=1)
            d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).bk_ftF1().modulus().rem_ridge()
        else:
            d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim()
            if sanerank != 0:
                d.sane(rank=sanerank, axis=1)
            d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).bk_ftF1().modulus().rem_ridge()
        scale = 50.0
        d.axis2.offset += RunConfig['ppm_offset']*d.axis2.frequency
        if RunConfig['TMS']:
            d = autozero(d)

    #2. If COSY DQF
    elif exptype == "COSY":
        d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim()
        if sanerank != 0:
            d.sane(rank=sanerank, axis=1)
        d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).bk_ftF1().modulus().rem_ridge()
        scale = 20.0
        d.axis2.offset += RunConfig['ppm_offset']*d.axis2.frequency
        if RunConfig['TMS']:
            d = autozero(d)
    #3. If HSQC
    elif exptype == "HSQC":
        if 'ml' in pulprog:
            print ("TOCSY-HSQC")
        d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim()
        if sanerank != 0:
            if d.size1 > 200:   # some HSQC are very short!
                d.sane(rank=sanerank, axis=1)
            else:
                print('size too small for sane')
        d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).bk_ftF1().modulus().rem_ridge()  # ft_sh()
        scale = 10.0
        d.axis2.offset += RunConfig['ppm_offset']*d.axis2.frequency
        if RunConfig['TMS']:
            d = autozero(d, z1=(5,-5))

    #4. If HMBC
    elif exptype == "HMBC":
        d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim()
        if 'et' in pulprog:
            d.conv_n_p()
        if sanerank != 0:
            d.sane(rank=sanerank, axis=1)
        d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).bk_ftF1().modulus().rem_ridge() # For Pharma MB1-X-X series
        scale = 10.0
        d.axis2.offset += RunConfig['ppm_offset']*d.axis2.frequency
        if RunConfig['TMS']:
            d = autozero(d, z1=(5,-5))

    #5. If DOSY - Processed in process_DOSY
    elif exptype == "DOSY":   # Should not happen, as DOSY are processed independtly
        d = process_DOSY(numb2)
        scale = 50.0
    # else die
    else:
        raise ValueError("Unknown PULPROG in acqus")

    analyze_2D( d, name=op.join(resdir, '2D', exptype+'_'+fidname) )
    d.save(op.join(fiddir,"processed.gs2"))
    return d, scale

def Dprocess_2D( numb2, resdir ):
    "Performs DOSY processing of experiment 'numb2' and produces the spectrum with and without peaks"
    fiddir =  op.dirname(numb2)
    basedir, fidname = op.split(fiddir)
    base, manip =  op.split(basedir)
    exptype =  bk.read_param(bk.find_acqu( fiddir ) ) ['$PULPROG']
    exptype =  exptype[1:-1]  # removes the <...>

    LocParam = get_localparameters(numb2)

    d = bk.Import_2D(numb2)
    d.unit = 'ppm'
    if 'ste' in exptype or 'led' in exptype:
        print ("DOSY")
        d = process_DOSY(numb2)
        scale = 50.0
    else:
        raise Exception("This is not a DOSY: " + numb2)

    dd = analyze_2D( d, name=op.join(resdir, '2D', 'DOSY_'+fidname) )
    d.save(op.join(fiddir,"processed.gs2"))
    return dd, scale


def plot_2D(d, scale, numb2, resdir ):
    fiddir =  op.dirname(numb2)
    basedir, fidname = op.split(fiddir)
    base, manip =  op.split(basedir)
    exptype =  bk.read_param(bk.find_acqu( fiddir ) ) ['$PULPROG']
    exptype =  exptype[1:-1]  # removes the <...>

    d.display(scale="auto", autoscalethresh=10) #scale)
    if RunConfig['PDF']:
	    plt.savefig( op.join(resdir, '2D', exptype+'_'+fidname+'.pdf') ) # Creates a PDF of the 2D spectrum without peaks
    if RunConfig['PNG']:
	    plt.savefig( op.join(resdir, '2D', exptype+'_'+fidname+'.png') ) # Creates a png of the 2D spectrum without peaks
    d.display_peaks(color="g")
    if RunConfig['PDF']:
	    plt.savefig( op.join(resdir, '2D', exptype+'_'+fidname+'_pp.pdf') ) # Creates a PDF of the 2D spectrum with peaks
    if RunConfig['PNG']:
	    plt.savefig( op.join(resdir, '2D', exptype+'_'+fidname+'_pp.png') ) # Creates a png of the 2D spectrum with peaks
    plt.close()
    return d

def process_DOSY(fid):
    "Performs all processing of DOSY "
    import spike.plugins.NMR.PALMA as PALMA
    global POOL
    lazy=RunConfig['DOSY_LAZY']
    d = PALMA.Import_DOSY(fid)
    print('PULPROG', d.params['acqu']['$PULPROG'],'   dfactor', d.axis1.dfactor)
    # process in F2
    processed = op.join( op.dirname(fid),'processed.gs2' )
    if op.exists( processed ) and lazy:
        dd = npkd.NMRData(name=processed)
        npkd.copyaxes(d, dd)
        dd.axis1.itype = 0
        dd.axis2.itype = 0
        dd.adapt_size()
    else:
        d.chsize(sz2=min(16*1024,d.axis2.size))
        d.apod_em(RunConfig['LB_1H'],axis=2).ft_sim().bruker_corr()
        # automatic phase correction
        r = d.row(2)
        r.apmin()
        d.phase(r.axis1.P0, r.axis1.P1, axis=2).real()
        # correct
        d.axis2.offset += RunConfig['ppm_offset']*d.axis2.frequency
        # save
        fiddir =  op.dirname(fid)
        d.save(op.join(fiddir,"preprocessed.gs2"))
        # ILT
        NN = 256
        d.prepare_palma(NN, 10.0, 10000.0)
        mppool = POOL
        dd = d.do_palma(miniSNR=20, nbiter=RunConfig['PALMA_ITER'], lamda=0.05, mppool=mppool )
        if RunConfig['TMS']:
            r = autozero(r)  # calibrate only F2 axis !
            dd.axis2.offset = r.axis1.offset
    dd.axis2.currentunit = 'ppm'
    return dd


def analyze_2D(d, name, pplevel=10):
    "Computes peak and bucket lists and exports them as CSV files"
    from spike.NMR import NMRAxis
    dd = d.copy() # Removed because of error with 'sane' algorithm

    dd.sg2D(window_size=7, order=2) # small smoothing
    noise = findnoiselevel( dd.get_buffer().ravel() )
    threshold = pplevel*noise
    if noise == 0:          # this might happen on DOSY because of 0 values in empty columns
        rr = dd.get_buffer().ravel()
        threshold = pplevel*findnoiselevel( rr[rr>0] )
    dd.pp(threshold)
    try:
        dd.centroid()            # optimize the peaks
    except AttributeError:
        pass
#    dd.display_peaks(color="g")
    
    pkout = open( name+'_peaklist.csv'  , 'w')
    dd.report_peaks(file=pkout)
    pkout.close() 
    bkout = open( name+'_bucketlist.csv'  , 'w')
    BCK_1H_2D = RunConfig['BCK_1H_2D']
    BCK_13C_2D = RunConfig['BCK_13C_2D']
    BCK_1H_LIMITS = RunConfig['BCK_1H_LIMITS']
    BCK_13C_LIMITS = RunConfig['BCK_13C_LIMITS']
    BCK_DOSY =  RunConfig['BCK_DOSY']
    BCK_PP = RunConfig['BCK_PP']
    if name.find('COSY') != -1 or name.find('TOCSY') != -1:
        dd.bucket2d(file=bkout, zoom=(BCK_1H_LIMITS, BCK_1H_LIMITS), bsize=(BCK_1H_2D, BCK_1H_2D), pp=BCK_PP, sk=RunConfig['BCK_SK'] )
    elif name.find('HSQC') != -1 or name.find('HMBC') != -1:
        dd.bucket2d(file=bkout, zoom=( BCK_13C_LIMITS, BCK_1H_LIMITS), bsize=(BCK_13C_2D, BCK_1H_2D), pp=BCK_PP, sk=RunConfig['BCK_SK'] )
    elif name.find('DOSY') != -1 :
        ldmin = np.log10(d.axis1.dmin)
        ldmax = np.log10(d.axis1.dmax)
        sw = ldmax-ldmin
        dd.buffer[:,:] = dd.buffer[::-1,:]  # return axis1 
        dd.axis1 = NMRAxis(size=dd.size1, specwidth=100*sw, offset=100*ldmin, frequency = 100.0, itype = 0)     # faking a 100MHz where ppm == log(D)
        dd.bucket2d(file=bkout, zoom=( (ldmin, ldmax) , BCK_1H_LIMITS), bsize=(BCK_DOSY, BCK_1H_2D), pp=BCK_PP, sk=RunConfig['BCK_SK'] ) #original parameters
    else:
        print ("*** Name not found!")
    bkout.close()
    d.peaks = dd.peaks
    return d

def process_sample(sample, resdir):
    "Redistributes NMR experiment to corresponding processing"
    global POOL
    print("%%%%%%%%%%%%%%%%", sample, resdir)
    sample_name = op.basename(sample)
    print (sample_name)
# First 1D
#    for exp in glob( op.join(sample, "*/fid") ): # For 1D processing
#        print (exp)
#		 process_1D(exp, resdir)
    lfid = glob( op.join(sample, "*", "fid") )
    lgf1 = glob( op.join(sample, "*", "*.gf1") )
    l1D = lfid + lgf1
    if l1D != []:
        xarg = list( zip_longest(l1D, [resdir], fillvalue=resdir) )
        # print (xarg)
        if POOL is None:
            result = imap(process_1D, xarg)
        else:
            result = POOL.imap(process_1D, xarg)
        for i,d in enumerate(result):
            print(d)
            plot_1D(d, l1D[i], resdir )
# then 2D
#    for exp in glob( op.join(sample, "*/ser") ): # For 2D processing
#        print (exp)
#        process_2D(exp, resdir)
    l2D = []
    lDOSY = []
    for f in glob( op.join(sample, "*", "ser") ):
        fiddir =  op.dirname(f)
        if op.exists( op.join(fiddir,'difflist') ):  # DOSY should have their difflist
            lDOSY.append(f)
        else:
            l2D.append(f)
    if l2D != []:
        xarg = list( zip_longest(l2D, [resdir], fillvalue=resdir) )
        # print (xarg)
        if POOL is None:
            result2 = imap(process_2D, xarg)
        else:
            result2 = POOL.imap(process_2D, xarg)
        for i, r in enumerate(result2):
            if r is not None:
                d, scale = r
                print(d)
                if d.dim == 2:
                    plot_2D(d, scale, l2D[i], resdir )
                elif d.dim == 1:
                    plot_1D(d, l2D[i], resdir )

# finally DOSYs internally //ized
    for dosy in lDOSY:
        d, scale = Dprocess_2D( dosy, resdir )
        print(d)
        print(len(d.peaks), 'Peaks')
        plot_2D(d, scale, dosy, resdir )
    

def analysis_report(resdir, fname):
    """
    Generate a csv report for all bucket lists and peak lists found during processing
    """
    with open(fname,'w') as F:
        print("# report from", resdir, file=F)                                 # csv comment
        print("manip, expno, type, file, content", file=F )
        for exp in glob(op.join(resdir,'*')):
            for f1d in glob(op.join(exp, '1D', '*.csv')):   # all 1D
                csvname = (op.basename(f1d))
                csvsplit = csvname.split('_')
                firstl = open(f1d,'r').readline()
                print (op.basename(exp), csvsplit[0], '1D', csvname, firstl[1:], sep=',', file=F)
            for f2d in glob(op.join(exp, '2D', '*.csv')):   # all 1D
                csvname = (op.basename(f2d))
                csvsplit = csvname.split('_')
                firstl = open(f2d,'r').readline()
                print (op.basename(exp), csvsplit[1], csvsplit[0], csvname, firstl[1:], sep=',', file=F)

#---------------------------------------------------------------------------
def main(args):
    "Creates a new directory for every sample along with subdirectories for the 1D and 2D data"
    import traceback
    global POOL
    Nproc = args.Nproc
    DIREC = args.DIREC
    if not op.isdir(DIREC):
        raise Exception("\n\nDirectory %s is non-valid"%DIREC)
    if len( glob( op.join(DIREC, '*') ) )==0:
        print( "WARNING\n\nDirectory %s is empty"%DIREC)

    Bruker_Report.generate_report( DIREC, op.join(DIREC, 'report.csv'), \
            do_title=RunConfig['TITLE'], addpar=RunConfig['addpar'], add2Dpar=RunConfig['add2Dpar'], addDOSYpar=RunConfig['addDOSYpar'] )

    if args.template:
        # first RunConfig
        with open(op.join(DIREC,'RunConfig_templ.json'), 'w') as F:
            json.dump(Config, F, indent=4)
        # then parameters
        dic = {}      # build dic for file tree
        for sp in sorted(glob( op.join(DIREC, '*') )): 
            if not op.isdir(sp):
                continue
            sample = op.basename(sp)
            print("#############", sample)
            explist = []
            for exp in sorted(glob( op.join(sp, "*", "fid")) ):
                expno =  op.basename(op.dirname(exp))
                dic[f"{sample}/{expno}"] = {"remark":"1D experiment"}
            for exp in sorted(glob( op.join(sp, "*", "*.gf1")) ):
                expno =  op.basename(op.dirname(exp))
                dic[f"{sample}/{expno}"] = {"remark":"1D experiment"}
            for exp in glob( op.join(sp, "*", "ser") ):
                expno =  op.basename(op.dirname(exp))
                dic[f"{sample}/{expno}"] = {"remark":"2D experiment"}
        with open(op.join(DIREC,'parameters_templ.json'), 'w') as F:
            json.dump(dic, F, indent=4)
        args.dry = True

    # test left overs
    if op.exists(op.join(DIREC, 'Results')):       # leftovers...
        print("""
Results from a previous run are present, STOPPING NOW...
delete or move to a safe place the folder "Results" located in %s"""%(DIREC))
        return

    with open(op.join(DIREC,'Config.dump'), 'w') as F:
        json.dump(Config, F, indent=4)

    if args.dry:
        return

    if Nproc > 1:
        print('Processing on %d processors'%Nproc)
        mkl_set_num_threads(1)
        copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
        POOL = mp.Pool(Nproc)
    else:
        POOL = None

    for sp in glob( op.join(DIREC, '*') ): 
        # validity of sp
        if not op.isdir(sp):
            # print("alien files")
            continue
        if  op.basename(sp)  == '__pycache__':  # python internal
            continue
        # ok, go on
        resdir = op.join( DIREC, 'Results', op.basename(sp) )
        mkdir(resdir)
        for folder in ['1D', '2D']:
            mkdir( op.join(resdir, folder) )
        try:
            process_sample(sp, resdir)
        except IOError:
            print("**** ERROR with file {}\n---- not processed\n".format(sp))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
    analysis_report(op.join( DIREC, 'Results'), op.join( DIREC,'analysis.csv'))

if __name__ == "__main__":

    print ("Processing ...")

    # import cProfile
    # cProfile.run( 'main(args)', 'restats')
    main(args)


