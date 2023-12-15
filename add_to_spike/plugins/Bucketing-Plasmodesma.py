#!/usr/bin/env python
# encoding: utf-8
"""
A set of tools for computing bucketing for 1D and 2D NMR spectra


Created by DELSUC Marc-André on 2015-09-06.
"""

from __future__ import print_function
import numpy as np
import unittest

from spike import NPKError
from spike.NPKData import NPKData_plugin, NPKData

#---------------------------------------------------------------------------
def bucket1d(data, zoom=(0.5, 9.5), bsize=0.04, file=None):
    """
 This tool permits to realize a bucket integration from the current 1D data-set.
 You will have to determine  (all spectral values are in ppm)
   - zoom (low,high),  : the starting and ending ppm of the integration zone in the spectrum
   - bsize: the size of the bucket
   - file: the filename to which the result is written


 For a better bucket integration, you should be careful that :
   - the bucket size is not too small, size is better than number !
   - the baseline correction has been carefully done
   - the spectral window is correctly determined to encompass the meaningfull spectral zone.

    """
    data.check1D()
    start, end = zoom
    if (bsize <= 0):        NPKError( "Negative bucket size not allowed")
    if (start-bsize/2 < data.axis1.itop(data.size1)):        NPKError( "Starting point outside spectrum")
    if (end+bsize/2 > data.axis1.itop(0)):        NPKError( "Ending point outside spectrum")
    if  ((end-start)/bsize < 10):        NPKError( "Integration zone too small or Bucket too large")
    ppm_per_point = (data.axis1.specwidth/data.axis1.frequency/data.size1)
    if (bsize < 2*ppm_per_point):        NPKError( "Bucket size smaller than digital resolution !")

    dcopy = data.copy()   # work now on a real version of the data
    dcopy.real(axis=1)

    s = "# %i buckets with a mean size of %.2f data points" % \
        ( round((end-start+bsize)/bsize), bsize/ppm_per_point)
    print(s, file=file)
    if file is not None:    # wants the prompt on the terminal
        print(s)
    print("center, bucket, max, min, std, bucket_size", file=file)
    there = max(start,end)   # end of the bucket region
    here = min(start,end)    # running center of the bucket - initialized to begining
    here2 = (here-bsize/2)   # running beginning of the bucket
    while (here2 < there):
        ih = round(dcopy.axis1.ptoi(here2))   # int of running beginning of the bucket
        next = (here2+bsize)                  # running en of bucket
        inext = (round(dcopy.axis1.ptoi(next))) # int of running en of bucket
        if ih<0 or inext<0:
            break
        integ = dcopy.buffer[inext:ih].sum()
        try:
            maxv = dcopy.buffer[inext:ih].max()
            minv = dcopy.buffer[inext:ih].min()
        except ValueError:
            maxv = np.NaN     # sum and std returns nan - max returns an error ???
            minv = np.NaN     # sum and std returns nan - min returns an error ???
        stdv = dcopy.buffer[inext:ih].std()
        print("%.3f, %.1f, %.1f, %.1f, %.1f, %d"%(here, integ/((ih-inext)*bsize), maxv, minv, stdv,  (ih-inext) ), file=file)
        here2 = next
        here = (here+bsize)
    return data
#---------------------------------------------------------------------------
def bucket2d(data, zoom=((0.5, 9.5),(0.5, 9.5)), bsize=(0.1, 0.1), file=None):
    """
 This tool permits to realize a bucket integration from the current 2D data-set.
 You will have to determine  (all spectral values are in ppm)
   - zoom (F1limits, F2limits),  : the starting and ending ppm of the integration zone in the spectrum
   - bsize (F1,F2): the sizes of the bucket
   - file: the filename to which the result is written


 For a better bucket integration, you should be careful that :
   - the bucket size is not too small, size is better than number !
   - the baseline correction has been carefully done
   - the spectral window is correctly determined to encompass the meaningfull spectral zone.

    """
    data.check2D()
    start1, end1 = zoom[0]
    start2, end2 = zoom[1]
    bsize1, bsize2 = bsize
    if (bsize1 <= 0 or bsize2<=0):        NPKError( "Negative bucket size not allowed")
    if (start1-bsize1/2 < data.axis1.itop(data.size1)):        NPKError( "Starting point outside spectrum")
    if (start2-bsize2/2 < data.axis2.itop(data.size2)):        NPKError( "Starting point outside spectrum")
    if (end1+bsize1/2 > data.axis1.itop(0)):        NPKError( "Ending point outside spectrum")
    if (end2+bsize2/2 > data.axis2.itop(0)):        NPKError( "Ending point outside spectrum")
    if  ((end1-start1)/bsize1 < 4):        NPKError( "Integration zone too small or Bucket too large")
    if  ((end2-start2)/bsize2 < 4):        NPKError( "Integration zone too small or Bucket too large")
    ppm_per_point1 = (data.axis1.specwidth/data.axis1.frequency/data.size1)
    ppm_per_point2 = (data.axis2.specwidth/data.axis2.frequency/data.size2)
    if (bsize1 < 2*ppm_per_point1):        NPKError( "Bucket size smaller than digital resolution !")
    if (bsize2 < 2*ppm_per_point2):        NPKError( "Bucket size smaller than digital resolution !")

    dcopy = data.copy()   # work now on a real version of the data
    dcopy.real(axis=2)
    dcopy.real(axis=1)

    s = "# %i rectangular buckets with a mean size of %.2f x %.2f data points" % \
        ( round((end1-start1+bsize1)/bsize1)*round((end2-start2+bsize2)/bsize2), \
        bsize1/ppm_per_point1, bsize2/ppm_per_point2)
    print(s, file=file)
    if file is not None:    # wants the prompt on the terminal
        print(s)
    print("centerF1, centerF2, bucket, max, min, std, bucket_size_F1, bucket_size_F2", file=file)
    here1 = min(start1, end1)
    here1_2 = (here1-bsize1/2)
    there1 = max(start1, end1)
#    F = open('toto.txt','w')
    while (here1_2 < there1):
        ih1 = int(round(dcopy.axis1.ptoi(here1_2)))
        next1 = (here1_2+bsize1)
        inext1 = int(round(dcopy.axis1.ptoi(next1)))
        if ih1<0 or inext1<0:
            break
        here2 = min(start2, end2)
        here2_2 = (here2-bsize2/2)
        there2 = max(start2, end2)
        while (here2_2 < there2):
            ih2 = int(round(dcopy.axis2.ptoi(here2_2)))
            next2 = (here2_2+bsize2)
            inext2 = int(round(dcopy.axis2.ptoi(next2)))
            if ih2<0 or inext2<0:
                break
            integ = dcopy.buffer[inext1:ih1, inext2:ih2].sum()
            area = ((ih1-inext1)*bsize1) * ((ih2-inext2)*bsize2)
            try:
                maxv = dcopy.buffer[inext1:ih1, inext2:ih2].max()
                minv = dcopy.buffer[inext1:ih1, inext2:ih2].max()
            except ValueError:
                maxv = np.NaN     # sum and std returns nan - max returns an error ???
                minv = np.NaN     # sum and std returns nan - max returns an error ???
            stdv = dcopy.buffer[inext1:ih1, inext2:ih2].std()
            print("%.3f, %.3f, %.1f, %.1f, %.1f, %.1f, %d, %d"%(here1, here2, integ/area, maxv, minv, stdv, (ih1-inext1), (ih2-inext2) ), file=file)
#            print(here1, here2, here1_2, here2_2, inext1, ih1, inext2, ih2, file=F)
            here2_2 = next2
            here2 = (here2+bsize2)
        here1_2 = next1
        here1 = (here1+bsize1)

    return data


class BucketingTests(unittest.TestCase):
    def setUp(self):
        self.verbose = 1    # verbose >0 switches on messages
    def announce(self):
        if self.verbose >0:
            print (self.shortDescription())
    def _test_log(self):
        """testing log"""
        import math
        self.announce()
        x = 0.0
        y = math.log(1.0)
        self.assertAlmostEqual(x, y )

NPKData_plugin("bucket1d", bucket1d)
NPKData_plugin("bucket2d", bucket2d)
