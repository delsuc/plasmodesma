#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
This program takes a directory and build a report of all NMR experiments inside.

Usage 

>Bruker_Report directory

which creates a report.csv file

Details of what is in report is defined internaly by the paramtoprint list.

written by M-A Delsuc, first version on march 2016
some parts from spike ( github.com/spike-project/spike )
use it freely, licence is CC-BY 4.0 (use it freely, mention its origin)

"""
from __future__ import print_function
# make it python2 / python 3 compatile
import os
import os.path as op
import sys
import re
import datetime

################################################################

# list of param to print - you may modify !
paramtoprint = ['PULPROG', 'SFO1', 'NS', 'TE', 'TD', 'RG', 'SW', 'O1','D1','P1']
param2Dtoprint = ['SFO1', 'TD','SW', 'O1', 'D9', 'FnMODE']
paramDOSYtoprint = ['D20','P30']

# for the program to recognize arrays, the arrayname should be given below ( | separated, no space)
arraynames = 'D|P|PL|PCPD|SP'

# name of the report file
Reportfile = 'report.csv'

################################################################
def read_param(filename="acqus"):
    """ 
    load a Bruker acqu or proc file as a dictionnary
    
    arrayed values are stored in python array
    
    comments (lines starting with $$) are stored in the special entrey [comments]
    
    M-A Delsuc jan 2006
    oct 2006 : added support for array
    """
    debug = 0
    with open(filename) as fin:
        # read file
        dico = {}
        dico['comments']=""
        f=fin.read()
        fin.close()
        ls= f.split("\n")

    #    for v in ls:
        while ls:
            v=ls.pop(0)
            v = v.strip()
            if debug: print("-",v,"-")
            if (re.search(r"^\$\$",v)):  # print comments
                dico['comments']=dico['comments']+"\n"+v
            else:
                m=re.match(r"##(.*)= *\(0\.\.([0-9]*)\)(.*)$",v )   # match arrays
                if (m is not None):
                    if debug: print("ARRAY",v,m.group(1,2,3))
                    (key,numb,line)=m.group(1,2,3)
                    v=ls.pop(0)
                    v = v.lstrip()
                    while (not re.match(r"##",v)):    # concatenate all array lines
                        line = line+" "+v
                        v=ls.pop(0)
                        if debug: v = v.lstrip()
                    ls.insert(0,v)
                    array=line.split()
                    if debug: print(key,numb,len(array),array)
                    if ((int(numb)+1) != len(array)):   # (0..9) is 10 entries !
                        raise "size mismatch in array"
                    dico[key] = array
                    continue
                m=re.match(r"##(.*)= *<(.*)>",v )   #match string
                if (m is not None): 
                    if debug: print("STRING",v)
                    (key,val) = m.group(1,2)
                    dico[key] = val
                    continue
                m=re.match(r"##(.*)= *(.*)$",v )   #match value
                if (m is not None):
                    if debug: print("VAL",v)
                    (key,val) = m.group(1,2)
                    dico[key] = val
                    continue
# debug code
    if debug:
        for i in dico.keys():
            print(i+" = "+str(dico[i]))
    return dico

def title_parser(textfile):
    """
    given a title file, parses the content for standardized information:
    eg:
        [PFDA] = 10 mM, ds DMSO 5mm TE 298K
        EMGE_016
    parsed as:
        [~name_of_product~] = ~concentration~ mM, ds ~Solvent_name~  TE ~Temp_value~ 
        ~product_reference~
    Arthur Steur - july 2019
    """
    dico = {} # create dictionary 'dico' to store our title elements
    
    # we make a re to find various parameters, following a rather strict syntax:

    # [ produc name ] = val, 
    # produit_and_conc = re.compile(r'\[([\w\s]+)\]\s*=\s*([0-9\.]+)\s*(\w+)M,?')
    produit_and_conc = re.compile(r'\[([\w\s]+)\]\s*=\s*([0-9\.]*\s\w+),?')

    matches1 = produit_and_conc.search(textfile)
    
    # ds Solvent  
    solvent = re.compile(r'ds +([0-9a-zA-Z/-]+)') # matches the solvent
    matches2 = solvent.search(textfile)
    
    temperature = re.compile(r' *([0-9]+) *K',) 
    # matches the temperature with the unit magnitude
    matches3 = temperature.search(textfile)
    
    internal_reference = re.compile(r'^([0-9a-zA-Z]+[_-][0-9a-zA-Z_-]+)',re.M)
    # matches the internal reference 
    matches4 = internal_reference.search(textfile)
    
    # list making process begins.
    try:
        dico['product'] = matches1.group(1) 
    except AttributeError:
        dico['product'] = "-"
        dico['concentration'] = "NaN"
    else:
        dico['product'] = matches1.group(1)
        dico['concentration'] = matches1.group(2)
    
    try:
        dico['solvent'] = matches2.group(1)
    except AttributeError:
        dico['solvent'] = '-'
    
    try:
        dico['temperature'] = matches3.group(1)
    except AttributeError:
        dico['temperature'] = '-'

    try:
        dico['product_reference'] = matches4.group()
    except AttributeError:
        dico['product_reference'] = '-'
    
    for i in range(1,len(dico)): # loops i for each element in the dictionary
        try:
            textfile = textfile.replace(eval('matches'+ str(i)).group(),"")
        except AttributeError:
            continue
        # replace the match(i) with "" which is nothing. 
        
    # everything in textfile that was already put in the dictionary is now deleted from
    # the previous loop above. now we take that last bit of string and put it back in 'dico'.
    dico['comment'] = textfile.strip().replace(',',' ').replace('\n',' ') 
#    print(dico)
    return dico

def readplist(paramtoadd, paramdict):
    "parse lists from acqus files - only that ones defined in arraynames"
    m = (re.match('(%s)([0-9]+)'%(arraynames,),paramtoadd))
    if m :    # arrays are special !
        i = int(m.group(2))
        pp = paramdict['$'+m.group(1)]
        val = pp[i]
    else:
        val = paramdict['$%s'%paramtoadd]
    return val
    
def generate_report(direc, reportfile, do_title=True, addpar=[], add2Dpar=[], addDOSYpar=[] ):
    """
    create a file 'reportfile' with parameters of all experiments found in direc
    if do_title is true, the title file will be parsed for standard values (see documentation)
    addpar, add2Dpar addDOSYpar are added to the parameter lists
    """
    for p in addpar:
        paramtoprint.append(p)
    for p in add2Dpar:
        param2Dtoprint.append(p)
    for p in addDOSYpar:
        paramDOSYtoprint.append(p)

    mcurr = '--'
    count = 0
    with open(reportfile,'w') as F:
        print("# report from", direc, file=F)                                 # csv comment
        second_head = "# , , , parameters, " + " ,"*(len(paramtoprint)-1)
        second_head += "2D, " + " ,"*(len(param2Dtoprint)-1)
        second_head += "DOSY"
        print(second_head, file=F)                                 # csv comment
        parm_header = ["manip", "expno", "date"] + paramtoprint + param2Dtoprint + paramDOSYtoprint
        if do_title:
            title_keys = ['product', 'concentration', 'solvent', 'temperature', 'product_reference', 'comment']
            parm_header = parm_header + title_keys
        print ( *parm_header, sep=',', file=F)  # csv header
    #    for f in glob.glob('/DATA/DOSY_Sumofusion/*/*/acqus'):
        for root, dirs, files in os.walk(direc):
            if 'acqus' in files:
                p = read_param( op.join(root,'acqus') )
                if 'ser' in files:
                    is_2D = True
                    p2 = read_param( op.join(root,'acqu2s') )
                else:
                    is_2D = False                
                is_DOSY = 'difflist' in files
                ff =  root.split(op.sep)   #[... 'manip', 'expno']
                expno = ff[-1]
                manip = ff[-2]
                if manip != mcurr:
                    mcurr = manip
                    print(file=F)   # skip a line on new manip
                date = datetime.date.fromtimestamp(float(p['$DATE'])) #.strftime("%d %b %Y %H:%M:%S")
                plist = [manip, expno, date]
                for param in paramtoprint: # first regular params 
                    plist.append( readplist(param, p) )
                if is_2D:
                    for param in param2Dtoprint: # then 2D
                        try:
                            plist.append( readplist(param, p2) )
                        except KeyError:
                            plist.append( readplist(param, p) )
                else:
                    plist = plist + ['-']*len(param2Dtoprint)
                if is_DOSY:
                    for param in paramDOSYtoprint: # then DOSY
                        plist.append( readplist(param, p) )
                else:
                    plist = plist + ['-']*len(paramDOSYtoprint)

                # then title
                if do_title:
                    titlefound = False
                    for titre in ('title', 'TITLE', 'title.txt', 'TITLE.TXT'):   # search in possible names
                        title = op.join(root,'pdata','1', titre)
                        titlefound = op.exists(title)
                        if titlefound:
                            break
                    if titlefound:
                        pt = title_parser(open(title).read())
                        for k in title_keys:
                            plist.append(pt[k])
                    else:
                        for k in title_keys:
                            plist.append('...')
                print (*plist, sep=',',file=F)
                count += 1
    if mcurr == '--':
        print ('NO DATA')
    else:
        print('found %d entries - results are in %s'%(count, reportfile))

def main():
    try:
        direc = sys.argv[1]
    except:
        print(__doc__)
        sys.exit(0)
    if not op.exists(direc) or not op.isdir(direc):
        print ('\nWARNING - %s is not a valid directory\n'%direc)
        print(__doc__)
        sys.exit(1)
    generate_report(direc, Reportfile)

if __name__ == "__main__":
    main()

