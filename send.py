#!/usr/bin/python
import random
import os
import sys
import socket

##### Modify parameters here  ###############
hostname = socket.gethostname()
servers = ['einstein','server','R630']
if (hostname in servers):
    Cluster = "PBS"
else:
    Cluster = "local"
# Cluster="condor"

############################################
print("Cluster: {0}".format(Cluster))

if len(sys.argv) == 1:
    folderPre = ""
    execute = "feyncalc.exe"
elif len(sys.argv) >= 2:
    folderPre = "_".join(sys.argv[1:])+"_"
    execute = "feyncalc_"+folderPre[:-1]+".exe"


rootdir = os.getcwd()
inlist = open(rootdir+"/inlist", "r")
merge = "merge.py"
infile = "inlist"

lines = inlist.readlines()
inlist.close()
for eachline in lines:
    os.chdir(rootdir)
    para = eachline.split()

    if len(para) == 0:
        print("All submitted!")
        break

    # if int(para[-2])==0:
    #     title="freq"
    # elif int(para[-2])==1:
    #     title="eqTime"
    # else:
    #     print "Not yet implemented!"
    #     break

    homedir = os.getcwd() + \
        "/"+folderPre+"Order{0}_Beta{1}_lambda{2}".format(para[0],para[1],para[3])
    if os.path.exists(homedir):
        os.system("rm -fr "+homedir)
    os.system("mkdir "+homedir)

    os.system("cp -r groups "+homedir)
    os.system("cp {0} {1}".format(execute, homedir))
    os.system("cp {0} {1}".format(merge, homedir))
#    os.system("cp {0} {1}".format(infile, homedir))
    inf = open(homedir+"/"+infile, "w")
    inf.write(eachline+"\n"+lines[-1])
    inf.close()

    infilepath = homedir+"/infile"
    if (os.path.exists(infilepath) != True):
        os.system("mkdir "+infilepath)
    outfilepath = homedir+"/outfile"
    if(os.path.exists(outfilepath) != True):
        os.system("mkdir "+outfilepath)
    jobfilepath = homedir+"/jobfile"
    if(os.path.exists(jobfilepath) != True):
        os.system("mkdir "+jobfilepath)
    weightpath = homedir+"/weight"
    if(os.path.exists(weightpath) != True):
        os.system("mkdir "+weightpath)

    for pid in range(int(para[-1])):

        ########## Generate input files ################
        infile_pid = "_in"+str(pid)
        f = open(infilepath+"/"+infile_pid, "w")
        item = para[0:-1]
        item.append(str(-int(random.random()*1000000)))
        item.append(str(pid))
        stri = " ".join(item)
        f.write(stri)
        f.close()

        ### terminal output goes here #############
        outfile = "_out"+str(pid)
        ### job file to submit to cluster  ########
        jobfile = "_job"+str(pid)+".sh"

        if Cluster == "local":
            os.chdir(homedir)
            os.system("./"+execute+" < "+infilepath+"/" +
                      infile_pid+" > "+outfilepath+"/"+outfile+" &")

        elif Cluster == "condor":
            with open(jobfilepath+"/"+jobfile, "w") as fjob:
                fjob.write("executable = {0}\n".format(execute))
                fjob.write("input ={0}/{1}\n".format(infilepath, infile_pid))
                fjob.write("output ={0}/{1}\n".format(outfilepath, outfile))
                fjob.write("initialdir ={0}\n".format(homedir))
                fjob.write("queue")

            os.chdir(homedir)
            os.system("condor_submit {0}/{1}".format(jobfilepath, jobfile))
            os.system("rm "+jobfilepath + "/"+jobfile)
        elif Cluster == "PBS":
            with open(jobfilepath+"/"+jobfile, "w") as fjob:
                fjob.write("#!/bin/sh\n"+"#PBS -N "+jobfile+"\n")
                fjob.write("#PBS -o "+homedir+"/Output\n")
                fjob.write("#PBS -e "+homedir+"/Error\n")
                fjob.write("#PBS -l walltime=2000:00:00\n")
                fjob.write("echo $PBS_JOBID >>"+homedir+"/id_job.log\n")
                fjob.write("cd "+homedir+"\n")
                fjob.write("./"+execute+" < "+infilepath+"/" +
                           infile_pid+" > "+outfilepath+"/"+outfile)

            os.chdir(homedir)
            os.system("qsub "+jobfilepath + "/"+jobfile)
            os.system("rm "+jobfilepath + "/"+jobfile)
        else:
            print("I don't know what is {0}".format(Cluster))
            break

    os.chdir(homedir)
    if "bare" not in folderPre.lower():
        os.system("./" + merge + " > weight.log &")

print("Jobs manage daemon is ended")
sys.exit(0)
