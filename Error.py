#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
import sys
import glob
import os
import re
import copy
#from scipy import integrate
mat.rcParams.update({'font.size': 16})
mat.rcParams["font.family"] = "Times New Roman"
size = 12

# XType = "Tau"
XType = "Mom"
# XType = "Angle"
l = 1
orderAccum = None

# 0: I, 1: T, 2: U, 3: S
# Channel = [0, 1, 2, 3]
Channel = [3]

ITUSPlot = False
SPlot = False
if len(Channel)==4:
    ITUSPlot = True
if len(Channel)==1:
    chan = Channel[0]
    if Channel[0]==3:
        SPlot = True

ChanName = {0: "I", 1: "T", 2: "U", 3: "S"}
# 0: total, 1: order 1, ...
# Order = [0, 1, 2, 3]


Order = None
kF = None
Lambda = None
AngleBin = None
ExtMomBin = None
AngleBinSize = None
ExtMomBinSize = None

steps = []
Gamma4q = []
Data = []



##############   2D    ##################################
###### Bare Green's function    #########################
# kF = np.sqrt(2.0)/rs  # 2D
# Bubble=0.11635  #2D, Beta=0.5, rs=1
# Bubble = 0.15916/2  # 2D, Beta=10, rs=1
# Bubble = 0.0795775  # 2D, Beta=20, rs=1

#############  3D  ######################################
Bubble = 0.0971916  # 3D, Beta=10, rs=1



def AngleIntegation(Data, l):
    # l: angular momentum
    shape = Data.shape[1:]
    Result = np.zeros(shape)
    for x in range(AngleBinSize):
        theta = np.arccos(AngleBin[x])
        if l==1:
            # Result += Data[x]*np.cos(l*AngleBin[x])/AngleBinSize
            Result += Data[x]*AngleBin[x]*( 2*l+1 )/AngleBinSize
        elif l==0:
            Result += Data[x, ...]*2.0/AngleBinSize
        elif l==2:
            Result += Data[x]*( 3*np.cos(2*theta)+1 )*( 2*l+1 )/(4*AngleBinSize)
        elif l==3:
            Result += Data[x]*( 5*np.cos(3*theta)+3*np.cos(theta) )*( 2*l+1 )/(8*AngleBinSize)
    return Result/2.0
    # return Result



def readData(folder, qIndex):
    global AngleBin, ExtMomBin, AngleBinSize, ExtMomBinSize
    global Data, Gamma4q, steps, chan

    files = os.listdir(folder)
    Num = 0
    Norm = 0
    Data0 = None


    files = os.listdir(folder)
    FileName = "vertex[0-9]_[0-3]_pid[0-9]+.dat"

    for f in files:
        if re.match(FileName, f):
            print("Reading ExtMomBin and AngleBin from: ", f)
            filePath = os.path.join(folder, f)
            with open(filePath, "r") as file:
                line0 = file.readline()
                Step = int(line0.split(":")[-1])/1000000
                line1 = file.readline()
                Norm += float(line1.split(":")[-1])
                line3 = file.readline()
                if AngleBin is None:
                    AngleBin = np.fromstring(line3.split(":")[1], sep=' ')
                    AngleBinSize = len(AngleBin)
                line4 = file.readline()
                if ExtMomBin is None:
                    ExtMomBin = np.fromstring(line4.split(":")[1], sep=' ')
                    ExtMomBinSize = len(ExtMomBin)
                    ExtMomBin /= kF
                break

    FileName = "weight_step"
    filePath = os.path.join(folder, FileName, "weight{0}.data".format(chan))

    with open(filePath, "r") as file:
        while True:
            line0 = file.readline()
            if not line0:
                break
            step = float(line0.split(":")[-1])
            line1 = file.readline()
            dat = line1.strip().split(" ")
            dat = np.array([float(i) for i in dat])
            dat = dat.reshape((AngleBinSize, ExtMomBinSize))
            dat = AngleIntegation(dat, l)
            Data.append(dat)
            steps.append(step)
            Gamma4q.append(dat[qIndex])



def ErrorPlot(p, x, d, color, marker, label=None, size=4, shift=False):
    p.plot(x, d, marker=marker, c=color, label=label,
           lw=1, markeredgecolor="None", linestyle="--", markersize=size)



def plot(flag):

    if flag == 1:
        ax = plt.subplot(1,1,1)

        MarkerList = ['s','o','v','d','x','^','<','>','*','2','3','4','H','+','D', '.', ',']
        ColorList = ['r', 'b', 'g', 'm', 'c', 'navy', 'y','lime','fuchsia', 'aqua','sandybrown','slategrey']
        ColorList = ColorList*40
        xaxel = [i for i in range(1, len(Gamma4q)+1)]

        # print(Gamma4q)
        ErrorPlot(ax, xaxel, Gamma4q, ColorList[5], MarkerList[1], "")

        # ax.set_xlim([0.0, ExtMomBin[-1]])
        ax.set_xlabel("Steps/$10^7$", size=size)
        ax.set_ylabel("$-\Gamma_4(\omega=0, q=0)$", size=size)
    elif flag == 2:
        ax = plt.subplot(1,1,1)

        MarkerList = ['s','o','v','d','x','^','<','>','*','2','3','4','H','+','D', '.', ',']
        ColorList = ['r', 'b', 'g', 'm', 'c', 'navy', 'y','lime','fuchsia', 'aqua','sandybrown','slategrey']
        ColorList = ColorList*40

        # print(len(Data), Data[1])
        ErrorPlot(ax, ExtMomBin, Data[1], ColorList[5], MarkerList[1], "")

        ax.set_xlim([0.0, ExtMomBin[-1]])
        ax.set_xlabel("$q/k_F$", size=size)
        ax.set_ylabel("$-\Gamma_4(\omega=0, q)$", size=size)


    plt.legend(loc=1, frameon=False, fontsize=size)
    plt.tight_layout()


def main():
    global Order, orderAccum, Lambda, kF

    qIndex = 0
    flag = 1
    figNum = 0

    if len(sys.argv) > 1:
        folders = sys.argv[1:]
        for folder in folders:
            inlistf = os.path.join(folder, "inlist")
            with open(inlistf, "r") as file:
                line = file.readline()
            para = line.split(" ")
            MaxOrder = int(para[0])
            BetaStr = para[1]
            Beta = float(BetaStr)
            rsStr = para[2]
            rs = float(rsStr)
            LambdaStr = para[3]
            Lambda = float(LambdaStr)
            TotalStep = float(para[5])

            kF = (9.0*np.pi/4.0)**(1.0/3.0)/rs
            Order = range(0, MaxOrder+1)
            orderAccum = MaxOrder

            figNum += 1
            print(folder)
            readData(folder, qIndex)
            plt.figure(figNum)
            plot(flag)

    plt.show()





if __name__ == "__main__":
    main()
