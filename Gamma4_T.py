#!/usr/bin/python
import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mat
import sys
import glob
import os
import re
import copy
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
    SPlot = True


ChanName = {0: "I", 1: "T", 2: "U", 3: "S"}
# 0: total, 1: order 1, ...
# Order = [0, 1, 2, 3]


Order = None
MaxOrder = 0
kF = None
Lambda = None
AngleBin = None
ExtMomBin = None
AngleBinSize = None
ExtMomBinSize = None





##############   2D    ##################################
###### Bare Green's function    #########################
# kF = np.sqrt(2.0)/rs  # 2D
# Bubble=0.11635  #2D, Beta=0.5, rs=1
# Bubble = 0.15916/2  # 2D, Beta=10, rs=1
# Bubble = 0.0795775  # 2D, Beta=20, rs=1

#############  3D  ######################################
Bubble = 0.0971916  # 3D, Beta=10, rs=1

Data = {}  # key: (order, channel)
DataAccum = {}
DataWithAngle = {}  # key: (order, channel)
DataErr = {}
DataAccumErr = {}
Gamma4q =[]
Gamma4qErr =[]
Gamma4qLog =[]
Gamma4qLogErr =[]
TList = []
TListLog = []


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


def Mirror(x, y):
    # print x
    # print len(x)
    x2 = np.zeros(len(x)*2)
    # x2[:len(x)] = -x[::-1]
    # x2[len(x):] = x
    x2[:len(x)] = x
    x2[len(x):] = -x[::-1]
    y2 = np.zeros(len(y)*2)
    y2[:len(y)] = y
    y2[len(y):] = y[::-1]
    return x2, y2

# def TauIntegration(Data):
#     return np.sum(Data, axis=-1) * \
#         Beta/kF**2/TauBinSize


def readData(folder, qIndex):
    global AngleBin, ExtMomBin, AngleBinSize, ExtMomBinSize
    global Data, DataAccum, DataWithAngle, DataErr, DataAccumErr
    global Gamma4q, Gamma4qErr, Gamma4qLog, Gamma4qLogErr
    global TList, MaxOrder

    for order in Order:
        for chan in Channel:

            files = os.listdir(folder)
            Num = 0
            Norm = 0
            Normf = 0
            Data0 = None
            DataList = []
            # if(order == 0):
            #     FileName = "vertex{0}_pid[0-9]+.dat".format(chan)
            # else:
            #     FileName = "vertex{0}_{1}_pid[0-9]+.dat".format(order, chan)

            FileName = "vertex{0}_{1}_pid[0-9]+.dat".format(order, chan)

            for f in files:
                if re.match(FileName, f):
                    print("Loading ", f)
                    with open(folder+f, "r") as file:
                        line0 = file.readline()
                        Step = int(line0.split(":")[-1])/1000000
                        # print "Step:", Step
                        line1 = file.readline()
                        Normf = float(line1.split(":")[-1])
                        Norm += Normf
                        line3 = file.readline()
                        if AngleBin is None:
                            AngleBin = np.fromstring(line3.split(":")[1], sep=' ')
                            AngleBinSize = len(AngleBin)
                        line4 = file.readline()
                        if ExtMomBin is None:
                            ExtMomBin = np.fromstring(line4.split(":")[1], sep=' ')
                            ExtMomBinSize = len(ExtMomBin)
                            ExtMomBin /= kF
                    Num += 1
                    d = np.loadtxt(folder+f)
                    if Data0 is None:
                        Data0 = d
                    else:
                        Data0 += d
                    dataf = d.reshape((AngleBinSize, ExtMomBinSize))/Normf
                    DataList.append(AngleIntegation(dataf, l))
            # print(order, chan,Norm, Data0)
            Data0 /= Norm
            Data0 = Data0.reshape((AngleBinSize, ExtMomBinSize))

            DataWithAngle[(order, chan)] = Data0

            # average the angle distribution
            Data[(order, chan)] = AngleIntegation(Data0, l)
            DataErr[(order, chan)] = np.std(np.array(DataList), axis=0)/np.sqrt(len(DataList))

    DataAccum = copy.deepcopy(Data)
    DataAccumErr = copy.deepcopy(DataErr)
    for order in range(2, orderAccum+1):
        for chan in Channel:
            DataAccum[(order, chan)] = DataAccum[(order-1, chan)]+Data[(order, chan)]
            DataAccumErr[(order, chan)] = np.power(DataAccumErr[(order-1, chan)],2)+np.power(DataErr[(order, chan)],2)
            DataAccumErr[(order, chan)] = np.sqrt(DataAccumErr[(order, chan)])
#    print(DataErr)
#    print(DataAccumErr)
    # print(DataAccum[(MaxOrder, Channel[0])][qIndex],DataAccumErr[(MaxOrder, Channel[0])][qIndex])
    Gamma4q.append( DataAccum[(MaxOrder, Channel[0])][qIndex] )
    Gamma4qErr.append( DataAccumErr[(MaxOrder, Channel[0])][qIndex] )
    Gamma4qLog.append( 1.0/DataAccum[(MaxOrder, Channel[0])][qIndex] )
    Gamma4qLogErr.append( DataAccumErr[(MaxOrder, Channel[0])][qIndex]/DataAccum[(MaxOrder, Channel[0])][qIndex] )


def ErrorPlot(p, x, d, e, color, marker, label=None, size=4, shift=False):
    if e == None:
        p.plot(x, d, marker=marker, c=color, label=label, lw=1, markeredgecolor="None", linestyle="--", markersize=size)
    else:
        p.errorbar(x,d, yerr=e,marker=marker, c=color, ecolor=color, label=label, lw=1, linestyle="--", markersize=size)


def getFileName(pre, para):
    return "./" + pre + "Order{0}_Beta{1}_lambda{2}/".format(para[0], para[1], para[3])


def f_1(x, A, B):
    return A*x + B


def main():
    global Order, orderAccum, Lambda, kF, TList, Gamma4q, MaxOrder
    MarkerList = ['s','o','v','d','x','^','<','>','*','2','3','4','H','+','D', '.', ',']
    ColorList = ['k', 'r', 'b', 'g', 'm', 'c', 'navy', 'y','lime','fuchsia', 'aqua','sandybrown','slategrey']


    qIndex = 0
    if len(sys.argv[1:]) == 1:
        inlistPath = sys.argv[1]
    else:
        print("Need to read the inlist file.")
        sys.exit(0)

    ax = plt.subplot(1,2,1)
    bx = plt.subplot(1,2,2)
    with open(inlistPath, "r") as file:
        lines = file.readlines()
    for line in lines:
        if len(line) < 2:
            break
        para = line.split(" ")
        MaxOrder = int(para[0]) + 1 #+1 for new S-resum method
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
        # MaxOrder = 2

        folderPre = "RenormAttrct_SChain_"
        folder = folderPre + "Order{0}_Beta{1}_lambda{2}/".format(para[0], para[1], para[3])
        print(folder)
        TList.append(1.0/Beta)
        TListLog.append(1.0/np.log(Beta))
        readData(folder, qIndex)
    ErrorPlot(ax, TList, Gamma4q, Gamma4qErr, ColorList[0], MarkerList[0], " ")
    ErrorPlot(bx, TListLog, Gamma4qLog, Gamma4qLogErr, ColorList[0], MarkerList[0], "numerical data")

    a1, b1 = optimize.curve_fit(f_1, TListLog[:5], Gamma4qLog[:5])[0]
    x1 = np.arange(0, max(TListLog), 0.01)
    y1 = a1*x1 + b1
    ErrorPlot(bx, x1, y1, None, ColorList[1], "None", "extrapolate")
    g = -1.0 / b1
    x0 = -1.0*b1/a1
    Tc = 1.0/np.exp(1.0/x0)
    rhoE = -1.0 * x0 / g
    print("Tc:{0} TF".format(Tc))
    print("g:{0}, rhoE:{1}".format(g, rhoE))
    amp = 5.0
    gTheory = 3.0*amp/2.0*(-1.0/kF**2-(Lambda**2+2*kF*kF)/(4*kF**4)*np.log((Lambda**2)/(Lambda**2+4*kF**2)))
    print("gTheory:{0}".format(gTheory))




    # ax.set_title(ChanName[Channel[0]]+"-channel")
    ax.set_xlabel("$T/T_F$")
    ax.set_ylabel("$\Gamma_4(q=0,l="+str(l)+")$")
    # bx.set_title(ChanName[Channel[0]]+"-channel")
    bx.set_xlabel("$1/\log(T_F/T)$")
    bx.set_ylabel("$1/\Gamma_4(q=0,l="+str(l)+")$")
    bx.set_ylim(bottom=0)
    plt.legend(loc=1, frameon=False, fontsize=size)
    plt.suptitle(ChanName[Channel[0]]+"-channel")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
