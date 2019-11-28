#include "global.h"
#include "utility/abort.h"
#include "utility/fmt/format.h"
#include "utility/vector.h"
#include "weight.h"
#include <array>
#include <iostream>
#include <stack>
#include <string>

using namespace diag;
using namespace std;
using namespace dse;

#define TIND(Shift, LTau, RTau) ((LTau - Shift) * MaxTauNum + RTau - Shift)

double weight::Evaluate(int LoopNum, int Channel) {
  if (LoopNum == 0) {
    // normalization
    // return VerQTheta.Interaction(Var.LoopMom[1], Var.LoopMom[2],
    // Var.LoopMom[0],
    //                              0.0, -2);
    return 1.0;
  } else {

    ver4 &Root = Ver4Root[LoopNum][Channel];
    if (Root.Weight.size() == 0)
      // empty vertex
      return 0.0;

    // if (Para.Counter == 12898) {
    //   cout << Root.ID << endl;
    // }


/*    if (Channel == dse::S) {
      *Root.LegK[INR] = Var.LoopMom[0] - Var.LoopMom[1];
      *Root.LegK[OUTR] = Var.LoopMom[0] - Var.LoopMom[2];
    // } else if (Channel == dse::T) {
    //   *Root.LegK[OUTL] = Var.LoopMom[1] - Var.LoopMom[0];
    //   *Root.LegK[OUTR] = Var.LoopMom[2] + Var.LoopMom[0];
    } else if (Channel == dse::U) {
      *Root.LegK[OUTL] = Var.LoopMom[2] + Var.LoopMom[0];
      *Root.LegK[OUTR] = Var.LoopMom[1] - Var.LoopMom[0];
    } else {
      *Root.LegK[OUTL] = Var.LoopMom[1] - Var.LoopMom[0];
      *Root.LegK[OUTR] = Var.LoopMom[2] + Var.LoopMom[0];
    } */

    *Root.LegK[INR] = Var.LoopMom[0] - Var.LoopMom[1];
    *Root.LegK[OUTR] = Var.LoopMom[0] - Var.LoopMom[2];


    Vertex4(Root);

    double Weight = 0.0;
    for (auto &w : Root.Weight)
      Weight += w;
    // if (LoopNum == 3 && Channel == dse::I) {
    //   cout << "loopnum: " << Root.LoopNum << endl;
    //   cout << "channel: " << Root.Channel[0] << endl;
    //   cout << Weight << endl;
    // }
    // cout << count << endl;
    return Weight / pow(2.0 * PI, D * LoopNum);
    // return Weight;
  }
}

void weight::Ver0(ver4 &Ver4) {
  array<momentum *, 4> &K = Ver4.LegK;
  // momentum DiQ = *K[INL] - *K[OUTL];
  // momentum ExQ = *K[INL] - *K[OUTR];
  Ver4.Weight[0] = VerQTheta.Interaction(K, 0.0, 0);
  // Ver4.Weight[0] = 1.0 / Para.Beta;
  if (Ver4.RexpandBare) {
    // cout << Ver4.T[0][INR] << ", " << Ver4.T[0][INL] << endl;
    // double Tau = Var.Tau[Ver4.T[1][INR]] - Var.Tau[Ver4.T[1][INL]];
    // cout << Ver4.T[1][INR] << ", " << Ver4.T[1][INL] << "; " <<
    // Ver4.T[2][INR]
    //      << ", " << Ver4.T[2][INL] << endl;
    Ver4.Weight[0] += +VerQTheta.Interaction(K, 0.0, 1);
    // Ver4.Weight[1] = 0.0;
    // Ver4.Weight[2] = 0.0;

    // Ver4.Weight[1] = +VerQTheta.Interaction(K, DiQ, Tau, 1);
    // Ver4.Weight[2] = -VerQTheta.Interaction(K, ExQ, Tau, 1);
  }
  return;
}
void weight::Vertex4(dse::ver4 &Ver4) {
  // cout << Ver4.LoopNum << endl;
  if (Ver4.LoopNum == 0) {
    Ver0(Ver4);
  } else {
    for (auto &w : Ver4.Weight)
      w = 0.0;
    ChanUST(Ver4);
    if (Ver4.LoopNum >= 3)
      ChanI(Ver4);
  }
  return;
}

void weight::ChanUST(dse::ver4 &Ver4) {
  double Weight = 0.0;
  double Ratio;
  array<momentum *, 4> &LegK0 = Ver4.LegK;

  // if (Ver4.ContainProj) {
  // }

  for (auto &bubble : Ver4.Bubble) {
    auto &G = bubble.G;
    const momentum &K0 = *G[0].K;
    int InTL = bubble.InTL;

    for (auto &chan : bubble.Channel)
      if (bubble.IsProjected)
        bubble.ProjFactor[chan] = 0.0;
      else
        bubble.ProjFactor[chan] = 1.0;

//    double ExpINL = pow((*LegK0[INL]).norm()-Para.Kf,2) * Para.Beta;
//    double ExpINR = pow((*LegK0[INR]).norm()-Para.Kf,2) * Para.Beta;
//    double ExpOUTL = pow((*LegK0[OUTL]).norm()-Para.Kf,2) * Para.Beta;
//    double ExpOUTR = pow((*LegK0[OUTR]).norm()-Para.Kf,2) * Para.Beta;
    double ExpINL = abs((*LegK0[INL]).norm()-Para.Kf); //* sqrt(Para.Beta);
    double ExpINR = abs((*LegK0[INR]).norm()-Para.Kf); //* sqrt(Para.Beta);
    double ExpOUTL = abs((*LegK0[OUTL]).norm()-Para.Kf); //* sqrt(Para.Beta);
    double ExpOUTR = abs((*LegK0[OUTR]).norm()-Para.Kf); //* sqrt(Para.Beta);

    if (bubble.IsProjected) {
      double extKFactor = exp(-(ExpINL+ExpINR+ExpOUTL+ExpOUTR)/decayExtK);
      momentum InMom = *LegK0[INL] - *LegK0[INR];
      momentum OutMom = *LegK0[OUTL] - *LegK0[OUTR];
      Ratio = Para.Kf / InMom.norm();
      InMom = InMom * Ratio;
      Ratio = Para.Kf / OutMom.norm();
      OutMom = OutMom * Ratio;
      double DirQ = (*LegK0[INL] - *LegK0[OUTL]).norm();
      double ExQ = (*LegK0[INL] - *LegK0[OUTR]).norm();
      double InQ = (*LegK0[INL] + *LegK0[INR]).norm();

      if(bubble.HasT && !OnlySProj){
        if (DirQ < 1.0 * Para.Kf) {
          Ratio = Para.Kf / (*LegK0[INL]).norm();
          *bubble.LegK[T][INL] = *LegK0[INL] * Ratio;
          Ratio = Para.Kf / (*LegK0[INR]).norm();
          *bubble.LegK[T][INR] = *LegK0[INR] * Ratio;
          *bubble.LegK[T][OUTL] = *LegK0[INL] * (-1.0);
          *bubble.LegK[T][OUTR] = *LegK0[INR] * (-1.0);
          bubble.ProjFactor[T] = exp(-DirQ * DirQ / decayTU) * extKFactor;
        } 
/*      if(InQ < 1.0*Para.Kf){
        *bubble.LegK[T][INL] = InMom;
        *bubble.LegK[T][OUTL] = OutMom;
        *bubble.LegK[T][INR] = *bubble.LegK[T][INL] * (-1.0);
        *bubble.LegK[T][OUTR] = *bubble.LegK[T][OUTL] * (-1.0);
//        bubble.ProjFactor[T] = exp(-InQ * InQ / Para.Ef / decayS) * extKFactor;
        bubble.ProjFactor[T] = exp(-InQ * InQ / decayS) * extKFactor;
       } */
      }
      if(bubble.HasU && !OnlySProj){
        if (ExQ < 1.0 * Para.Kf) {
          Ratio = Para.Kf / (*LegK0[INL]).norm();
          *bubble.LegK[U][INL] = *LegK0[INL] * Ratio;
          Ratio = Para.Kf / (*LegK0[INR]).norm();
          *bubble.LegK[U][INR] = *LegK0[INR] * Ratio;
          *bubble.LegK[U][OUTL] = *LegK0[INR] * (-1.0);
          *bubble.LegK[U][OUTR] = *LegK0[INL] * (-1.0);
          bubble.ProjFactor[U] = exp(-ExQ * ExQ / decayTU) * extKFactor;
        } 
/*        if(InQ < 1.0*Para.Kf){
          *bubble.LegK[U][INL] = InMom;
          *bubble.LegK[U][OUTL] = OutMom;
          *bubble.LegK[U][INR] = *bubble.LegK[U][INL] * (-1.0);
          *bubble.LegK[U][OUTR] = *bubble.LegK[U][OUTL] * (-1.0); 
//          bubble.ProjFactor[U] = exp(-InQ * InQ / Para.Ef / decayS) * extKFactor;
          bubble.ProjFactor[U] = exp(-InQ * InQ / decayS) * extKFactor;
        } */
      }
      if (bubble.HasS && !OnlyTUProj){
        if(InQ < 1.0*Para.Kf){
          *bubble.LegK[S][INL] = InMom;
          *bubble.LegK[S][OUTL] = OutMom;
          *bubble.LegK[S][INR] = *bubble.LegK[S][INL] * (-1.0);
          *bubble.LegK[S][OUTR] = *bubble.LegK[S][OUTL] * (-1.0); 
//          bubble.ProjFactor[S] = exp(-InQ * InQ / Para.Ef / decayS) * extKFactor;
          bubble.ProjFactor[S] = exp(-InQ * InQ/ decayS) * extKFactor;
        }
      }
    }

    for (auto &chan : bubble.Channel) {
      array<momentum *, 4> &LegK = bubble.LegK[chan];
      if (chan == T)
        *G[T].K = *LegK[OUTL] + K0 - *LegK[INL];
      else if (chan == U)
        *G[U].K = *LegK[OUTR] + K0 - *LegK[INL];
      else if (chan == S)
        *G[S].K = *LegK[INL] + *LegK[INR] - K0;
    }
    for (int lt = InTL; lt < InTL + Ver4.TauNum - 1; ++lt)
      for (int rt = InTL + 1; rt < InTL + Ver4.TauNum; ++rt) {
        double dTau = Var.Tau[rt] - Var.Tau[lt];
        G[0](lt, rt) = Fermi.Green(dTau, K0, UP, 0, Var.CurrScale);
        for (auto &chan : bubble.Channel) {
          // if (chan > 3)
          //   ABORT("too many chan " << chan);
          if (abs(bubble.ProjFactor[chan]) > EPS)
            if (chan == S)
              // LVer to RVer
              G[S](lt, rt) = Fermi.Green(dTau, *G[S].K, UP, 0, Var.CurrScale);
            else
              // RVer to LVer
              G[chan](rt, lt) =
                  Fermi.Green(-dTau, *G[chan].K, UP, 0, Var.CurrScale);
        }
      }

    // for vertex4 with one or more loops
    for (auto &pair : bubble.Pair) {
      if (abs(bubble.ProjFactor[pair.Channel]) < EPS)
        continue;
      ver4 &LVer = pair.LVer;
      ver4 &RVer = pair.RVer;
      Vertex4(LVer);
      Vertex4(RVer);

      for (auto &map : pair.Map) {
        Weight = pair.SymFactor * bubble.ProjFactor[pair.Channel];
        Weight *= G[0](map.G0T) * G[pair.Channel](map.GT);
        // cout << Weight << endl;
        Weight *= LVer.Weight[map.LVerTidx] * RVer.Weight[map.RVerTidx];
        // cout << Weight << endl;
        // cout << endl;
        Ver4.Weight[map.Tidx] += Weight;
      }
    }
  }
}

void weight::ChanI(dse::ver4 &Ver4) {
  if (Ver4.LoopNum != 3)
    return;
  for (auto &Env : Ver4.Envelope) {
    const momentum &InL = *Env.LegK[INL];
    const momentum &OutL = *Env.LegK[OUTL];
    const momentum &InR = *Env.LegK[INR];
    const momentum &OutR = *Env.LegK[OUTR];

    auto &G = Env.G;

    *G[3].K = *G[0].K + *G[1].K - InL;
    *G[4].K = *G[1].K + *G[2].K - OutL;
    *G[5].K = *G[0].K + InR - *G[2].K;
    *G[6].K = *G[1].K + *G[2].K - OutR;
    *G[7].K = *G[2].K + OutR - *G[0].K;
    *G[8].K = *G[2].K + OutL - *G[0].K;

    for (auto &g : Env.G)
      g.Weight = Fermi.Green(Var.Tau[g.OutT] - Var.Tau[g.InT], *(g.K), UP, 0,
                             Var.CurrScale);

    for (auto &subVer : Env.Ver)
      Vertex4(subVer);

    double Weight = 0.0;
    double ComWeight = 0.0;
    for (auto &map : Env.Map) {
      auto &SubVer = Env.Ver;
      auto &GT = map.GT;
      auto &G = Env.G;
      ComWeight = G[0].Weight * G[1].Weight * G[2].Weight * G[3].Weight;
      // cout << "G: " << ComWeight << endl;
      ComWeight *= SubVer[0].Weight[map.LDVerTidx];
      // cout << "Ver: " << SubVer[0].Weight[map.LDVerT] << endl;
      // cout << "T: " << map.LDVerT << endl;

      Weight = Env.SymFactor[0] * ComWeight;
      Weight *= SubVer[1].Weight[map.LUVerTidx];
      Weight *= SubVer[3].Weight[map.RDVerTidx];
      Weight *= SubVer[6].Weight[map.RUVerTidx];
      Weight *= G[4].Weight * G[5].Weight;
      Ver4.Weight[map.Tidx[0]] += Weight;

      Weight = Env.SymFactor[1] * ComWeight;
      Weight *= SubVer[2].Weight[map.LUVerTidx];
      Weight *= SubVer[3].Weight[map.RDVerTidx];
      Weight *= SubVer[7].Weight[map.RUVerTidx];
      Weight *= G[6].Weight * G[5].Weight;
      Ver4.Weight[map.Tidx[1]] += Weight;
      // cout << Weight << endl;

      Weight = Env.SymFactor[2] * ComWeight;
      Weight *= SubVer[1].Weight[map.LUVerTidx];
      Weight *= SubVer[4].Weight[map.RDVerTidx];
      Weight *= SubVer[8].Weight[map.RUVerTidx];
      Weight *= G[4].Weight * G[7].Weight;
      Ver4.Weight[map.Tidx[2]] += Weight;
      // cout << Weight << endl;

      Weight = Env.SymFactor[3] * ComWeight;
      Weight *= SubVer[2].Weight[map.LUVerTidx];
      Weight *= SubVer[5].Weight[map.RDVerTidx];
      Weight *= SubVer[9].Weight[map.RUVerTidx];
      Weight *= G[6].Weight * G[8].Weight;
      Ver4.Weight[map.Tidx[3]] += Weight;
      // cout << Weight << endl;

      // if (map.LDVerT == 0 && map.LUVerT == 0 && map.RDVerT == 0 &&
      //     map.RUVerT == 0) {
      // cout << "Com: " << ComWeight << endl;
      // cout << "G[4]: " << G[4](GT[4]) << endl;
      // cout << "G[5]: " << G[5](GT[5]) << endl;
      // cout << SubVer[1].Weight[map.LUVerT] << endl;
      // cout << SubVer[3].Weight[map.RDVerT] << endl;
      // cout << SubVer[6].Weight[map.RUVerT] << endl;
      // cout << "First: " << Weight << endl;
      // }
    }
  }

  return;
}
