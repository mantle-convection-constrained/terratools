import numpy as np

def norm_vals(v,vu,vl):
    """
    Input: v = The absolute input value
          vu = Upper bound in table
          vl = Lower bound in table
    Returns: vnorm = Normalised value, +ve from vl
    """
    vrange=vu-vl
    if vrange==0. :
        vnorm=0.5
    else:
        vnorm=(v-vl)/vrange

#    vnorm=np.nan_to_num(vnorm,copy=False,posinf=0.5,neginf=0.5,nan=0.5)


    return vnorm


def int_linear(v1,v2,v3,v4,tn,pn):
    """
    Input: v1 = value at ipltl
           v2 = value at ipltu
           v3 = value at iputl
           v4 = value at iputu
           tn = normalised pressure
           pn = normalised temperature
    Returns: vinterp = interpolated value

    3 ---------- 4 ^
    |            | |pn
    |        .   | |
    1 ---------- 2
       ----->
        tn

    """
    v1n = (1-tn) * (1-pn) * v1
    v2n = tn * (1-pn) * v2
    v3n = (1-tn) * pn * v3
    v4n = tn * pn * v4

    vinterp = v1n + v2n + v3n + v4n

    return vinterp



def harmonic_mean_comp(bas,lhz,hzb,bas_fr,lhz_fr,hzb_fr):
    """
    Input: bas = value for basaltic composition
           lhz = value for lherzolite composition
           hzb = value for harzburgite composition
           bas_fr = basalt fraction
           lhz_fr = lherzolite fraction
           hzb_fr = harzburgite fraction
    Returns: hmean = harmonic mean of input values
    """
    m1=1./bas
    m2=1./lhz
    m3=1./hzb

    hmean=1/(m1+m2+m3)

    return hmean
