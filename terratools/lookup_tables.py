import numpy as np
from .utils import norm_vals, int_linear
from scipy.interpolate import interp2d, griddata
import os



class SeismicLookupTable:
    def __init__(self,table_path):
        """
        Inputs: table_path = '/path/to/data/table/'

        Example: basalt = lookup_tables.SeismicLookupTable('/path/to/basalt/table.dat')
        """
        try:
            self.table=np.genfromtxt(f'{table_path}')
        except:
            self.table=np.genfromtxt(f'{table_path}',skip_header=1)


        self.P = self.table[:,0]
        self.T = self.table[:,1]
        self.pres=np.unique(self.table[:,0])
        self.temp=np.unique(self.table[:,1])
        self.t_max=np.max(self.temp)
        self.t_min=np.min(self.temp)
        self.p_max=np.max(self.pres)
        self.p_min=np.min(self.pres)
        self.pstep=np.size(self.temp)

        self.Vp=np.zeros((len(self.temp),len(self.pres)))
        self.Vs=np.zeros((len(self.temp),len(self.pres)))
        self.Vp_an=np.zeros((len(self.temp),len(self.pres)))
        self.Vs_an=np.zeros((len(self.temp),len(self.pres)))
        self.Vphi=np.zeros((len(self.temp),len(self.pres)))
        self.Dens=np.zeros((len(self.temp),len(self.pres)))
        self.Qs=np.zeros((len(self.temp),len(self.pres)))
        self.T_sol=np.zeros((len(self.temp),len(self.pres)))
        self.fields = {'vp': 2, 'vs': 3, 'vp_ani': 4, 'vs_ani': 5, 'vphi': 6, 'density': 7, 'qs': 8, 't_sol': 9}


        for i, p in enumerate(self.pres):
            self.Vp[:,i]=self.table[0+(i*self.pstep):self.pstep+(i*self.pstep),2]
            self.Vs[:,i]=self.table[0+(i*self.pstep):self.pstep+(i*self.pstep),3]
            self.Vp_an[:,i]=self.table[0+(i*self.pstep):self.pstep+(i*self.pstep),4]
            self.Vs_an[:,i]=self.table[0+(i*self.pstep):self.pstep+(i*self.pstep),5]
            self.Vphi[:,i]=self.table[0+(i*self.pstep):self.pstep+(i*self.pstep),6]
            self.Dens[:,i]=self.table[0+(i*self.pstep):self.pstep+(i*self.pstep),7]
            self.Qs[:,i]=self.table[0+(i*self.pstep):self.pstep+(i*self.pstep),8]
            self.T_sol[:,i]=self.table[0+(i*self.pstep):self.pstep+(i*self.pstep),9]





#################################################
#Need to get temp, pres, comp at given point.
#Pressure could come from PREM or from simulation
#Comp will be in 3 component mechanical mixture
#We will then find the
#################################################

    def get_vals(self,pval,tval):
        """
        Inputs: pval=pressure at point
                tval=temperature at point
        Returns: Vp, Vs, Vp_an, Vs_an, Vphi, Dens
        For a given temperature and pressure, find the locations of the
        upper and lower bounds in a seismic conversion table.

        Example: vp, vs, vp_an, vs_an, vphi, dens = basalt.get_vals(P,T)
         """


        #First we find the upper and lower pressure bounds
        #Include some catches in case given pressure is out of bounds
        if pval<np.min(self.table[:,0]):
            print(f'#Error - Pressure {pval} less than minimum value in lookup tables')
            pu=np.min(self.table[:,0])
            pl=pu
        elif pval>np.max(self.table[:,0]):
            print(f'#Warning - Pressure {pval} exceeds maximum value in lookup tables')
            pu=np.max(self.table[:,0])
            pl=pu
        else:
            #Find the lower (pl) and upper (pu) bounds of pressure
            ipx = np.where(np.abs(self.table[:,0]-pval) == np.min(np.abs(self.table[:,0]-pval)))[0]
            if self.table[ipx[0],0]-pval < 0 :
                pl=self.table[ipx[0],0] ; pu=self.table[ipx[0]+self.pstep,0]
            else:
                pl=p2=self.table[ipx[0]-self.pstep,0] ; pu=self.table[ipx[0],0]



        #Now find upper and lower temperature bounds
        #Include catches in case given temp is out of bounds
        if tval<np.min(self.table[:,1]):
            print(f'#Warning - Temperature {tval} less than minimum value in lookup tables')
            tu=np.min(self.table[:,1])
            tl=tu
        elif tval>np.max(self.table[:,1]):
            print(f'#Warning - Temperature {tval} exceeds maximum value in lookup tables')
            tu=np.max(self.table[:,1])
            tl=tu
        else:
            #Find the lower (tl) and upper (tu) bounds of temperature
            itx = np.where(np.abs(self.table[:,1]-tval) == np.min(np.abs(self.table[:,1]-tval)))[0]
            if self.table[itx[0],1]-tval < 0 :
                tl=self.table[itx[0],1] ; tu=self.table[itx[0]+1,1]
            else:
                tl=self.table[itx[0]-1,1] ; tu=self.table[itx[0],1]


        #Now find the 4 indices (pl,tl) (pl,tu) (pu,tl) (pu,tu)
        ipltl=np.intersect1d(np.where(self.table[:,0]==pl),np.where(self.table[:,1]==tl))
        ipltu=np.intersect1d(np.where(self.table[:,0]==pl),np.where(self.table[:,1]==tu))

        iputl=np.intersect1d(np.where(self.table[:,0]==pu),np.where(self.table[:,1]==tl))
        iputu=np.intersect1d(np.where(self.table[:,0]==pu),np.where(self.table[:,1]==tu))

        #Normalise input against bounds
        tnorm=norm_vals(tval,tu,tl)
        pnorm=norm_vals(pval,pu,pl)


        Vp=int_linear(self.table[ipltl,2],self.table[ipltu,2],
                self.table[iputl,2],self.table[iputu,2],tnorm,pnorm)
        Vs=int_linear(self.table[ipltl,3],self.table[ipltu,3],
                self.table[iputl,3],self.table[iputu,3],tnorm,pnorm)
        Vp_an=int_linear(self.table[ipltl,4],self.table[ipltu,4],
                self.table[iputl,4],self.table[iputu,4],tnorm,pnorm)
        Vs_an=int_linear(self.table[ipltl,5],self.table[ipltu,5],
                self.table[iputl,5],self.table[iputu,5],tnorm,pnorm)
        Vphi=int_linear(self.table[ipltl,6],self.table[ipltu,6],
                self.table[iputl,6],self.table[iputu,6],tnorm,pnorm)
        Dens=int_linear(self.table[ipltl,7],self.table[ipltu,7],
                self.table[iputl,7],self.table[iputu,7],tnorm,pnorm)

        return Vp, Vs, Vp_an, Vs_an, Vphi, Dens


    def interp_grid(self,press,temps,field):
        """
        Given a range of pressures and temperatures, return a 2D
        grid of values of the field of choice.


        Inputs: press = pressures along P axis
                temps = temperatures along T axis
                field = property to interpolate eg. Vs
        Returns: interpolated values of a given table property
                 on a grid defined by press and temps

        eg. basalt.interp([pressures],[temperature],'Vs')
        """

        # get column index for field of interest
        i_field = self.fields[field.lower()]

        # set up interp2d object
        grid = interp2d(self.P,self.T,self.table[:,i_field], kind='linear')

        out = grid(press, temps)

        return out



    def interp_points(self,points,field):
        """
        Takes in pressure, temperature points in a 2D array and returns
        a 1D array of interpolated points to those pressures and
        temperatures.


        Inputs: points = pressure-temperature points in a 2D array.
                         The first column should be pressure and the
                         second column temperature.
                field = property to interpolate eg. Vs

        Returns:
        For a given table property (eg. Vs) return interpolated values
        for pressures and temperatures
        eg. basalt.interp_points(list(zip(pressures,temperature)),'Vs')
        """

        # get column index for field of interest
        i_field = self.fields[field.lower()]

        # set up interp2d object
        grid = griddata((self.P,self.T),self.table[:,i_field], points, method='linear')

        return grid




def harmonic_mean_comp(bas,lhz,hzb,bas_fr,lhz_fr,hzb_fr):
    """
    Input: bas = data for basaltic composition (eg. basalt.Vs)
           lhz = data for lherzolite composition
           hzb = data for harzburgite composition
           bas_fr = basalt fraction
           lhz_fr = lherzolite fraction
           hzb_fr = harzburgite fraction
    Returns: hmean = harmonic mean of input values

    bas, lhz, hzb must be of equal length
    This routine assumes 3 component mechanical mixture

    """
    m1=(1./bas)*bas_fr
    m2=(1./lhz)*lhz_fr
    m3=(1./hzb)*hzb_fr

    hmean=1/(m1+m2+m3)

    return hmean

def linear_interp_1d(vals1, vals2, c1, c2, cnew):
    """
    Inputs: v1 = data for composition 1
            v2 = data for composition 2
            c1 = C-value for composition 1
            c2 = C-value for composition 2
            cnew  = C-value(s) for new composition(s)

    Returns: interpolated values for compostions cnew
    """

    #Normalise table c-values
    cmin=min(c1,c2)
    cmax=max(c1,c2)
    if c1==cmin :
        v1=vals1
        v2=vals2
    else:
        v1=vals2
        v2=vals1
    interpolated = interp1d(np.array([cmin,cmax]),[v1.flatten(),v2.flatten()],
                            fill_value='extrapolate',axis=0)

    return interpolated(cnew)


#class MultiComponent:
#    def __init__(self, hzb_tab, lhz_tab, bas_tab, pt):
#        """
#        Inputs: hzb_tab = harzburgite table
#                lhz_tab = lherzolite table
#                bas_tab = basalt table
#                pt      = array with dimensions (n,2) where column 0
#                          is pressures and column 1 is temperatures
#        """
#
#
#        ipx = np.where(np.abs(bas_tab[:,0]-pt[:,0]) == np.min(np.abs(self.table[:,0]-pval)))[0]
