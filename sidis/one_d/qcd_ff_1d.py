import numpy as np
from ..qcdlib import params as par
from ..qcdlib import config_loader as cfg
from ..qcdlib.alphaS  import ALPHAS
from ..qcdlib.dglap   import DGLAP
from ..qcdlib.kernels import KERNELS
from ..qcdlib.mellin  import MELLIN
from ..qcdlib.special import beta

class FF_PIP:
    
    """
    This class computes the 1d QCF, called a fragmentation function (FF) for all flavors as a function of (z,Q2) for a pi^+ meson.
    Inputs:
        class instance mellin = contains routine to compute Mellin moments and inversions
        class instance alphaS = contains routine to evaluate the strong coupling constant at any scale
    """

    def __init__(self,mellin,alphaS):
  
        self.spl='upol_ff'
        self.Q20=cfg.Q20
        self.mc2=par.mc2
        self.mb2=par.mb2
  
        self.kernel=KERNELS(mellin,self.spl) #--contains splitting functions needed for evolution in Q2
        self.dglap=DGLAP(mellin,alphaS,self.kernel,'truncated',cfg.dglap_order) #--describes the evolution in Q2 of the 1d QCF
        self.mellin=mellin
 
        self.set_params()
        self.get_current_par_array()
        self.setup(self.current_par)
  
    def set_params(self):
        """
        D(z) = norm * z**a * (1-z)**b * (1+c*z**0.5+d*z)
        """
        self.params={}
  
        #--first shapes -- individual flavors (from pdf_ff_params.txt, format: N, a, b, c, d)
        self.params['g1']   = [6.4479183915853566e-06, 1.088682720159509, 5.39543934069994, -0.4545882053452076, -0.12310463103820038]
        self.params['u1']   = [0.037133548882077594, 0.5107246235720175, 5.405612887350822, -0.6471481806543868, -0.5327102329768223]
        self.params['d1']   = [0.115826210145427, -0.4661418824775969, 9.878756919937004, -2.790284375497949, 1.3745691824429274]
        self.params['s1']   = [0.2953441679057506, -0.6493300183984108, 4.4479391625899565, -0.4389726815746563, 0.7042631284412002]
        self.params['c1']   = [0.11300695032419823, -0.8896243845942003, 3.7821163200923746, -1.25116436015642, -1.1300698937418685]
        self.params['b1']   = [0.11403609145892839, -0.2799781981862734, 5.926483006903505, -2.2738500492053806, 0.0114705372755697]
        self.params['ub1']  = [0.115826210145427, -0.4661418824775969, 9.878756919937004, -2.790284375497949, 1.3745691824429274]
        self.params['db1']  = [0.037133548882077594, 0.5107246235720175, 5.405612887350822, -0.6471481806543868, -0.5327102329768223]
        self.params['sb1']  = [0.2953441679057506, -0.6493300183984108, 4.4479391625899565, -0.4389726815746563, 0.7042631284412002]
        self.params['cb1']  = [0.11300695032419823, -0.8896243845942003, 3.7821163200923746, -1.25116436015642, -1.1300698937418685]
        self.params['bb1']  = [0.11403609145892839, -0.2799781981862734, 5.926483006903505, -2.2738500492053806, 0.0114705372755697]

        #--second shapes (from pdf_ff_params.txt, format: N, a, b, c, d)
        self.params['g2']   = [0.12483162802592059, 1.810446051834352, 3.038903897603831, -0.834137162637043, -0.5424992878371933]
        self.params['u2']   = [0.21248500740875143, -1.0211919573747366, 0.36880605188991294, -1.0777608690103422, 0.15636899377791064]
        self.params['d2']   = [0.10604483082193661, 0.6324048523595832, 2.8904016769191507, -2.3715498378528963, 1.4805161310823294]
        self.params['s2']   = [0.02363867284478755, 2.1983371395612483, 3.8495423937824125, 2.463121683765423, 1.0680454577859622]
        self.params['c2']   = [0.13247236447769317, -0.07591681992831066, 2.9202614257990827, -0.43223551252552295, 0.13460619196738205]
        self.params['b2']   = [0.10124503210895401, -0.13689367786624848, 5.97625076649095, -2.388109588179393, -0.04709015361856303]
        self.params['ub2']  = [0.10604483082193661, 0.6324048523595832, 2.8904016769191507, -2.3715498378528963, 1.4805161310823294]
        self.params['db2']  = [0.21248500740875143, -1.0211919573747366, 0.36880605188991294, -1.0777608690103422, 0.15636899377791064]
        self.params['sb2']  = [0.02363867284478755, 2.1983371395612483, 3.8495423937824125, 2.463121683765423, 1.0680454577859622]
        self.params['cb2']  = [0.13247236447769317, -0.07591681992831066, 2.9202614257990827, -0.43223551252552295, 0.13460619196738205]
        self.params['bb2']  = [0.10124503210895401, -0.13689367786624848, 5.97625076649095, -2.388109588179393, -0.04709015361856303]

        """
        These fixed parameters are not to be changed during the fit.
        They specify the index of the array in self.params.
        """
        #--fixed params
        self.fixed={}
        self.fixed['g1']=[3,4]
        self.fixed['u1']=[3,4]
        self.fixed['d1']=[3,4]
        self.fixed['s1']=[3,4]
        self.fixed['c1']=[3,4]
        self.fixed['b1']=[3,4]
        self.fixed['ub1']=[0,1,2,3,4] #--will get fixed by fixed_from_others
        self.fixed['db1']=[0,1,2,3,4] #--will get fixed by fixed_from_others
        self.fixed['sb1']=[0,1,2,3,4] #--will get fixed by fixed_from_others
        self.fixed['cb1']=[0,1,2,3,4] #--will get fixed by fixed_from_others
        self.fixed['bb1']=[0,1,2,3,4] #--will get fixed by fixed_from_others
        
        self.fixed['g2']=[3,4]
        self.fixed['u2']=[3,4]
        self.fixed['d2']=[3,4]
        self.fixed['s2']=[0,1,2,3,4]
        self.fixed['c2']=[3,4]
        self.fixed['b2']=[3,4]
        self.fixed['ub2']=[0,1,2,3,4] #--will get fixed by fixed_from_others
        self.fixed['db2']=[0,1,2,3,4] #--will get fixed by fixed_from_others
        self.fixed['sb2']=[0,1,2,3,4]
        self.fixed['cb2']=[0,1,2,3,4] #--will get fixed by fixed_from_others
        self.fixed['bb2']=[0,1,2,3,4] #--will get fixed by fixed_from_others
        self.features=['g1','u1','d1','s1','c1','b1','ub1','db1','sb1','cb1','bb1','g2','u2','d2','s2','c2','b2','ub2','db2','sb2','cb2','bb2']
        
        self.fixed_from_others = {}
        self.fixed_from_others['ub1'] = 'd1'
        self.fixed_from_others['db1'] = 'u1'
        self.fixed_from_others['sb1'] = 's1'
        self.fixed_from_others['cb1'] = 'c1'
        self.fixed_from_others['bb1'] = 'b1'
        self.fixed_from_others['ub2'] = 'd2'
        self.fixed_from_others['db2'] = 'u2'
        self.fixed_from_others['sb2'] = 's2'
        self.fixed_from_others['cb2'] = 'c2'
        self.fixed_from_others['bb2'] = 'b2'

        self._parmin=[0,-1.89, 0,-0.5,-0.5] #--previously [-10,-2, 0,-10,-10]
        self._parmax=[1, 10,20,+10,+10] #--previously [+10,10,10,+10,+10]
        
    def get_current_par_array(self):
        
        self.current_par=[]
        self.order=[]
        self.parmin=[]
        self.parmax=[]
        for _ in self.features:
            for i in range(len(self.params[_])):
                if _ in self.fixed:
                    if i in self.fixed[_]: continue
                self.current_par.append(self.params[_][i])

                #--to fix entire parameters from another flavor
                for k in self.fixed_from_others:
                    if self.fixed_from_others[k] == _:
                        self.params[k][i] = self.params[_][i]
                
                self.order.append(_)
                self.parmin.append(self._parmin[i])
                self.parmax.append(self._parmax[i])
        
        self.current_par=np.array(self.current_par)
        self.parmin=np.array(self.parmin)
        self.parmax=np.array(self.parmax)
        
        return self.current_par

    def update_params(self,par_array):
        
        """
        This function is essential to pass a new array of parameters into the class during a fit.
        input:
            par_array: an array of the free parameters; len(par_array)=total number of free parameters across the flavors.
        """
        
        cnt=0
        for _ in self.features:
            for i in range(len(self.params[_])):
                if _ in self.fixed:
                    if i in self.fixed[_]: continue
                self.params[_][i]=par_array[cnt]

                #--to fix entire parameters from another flavor
                for k in self.fixed_from_others:
                    if self.fixed_from_others[k] == _:
                        self.params[k][i] = par_array[cnt]                
                
                cnt+=1
                
    def set_sumrules(self):
        pass
  
    def set_moms(self):
  
        moms={}
        moms['g']   = self.get_moments('g1') + self.get_moments('g2')
        moms['u1']  = self.get_moments('u1') + self.get_moments('u2')
        moms['d1']  = self.get_moments('d1') + self.get_moments('d2')
        moms['s1']  = self.get_moments('s1') + self.get_moments('s2')
        moms['c1']  = self.get_moments('c1') + self.get_moments('c2')
        moms['b1']  = self.get_moments('b1') + self.get_moments('b2')
        moms['ub1'] = self.get_moments('ub1') + self.get_moments('ub2')
        moms['db1'] = self.get_moments('db1') + self.get_moments('db2')
        moms['sb1'] = self.get_moments('sb1') + self.get_moments('sb2')
        moms['cb1'] = self.get_moments('cb1') + self.get_moments('cb2')
        moms['bb1'] = self.get_moments('bb1') + self.get_moments('bb2')

        moms['up']=moms['u1']+moms['ub1']
        moms['dp']=moms['d1']+moms['db1']
        moms['sp']=moms['s1']+moms['sb1']
        moms['cp']=moms['c1']+moms['cb1']
        moms['bp']=moms['b1']+moms['bb1']
        moms['um']=moms['u1']-moms['ub1']
        moms['dm']=moms['d1']-moms['db1']
        moms['sm']=moms['s1']-moms['sb1']
        moms['cm']=moms['c1']-moms['cb1']
        moms['bm']=moms['b1']-moms['bb1']

        self.moms0=moms
        self.get_BC(moms)
  
    def setup(self,par_array):
        """
        This function should be called with each update of parameters.
        The propagation of parameters sets new sum rules, which needs to update the fixed parameters.
        """
        self.update_params(par_array)
        self.set_sumrules()
        self.set_moms()
        self.storage={} 
 
    def get_moments(self,flav,N=None,b_shift=0):
        """
        if N==None: then parametrization is to be use to compute moments along mellin contour
        else the Nth moment is returned
        """
        if N==None: N=self.mellin.N
        M,a,b,c,d=self.params[flav]
        b += b_shift
        mom=beta(N+a,b+1)+c*beta(N+a+0.5,b+1)+d*beta(N+a+1.0,b+1)
        norm=beta(2+a,b+1)+c*beta(2+a+0.5,b+1)+d*beta(2+a+1.0,b+1) #--each flavor is normalized to its second moment
  
        result = M*mom/norm
  
        return result 
  
    def _get_BC(self,g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_):
        """
        This function computes the boundary conditions given the combination of quark and gluon moments.
        """
        N=self.mellin.N
        
        # flav composition
        vm,vp={},{}
        vm[35]= bm + cm + dm + sm - 5*tm + um
        vm[24]= -4*bm + cm + dm + sm + um 
        vm[15]= -3*cm + dm + sm + um
        vm[8] = dm - 2*sp + 2*(-sm + sp) + um
        vm[3] = -dm + um
        vm[0] = np.zeros(N.size,dtype=complex)
        vp[0] = np.zeros(N.size,dtype=complex)
        vp[3] = -dp + up
        vp[8] = dp - 2*sp + up 
        vp[15]= -3*cp + dp + sp + up 
        vp[24]= -4*bp + cp + dp + sp + up 
        vp[35]= bp + cp + dp + sp - 5*tp + up 
        qs    = bp + cp + dp + sp + tp + up 
        qv    = bm + cm + dm + sm + tm + um 
        q     = np.zeros((2,N.size),dtype=complex)
        q[0]=np.copy(qs)
        q[1]=np.copy(g)
  
        BC={}
        BC['vm']=vm 
        BC['vp']=vp 
        BC['qv']=qv
        BC['q'] =q
        BC['um_']=um_
        BC['dm_']=dm_
        return BC
    
    def get_BC(self,moms):
        """
        Computes the boundary conditions for the number of flavor threshold.
        inputs:
            moms = dictionary with keys of flavor and flavor combinations whose elements are arrays of Mellin moments over the Mellin contour N.
        """
  
        N=self.mellin.N
        zero=np.zeros(N.size,dtype=complex)
  
        ###############################################
        # BC for Nf=3
        g   = moms['g']
        up  = moms['up']
        um  = moms['um']
        dp  = moms['dp']
        dm  = moms['dm']
        sp  = moms['sp']
        sm  = moms['sm']
        cp  = zero
        cm  = zero
        bp  = zero
        bm  = zero
        self.BC3=self._get_BC(g,up,um,dp,dm,sp,sm,zero,zero,zero,zero,zero,zero,zero,zero)
  
        ###############################################
        # BC for Nf=4
        BC4=self.dglap.evolve(self.BC3,self.Q20,self.mc2,3)
        g =BC4['g']
        up=BC4['up']
        dp=BC4['dp']
        sp=BC4['sp']
        cp=moms['cp']
        bp=zero
        um=BC4['um']
        dm=BC4['dm']
        sm=BC4['sm']
        cm=moms['cm']
        bm=zero
        self.BC4=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,zero,zero,zero,zero)
  
        ###############################################
        # BC for Nf=5
        BC5=self.dglap.evolve(self.BC4,self.mc2,self.mb2,4)
        g =BC5['g']
        up=BC5['up']
        dp=BC5['dp']
        sp=BC5['sp']
        cp=BC5['cp']
        bp=moms['bp']
        um=BC5['um']
        dm=BC5['dm']
        sm=BC5['sm']
        cm=BC5['cm']
        bm=moms['bm']
        self.BC5=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,zero,zero,zero,zero)
  
    def evolve(self,Q2):
        """
        Evolve the moments of the QCFs up to the input Q2, for the appropriate number of flavors, Nf.
        """
        # Q2 = float(Q2) #--avoid errors in Python 3
        # self.storage={} #--this is to avoid huge memory allocation to the storages of these dictionaries.
        # if Q2 not in self.storage:
        #     if self.mb2<Q2: 
        #         self.storage[Q2]=self.dglap.evolve(self.BC5,self.mb2,Q2,5)
        #     elif self.mc2<=Q2 and Q2<=self.mb2: 
        #         self.storage[Q2]=self.dglap.evolve(self.BC4,self.mc2,Q2,4)
        #     elif Q2<self.mc2: 
        #         self.storage[Q2]=self.dglap.evolve(self.BC3,self.Q20,Q2,3)
        Q2 = float(Q2) #--avoid errors in Python 3
        self.storage={} #--this is to avoid huge memory allocation to the storages of these dictionaries.
        if Q2 not in self.storage:
            # if self.mb2<Q2: 
            #     self.storage[Q2]=self.dglap.evolve(self.BC5,self.mb2,Q2,5)
            # elif self.mc2<=Q2 and Q2<=self.mb2: 
            #     self.storage[Q2]=self.dglap.evolve(self.BC4,self.mc2,Q2,4)
            # elif Q2<self.mc2: 
            #     self.storage[Q2]=self.dglap.evolve(self.BC3,self.Q20,Q2,3)
            self.storage[Q2] = self.dglap.evolve(self.BC4,self.mc2,Q2,4)
    
    def get_xF(self,x,Q2,flav,evolve=True):
        """
        Calculates the x-space 1d QCF through a Mellin inversion of the moments.
        This function is used for plotting QCFs and sometimes for codes of QCD observables.
        Inputs:
            x  = momentum fraction; type = float
            Q2 = scale choice; type = float
            flav = flavor; type = str
        """
        if evolve: self.evolve(Q2)
        return x*self.mellin.invert(x,self.storage[Q2][flav])
  
    def get_xF0(self,x,flav,evolve=True):
        """
        Calculates the initial-scale x-space 1d QCF through a Mellin inversion of the moments.
        Inputs:
            x  = momentum fraction; type = float
            flav = flavor; type = str
        """
        if   flav=='g' : mom=self.moms0['g']
        elif flav=='u' : mom=(self.moms0['up']+self.moms0['um'])/2
        elif flav=='d' : mom=(self.moms0['dp']+self.moms0['dm'])/2
        elif flav=='s' : mom=(self.moms0['sp']+self.moms0['sm'])/2
        elif flav=='c' : mom=self.mellin.N*0
        elif flav=='b' : mom=self.mellin.N*0
        elif flav=='ub': mom=(self.moms0['up']-self.moms0['um'])/2
        elif flav=='db': mom=(self.moms0['dp']-self.moms0['dm'])/2
        elif flav=='sb': mom=(self.moms0['sp']-self.moms0['sm'])/2
        elif flav=='cb': mom=self.mellin.N*0
        elif flav=='bb': mom=self.mellin.N*0
        return x*self.mellin.invert(x,mom)


if __name__=='__main__':

    from scipy.integrate import quad 
    mellin=MELLIN(npts=8)
    alphaS=ALPHAS()
    pdf=PDF(mellin,alphaS)

    xmin=0; xmax=1
    for Q2 in [par.mc2+1e-2,10,100]:
        integrand=lambda x: (pdf.get_xF(x,Q2,'u')-pdf.get_xF(x,Q2,'ub'))/x
        mom=quad(integrand,xmin,xmax)[0]
        print('Q2=%4.0f uv(1)=%0.2f'%(Q2,mom))

    for Q2 in [par.mc2+1e-2,10,100]:
        integrand=lambda x: (pdf.get_xF(x,Q2,'d')-pdf.get_xF(x,Q2,'db'))/x
        mom=quad(integrand,xmin,xmax)[0]
        print('Q2=%4.0f dv(1)=%0.2f'%(Q2,mom))

    for Q2 in [par.mc2+1e-2,10,100]:
        integrand=lambda x: (pdf.get_xF(x,Q2,'s')-pdf.get_xF(x,Q2,'sb'))/x
        mom=quad(integrand,xmin,xmax)[0]
        print('Q2=%4.0f sv(1)=%0.2f'%(Q2,mom))






