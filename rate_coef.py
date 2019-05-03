import numpy as N
from math import pi
import re

M_EL =  9.109534e-31
K_B = 1.380662e-23
Q_EL = -1.602189e-19
Q0 = -Q_EL
AMU = 1.67e-27
#Te = 20000
h = 6.62606957e-34  # m^2 kg s^-1
c = 299792458.      # m s^-1
cm2J = 100*h*c
omega = 4401

def rate_coef(f, CS_E, Te, maxwell):
    CS_E = N.array(CS_E)

    ## resample to finer grid
    from scipy.interpolate import interp1d
    if (maxwell == False):
        if (CS_E[0,0] != 0): 
            #print(CS_E[0])
            CS_E = N.insert(CS_E,0,0, axis =0)            
        points = f[:,0]      
        points = points.T.reshape((-1,1))
        data = interp1d(CS_E[:,0], CS_E[:,1])(points)
        CS_E = N.hstack((points, data))
    else:
        left = CS_E[:-1,0]
        right = CS_E[1:,0]
        points = left
        if (len(CS_E) < 100): 
            N_sub = 100
            for i in range(N_sub-1):
                xnew = left + (right-left)*(i+1.)/N_sub
                points = N.vstack((points, xnew))
            points = points.T.reshape((-1,1))
            data = interp1d(CS_E[:,0], CS_E[:,1])(points)
            CS_E = N.hstack((points, data))

    # obtain sigma as function of velocity
    if (maxwell == False):        
        EeV = f[:,0]        # eV = 1e-19 kg. m^2. s^-2
        fe = f[:,1]
    else:
        EeV = CS_E[:,0]        # eV = 1e-19 kg. m^2. s^-2
        fe = mxw_E(CS_E[:,0], Te*K_B/Q0)
    CS = CS_E[:,1]     # 1e-20 m^2 = 1e-16 cm^2
    M_EL = 9.109e-31        # kg
    # convert energy to velocity and calculate f(v)*sigma(v)*v
    v = N.sqrt(2*EeV*Q0/M_EL)    # m . s-1
    integrand = fe * CS * v

    # approximate the integral by trapezoid rule
    left = N.arange(0, len(EeV)-1)
    right = left+1
    dE = EeV[right] - EeV[left]
    integral = 0.5*sum(dE*(integrand[left]+integrand[right]))
    #rate = integral * 1e-14   # Phelps  1e-16 cm^2
    rate = integral * 1e2  #Fusion cm^2
    return rate

def CS_balance(rovnice,r_rate, T):
    mez = rovnice.index("=>")
    pozn_i = rovnice.index("//")
    pozn = rovnice[pozn_i:]

    if ("v=" in rovnice):
        #mez = rovnice.index("=>")
        #pozn_i = rovnice.index("//")
        #pozn = rovnice[pozn_i:]
        if ("ENERGY LOSS = -" in pozn):
            ro = pozn.index("= ")
            pozn = pozn.replace(pozn[ro+2], " ")
            
        if ("v=" in rovnice[:mez]):
            v0 = int(rovnice[rovnice.index("v=")+2])
        else:   v0 = 0
        new_rovnice = rovnice[mez+2:pozn_i] + "=>" + rovnice[:mez] + "// dopocteno, " + pozn[2:]
        if ("v=" in rovnice[mez:]):
            v1 = int(new_rovnice[new_rovnice.index("v=")+2])
        else:   v1 = 0
        try:
            rate = r_rate * N.exp(((v1-v0)*omega*cm2J/Q0)/T)
            if (N.isnan(rate) == True): rate = 0
        except: rate = 0

    if("dE" in rovnice):
        new_rovnice = rovnice[mez+2:pozn_i] + "=>" + rovnice[:mez] + "// dopocteno, " + pozn[2:]
        d_E = re.search("dE=\s+(\S+)", rovnice).group(1)
        d_E = float(d_E)
        d_g = re.search("dg=\s+(\S+)", rovnice).group(1)
        d_g = float(d_g)
        #rate = r_rate * N.exp(d_E /(K_B * T * Q0))
        rate = r_rate * d_g * N.exp(d_E /(T)) # d_E in eV, T in eV
        #print(d_E, "\t", T)
        
    return new_rovnice, rate

    
        

def mxw_E(E,T,M=M_EL):   
    E_max = T
    return 2*pi*(N.sqrt(1/pi/E_max)**3)*N.sqrt(E)*N.exp(-E/E_max)

def energy_distribution(f_distr, Te, maxwell=False):
    file_distr = open(f_distr)
    f = []
    for line in file_distr:
        m1, m2 = line.split()
        if(maxwell == False):
            f.append([float(m1), float(m2)])
        else:
            f.append([float(m1), mxw_E(float(m1), Te*K_B/Q0)])
    f = N.array(f)
    file_distr.close()
    return f

"""
def reaction_equation(reaction, short = False):
    inverse = False
    if(short):
        s
    else:
"""
    

def load_CS(f_reaction):
    CS_E = []
    for line in f_reaction:
        if(len(line)<2):
            continue
        if("end" in line):
            return CS_E
        try:    
            m1, m2 = line.split()
            if ((float(m1) > 25) and CS_E[-1][0] >= 25): continue
            CS_E.append([float(m1), float(m2)])
        except:            
            print("Bad data - Energy/cross section")



def read_file(f_reaction, f_distribution, Te, maxwell = False):
    file_prurez = open(f_reaction, "r")
    file_reakce = open("data/collisions/reaction.txt", "w")

    f = energy_distribution(f_distribution, Te, maxwell) 
    for line in file_prurez:
        #CS_E = []
        if(len(line) < 2): continue
        if("begin_c" in line):
            rovnice = line[7:]
            CS_E = load_CS(file_prurez)
            r_rate = rate_coef(f, CS_E, Te, maxwell)                                 
        if("begin_s" in line):
            rovnice = line[7:]
            a = next(file_prurez)
            a = a.replace("\t","")
            a = a.replace("\n","")
            #print(a)
            r_rate = a
            #try:
            #    r_rate = float(a)
            #except:
            #    r_rate = "difuze"

        file_reakce.write(str(r_rate) + rovnice)
        
        if ("inverse=True" in rovnice):
                rovnice, rate = CS_balance(rovnice, r_rate, Te*K_B/Q0)
                file_reakce.write(str(rate) + rovnice)
        
    file_prurez.close()
    file_reakce.close()
    #del CS_E, rate, rovnice
