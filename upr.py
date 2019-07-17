import numpy as np 
import scipy.sparse as sp
from scipy.integrate import ode
import re
#import rate_coef as RC
import matplotlib.pyplot as plt

class Prvky:
    def __init__(self, name = 0, conc = 0, mass = 0, radii = 0, energy = None, degeneration = None):
        """ Prvky object contains species of the model. 
        name - name of the particle 
        conc - concentration of the species
        mass - means atomic mass; except an electron
        radii - atomic radius
        energy - energy of the excited state
        degeneration - degeneration of the excited state
        """
        self.name = name
        self.conc = conc
        self.mass = mass
        self.radii = radii
        self.energy = energy
        self.degeneration = degeneration

def find_index(species, name):
    """ Find index in the array of species by its name. """
    for i in range(len(species)):
        if(species[i].name == name):
            return i
    raise ValueError("Wrong name")

class Reaction:
    #def __init__(self, electron = False, k_c = 0, Eloss = 0, reakce):
    #def __init__(self, reaction = [[],[]], k_c = 0, Eloss = 0, calculating_k_rate = 0, cooling = None):
    def __init__(self, reaction = [[],[]], k_c = 0, calculating_k_rate = 0, cooling = None):
        """The Reaction object contains the chemical equation.
        
        reaction = [[reactants], [products]] - reactants and products in arrays are indexes representing species
        k_c - rate coefficient of the reaction        
        # Eloss - energy lost by the collision        
        calculating_k_rate - it tells how is rate coefficient calculated
            calculating_k_rate =  0 - countig from energy and cross section
                                  1 - Stevefelt formula
                                  2 - diffusion
                                  3 - ambipolar diffusion
                                  4 - constant value; any calculation
        cooling - type of reaction which influences electron temperature
            cooling = 0 - elastic collision
                      1 - inelastic collision
                      2 - super-elastic collision
                      3 - Coulomb collision
        """
        #self.electron = electron
        self.reaction = reaction
        self.k_c = k_c
        #self.Eloss = Eloss
        self.calculating_k_rate = calculating_k_rate 
        self.cooling = cooling

    def energy_transfer(self, species):
        """ Calculates energy lost/obtained by reaction. 
        (Meant for collision of electron with excited particle.) """
        #print(self.reaction)
        for i in self.reaction[0]:
            if(i != 0):
                E_reactant = species[i].energy
                break
        for i in self.reaction[1]:
            if(i != 0):
                E_product = species[i].energy
                break
        try:
            return E_product - E_reactant 
        except:
            print("Chyba vypoctu energy_transfer")

class ODE:
    def __init__(self, species, reactions, time, time_step, Te, cooling = False, solver = 'lsoda'):
        """ Class for create and solve ODE for the model."""
        self.solver = solver
        self.time = time
        self.time_step = time_step
        self.species = species
        self.reactions = reactions
        self.set_init_concentrations()  # load concentrations from species
        self.Te = Te
        self.cooling = True
        self.create_vector_rate_coefficients(reactions)
        matrix_reactants = self.create_matrix_from_reactions(species, reactions, 0)
        matrix_products  = self.create_matrix_from_reactions(species, reactions, 1)

        self.matrix_reactants = matrix_reactants 
        self.matrix_change = matrix_products - matrix_reactants
        self.matrix_change_transpose = np.transpose(self.matrix_change)

        self.cooling_type = self.cooling_type(reactions)
        self.calculating_rate_type = self.calculating_rate_type(reactions)
        #self.delta_E = self.init_delta_E(reactions, species)
        self.evolving = []
        self.time_evolnig = []

    def set_init_concentrations(self):
        concentration = []
        for i in range(len(self.species)):
            concentration.append(self.species[i].conc)
        self.concentration = np.array(concentration)
    
    def set_final_concentration(self, conc):
        for i in range(len(conc)):
            self.concentration[i] = conc[i]

    def create_vector_rate_coefficients(self, reactions):
        self.rate_coefficients = np.zeros(len(reactions))
        for i in range(len(reactions)):
            self.rate_coefficients[i] = self.reactions[i].k_c

    def update_rate_coef(self):
        # type: 0 - counting from CS; 1 - Stevefelt; 2 - diffusion; 3 - ambi. diffusion; 4 - const
        # cooling   - > for i in self.calculating_rate_type[0]:
        #print("Run update")
        for i in self.calculating_rate_type[1]:         
            self.rate_coefficients[i] = Stevefelt_formula(self.concentration[0], self.Te)
        for i in self.calculating_rate_type[2]:
            coll_rate = 1.26076522957e-12
            index_species = self.reactions[i].reaction[0][0]
            #concentration = self.concentration[index_species]
            concentration = 9.41e17
            atomic_mass   = self.species[index_species].mass
            self.rate_coefficients[i] = diffusion_rate(coll_rate, concentration, atomic_mass * AMU)
        for i in self.calculating_rate_type[3]:
            index_species = self.reactions[i].reaction[0][0]
            #concentration = self.concentration[index_species]
            concentration = 9.41e17
            atomic_mass   = self.species[index_species].mass
            self.rate_coefficients[i] = ambipolar_diffusion(concentration, atomic_mass * AMU, Te)

    def create_matrix_from_reactions(self, species, reactions, part):
        #self.matrix_reaction = np.zeros([len(reactions), len(species)])
        matrix_reaction = np.zeros([len(reactions), len(species)])
        for i in range(len(reactions)):
            for j in range(len(species)):
                matrix_reaction[i,j] = reactions[i].reaction[part].count(j)
        return sp.csr_matrix(matrix_reaction, dtype =  np.int8)

    def cooling_type(self, reactions):
       cooling_type = [[], [], [], []]
       for i in range(len(reactions)):
           cool = reactions[i].cooling
           if not(cool == None):
               cooling_type[cool].append(i)
       return cooling_type

    def calculating_rate_type(self, reactions):
        calc_rate_type = [[], [], [], [], []]
        for i in range(len(reactions)):
            calc_rate_type[reactions[i].calculating_k_rate].append(i)
        return calc_rate_type
    
    def init_delta_E(self, reactions, species):
       delta_E = np.zeros(len(reactions))
       for i in self.cooling_type[1]:
           delta_E[i] = reactions[i].energy_transfer(species) 
       for i in self.cooling_type[2]:
           delta_E[i] = reactions[i].energy_transfer(species)
       return delta_E 

    def create_ODE(self, time, concentration):
        concentration[concentration < 1e-12] = 0

        self.update_rate_coef()
        f = self.matrix_reactants.dot(np.log(concentration))
        f = np.exp(f) * self.rate_coefficients
        f = self.matrix_change_transpose * f
        return f

    def solve_ODE(self):
        t0 = 0
        y0 = self.concentration
        r = ode(self.create_ODE).set_integrator(self.solver)
        r.set_initial_value(y0, t0) # initial conditions in form (y0, t0)
        #r.set_f_params()    # self. * 
        stop = 0
        while r.successful() and r.t < self.time:
            try:
                r.integrate(r.t + self.time_step)
                self.time_evolnig.append(r.t)
                self.evolving.append(r.y)
                self.concentration = r.y
                successful = True
            except:
                print("Integration error in solve_ODE")
                successful = False
                break
        print(r.t)

        if(successful):
            print("Integration was succesfull.")

def collision_frequency(collision_rate, concentration):
    return collision_rate * concentration

# calculate diffusion coefficient
def diffusion_coef(collision_rate, concentration, mass):
    nu = collision_frequency(collision_rate, concentration)
    #if(nu):
    #    return k_b * Tn / (mass * nu)
    #else:
    #    return 0
    return k_b * Tn / (mass * nu)

# calculate rate coefficient of diffusion  - length means length of device
def diffusion_rate(collision_rate, concentration, mass):
    return diffusion_coef(collision_rate, concentration, mass) / length

# mass_diff means mass for calculation of diffusion defined below
def langevin_rate(mass):
    r_m = reduced_mass(mass, mass_diff)
    alpha = 0.228044e-40    # C2m2 J-1 polarizability
    #print(1e6*(elementary_charge / (2*epsilon_0)) *((alpha / r_m)**0.5)) # cm3 s-1
    return 1e6*(elementary_charge / (2*epsilon_0)) *((alpha / r_m)**0.5) # cm3 s-1

# calculate rate coefficient of ambipolar diffusion
def ambipolar_diffusion(concentration, mass, Te):
    Di = diffusion_coef(langevin_rate(mass), concentration, mass)
    #print(Di)
    return Di * (1 + Te/Tn) / length

def Stevefelt_formula(conc, Te):
    # concentration and temperature of electrons
    return (3.8e-9)*(Te**(-4.5))*conc + (1.55e-10) * (Te**(-0.63)) + (6e-9)\
            * (Te**(-2.18))*(conc**(0.37))

def reduced_mass(mass1, mass2):
    return mass1 * mass2 / (mass1 + mass2)

def load_species(file_species):
    species = []	
    with open(file_species, "r") as f_species:
        for line in f_species:
            if ("#" in line): continue
            try:
                data = line.split()
                if(len(data) == 4):
                    #            Prvky(name,        init_conc,        mass,           radii)
                    species.append(Prvky(data[0], float(data[1]), float(data[2]), float(data[3])))
                else:
                    #            Prvky(name,        init_conc         mass            radii           energy       degeneration)
                    species.append(Prvky(data[0], float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])))
            except: continue
    return species

def load_reaction(species, file_reaction):
    reactions = []
    with open(file_reaction, "r") as f_reactions:
        for line in f_reactions:
            reaction = line.split()
            divide = reaction.index("=>")
            note = reaction.index("//")
            
            reactants = []
            for i in range(len(reaction[1:divide])):
                reactants.append(find_index(species, reaction[1 + i]))
            products = []
            for i in range(len(reaction[divide + 1:note])):
                products.append(find_index(species, reaction[divide + 1 + i]))
            if(len(reaction[note + 1:]) == 2):
                #                                                       k_rate        calculating_k_rate        cooling
                reactions.append(Reaction([reactants, products], float(reaction[0]), int(reaction[-2]), int(reaction[-1])))
            else:
                #                                                       k_rate        calculating_k_rate
                reactions.append(Reaction([reactants, products], float(reaction[0]), int(reaction[-1])))

    return reactions 


# neutral elastic collision rate coeff
difu = 1.26076522957e-12

AMU = 1.667e-27
k_b = 1.38e-23
Tn = 77
mass_diff = 4.0026 * AMU
Q0 = 1.602189e-19
elementary_charge = Q0 
epsilon_0 = 8.854187817620e-12

# dimensions of the device
radius = 7.5e-3
length = (radius / 2.405) **2

soubor = "species.txt"
soubor2 = "reaction.txt"

Te = 22700
#time = 0.0017878 
#time = 0.000171
time = 20e-3
time_step = time/10e5

species = load_species(soubor)
reactions = load_reaction(species, soubor2)
concentration = ODE(species, reactions, time, time_step, Te, solver = 'lsoda')

concentration.solve_ODE()

for i in range(len(concentration.concentration)):
    print(concentration.species[i].name, ": \t %e" % concentration.concentration[i])








import matplotlib.pyplot as plt

plt.figure(figsize=(7.5,5))

concentration.evolving = np.array(concentration.evolving)
concentration.time_evolnig = np.array(concentration.time_evolnig)

for i in range(len(species)):
    plt.loglog(concentration.time_evolnig[:], concentration.evolving[:,i])
plt.xlabel(r"$t (\rm s))$")
plt.ylabel(r"$n (\rm cm^{-3})$")
plt.savefig("test.pdf")
plt.close()


# radiative
lambd = np.array([388.9, 587.5, 667.8, 706.5])
Aik = np.array([9.4746e+6, 7.0703e+7, 6.37e+7, 2.7853e+7])
#Aik = Aik ** 2
particles = ['He(1s3p3P)', 'He(1s3d3D)', 'He(1s3d1D)', 'He(1s3s3S)']
c_l = 299792458
h_p = 6.626e-34

coef = c_l * h_p / lambd * Aik * Aik
n = np.zeros(len(particles))
for i in range(len(particles)):
    n[i] = concentration.concentration[find_index(species, particles[i])]


#n = Aik * n
intenzity = coef * n
sum_intenz = sum(intenzity)
print(sum_intenz)

lambdd = np.array([387.9,388.9,389.9,586.5, 587.5,588.5,666.8, 667.8,668.8,705.5, 706.5,707.5])
proc = np.array([0,0.06,0,0,0.47,0,0,0.19,0,0,0.28,0])
plt.figure(figsize=(7.5, 5))
for i in range(len(lambd)):
    plt.bar(lambd[i], intenzity[i]/sum_intenz,  align="center",color="red", width=13, alpha=1)
    #plt.bar(deex[i][1], intenzity[i],  align="center",color="red", width=13, alpha=1)
plt.plot(lambdd, proc)
plt.savefig("intenz.pdf")
plt.close()
print(intenzity/sum_intenz)
