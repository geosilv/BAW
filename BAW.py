
import numpy as np
import math
from scipy.optimize import fsolve
from scipy.special import erf
import matplotlib.pyplot as plt



def bs_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = 0.5 * (1 + erf(d1 / np.sqrt(2)))
    N_d2 = 0.5 * (1 + erf(d2 / np.sqrt(2)))
    df = np.exp(-r*T)
    F=S*np.exp((r-q)*T)
    return df*(F*N_d1 - K*N_d2)

def bs_put(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_md1 = 0.5 * (1 + erf(-d1 / np.sqrt(2)))
    N_md2 = 0.5 * (1 + erf(-d2 / np.sqrt(2)))
    df = np.exp(-r*T)
    F=S*np.exp((r-q)*T)
    return df *(K*N_md2 - F*N_md1)   

def Nmd1(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return 0.5 * (1 + erf(-d1 / np.sqrt(2)))

def Nd1(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return 0.5 * (1 + erf(d1 / np.sqrt(2)))


def A2(S_star,K, T, r, q, sigma, q2) :
    return (S_star/q2)*( 1- np.exp(-q*T)*Nd1(S_star,K, T, r, q, sigma))

def A1(S_star,K, T, r, q, sigma, q1) :
    return -(S_star/q1)*(1- np.exp(-q*T)*Nmd1(S_star,K, T, r, q, sigma))   


#note that b the carry rate is r-d
def find_S_star(S_star, args):
    S, K, T, r, q, sigma ,phi=  args
    alpha=2*r/sigma**2  #M
    beta=2*(r-q)/sigma**2 #N
    K1=1-np.exp(-r*T)
    q2=(-(beta-1)+np.sqrt((beta-1)**2+(4*alpha/K1) ))/2
    q1=(-(beta-1)-np.sqrt((beta-1)**2+(4*alpha/K1) ))/2
    
    if phi==1:
        return bs_call(S_star, K, T, r, q, sigma)+A2(S_star,K, T, r, q, sigma,q2)  -(S_star-K)
    elif phi==-1:
        return bs_put(S_star, K, T, r, q, sigma)+A1(S_star,K, T, r, q, sigma,q1)  -(K-S_star)


def S_star_solution(S, K, T, r, q, sigma, phi):
    
    return fsolve(find_S_star, K, args=[S, K, T, r, q, sigma, phi])




def baw(S, S_star, K, T, r, q, sigma, phi):
    alpha=2*r/sigma**2
    beta=2*(r-q)/sigma**2
    K1=1-np.exp(-r*T)
    q1=(-(beta-1)-np.sqrt((beta-1)**2+(4*alpha/K1) ))/2
    q2=(-(beta-1)+np.sqrt((beta-1)**2+(4*alpha/K1) ))/2
    A1=-(S_star/q1)*(1- np.exp(-q*T)*Nmd1(S_star,K, T, r, q, sigma))   
    A2= (S_star/q2)*( 1- np.exp(-q*T)*Nd1(S_star,K, T, r, q, sigma))

    if phi==1:
        if (q<r):
             price = bs_call(S, K, T, r, q, sigma)
        
        elif S>=S_star:
             price = S-K
        else:
             price = bs_call(S, K, T, r, q, sigma)+A2*(S/S_star)**q2    
    elif phi==-1:
        if (q<r):
             price = bs_put(S, K, T, r, q, sigma)
        
        elif S<=S_star:
             price = K-S
        else:
             price = bs_put(S, K, T, r, q, sigma)+A1*(S/S_star)**q1           



    return price



def plot_boundary(S, K, r, q, sigma, phi):
    # Generate an array of time-to-maturities from a small positive number to 1 year
    T_values = np.linspace(0.1, 5, 100)
   #     K_values = np.linspace(40,1,150)
    S_star_values = []
    S_values =[]
    
    for T in T_values:
        S_star_solution1 = S_star_solution(S, K, T, r, q, sigma, phi)
        S_star = S_star_solution1[0]
        S_star_values.append(S_star)
     
    plt.figure()    
    plt.plot(T_values, S_star_values)

    plt.xlabel('Time to Maturity (T)')
    plt.ylabel('Early Exercise Boundary (S_star)')
    plt.title('Early Exercise Boundary vs Time to Maturity')
   
    plt.grid(True)
   
    
    return
    
    
def plot_bsvsbaw(S, K, r, q, sigma, phi):
    # Generate an array of time-to-mturities from a small positive number to 1 year
    T_values = np.linspace(0.1, 5, 300)
   #     K_values = np.linspace(40,1,150)
    bs_values = []
    baw_values =[]  
    for T in T_values:
        bsprice = bs_call(S, K, T, r, q, sigma)
   #     print("S, K, T, r, q, sigma = ", S, K, "{:.2f}".format(T), r, q, sigma)
        bs_values.append(bsprice)
        S_star = fsolve(find_S_star, K, args=[S, K, T, r, q, sigma])
        S_star = S_star[0]
        bawprice = baw(S, S_star, K, T, r, q, sigma)
        #bawprice=bawprice[0]
        baw_values.append(bawprice)
   #     print(type(bawprice) )
   #     print("T=", T, " bsprice= ", bsprice)
    plt.figure()   
    plt.plot(T_values, bs_values, label='BS prices', color = 'blue')
    plt.plot(T_values, baw_values, label='BAW prices', color='red' )

    plt.xlabel('Time to Maturity (T)')
    plt.ylabel('Option prices')
    plt.title('BS vs BAW prices')
    plt.legend()
    plt.grid(True)
    
    
    return


def plot_BSBAWS(S, K, r, q, sigma, phi):
    # Generate an array of time-to-mturities from a small positive number to 1 year
    S_values = np.linspace(10, 300, 300)
    bs_values = []
    baw_values =[]  
    if phi==1: 
        for S in S_values:
            bscall = bs_call(S, K, T, r, q, sigma)
            bs_values.append(bscall)
            S_star = fsolve(find_S_star, K, args=[S, K, T, r, q, sigma, 1])
            S_star = S_star[0]
            bawcall = baw(S, S_star, K, T, r, q, sigma, 1)
            baw_values.append(bawcall)

        plt.figure()  
        plt.title('BS vs BAW CALL prices')
        
    elif phi==-1:
        for S in S_values:
            bsput = bs_put(S, K, T, r, q, sigma)
            bs_values.append(bsput)
            S_star = fsolve(find_S_star, K, args=[S, K, T, r, q, sigma, -1])
            S_star = S_star[0]
            bawput = baw(S, S_star, K, T, r, q, sigma, -1)
            baw_values.append(bawput)          
        plt.figure()  
        plt.title('BS vs BAW PUT prices')



    plt.plot(S_values, bs_values, label='BS prices', color = 'blue')
    plt.plot(S_values, baw_values, label='BAW prices', color='red' )

    plt.xlabel('Time to Maturity (T)')
    plt.ylabel('Option prices')
  
    plt.legend()
    plt.grid(True)
    
    
    return



# Example usage:
S = 50  # Current stock price
K = 100  # Strike price
T = 0.25  # Time to maturity in years
r = 0.12 # Risk-free rate
q = 0.16 # Dividend yield
sigma = 0.2  # Volatility

#S_star_solution = fsolve(find_S_star2, K/10, args=[S, K, T, r, q, sigma])
#S_star = S_star_solution[0]



bscall = bs_call(S, K, T, r, q, sigma)
S_star = S_star_solution(S, K, T, r, q, sigma,1)
bawcall = baw(S, S_star, K, T, r, q, sigma, 1) 
print(f"S, K, T, r, q, sigma = {S:.1f}, {K:.1f}, {T: .2f}, {r:.2f}, {q: .2f}, {sigma:.2f}")
print(f"The BS  price for an European call option is: {bscall:.4f}")
print(f"The BAW price for an American call option is:", bawcall, "S*=", S_star)

bsput = bs_put(S, K, T, r, q, sigma)
S_star = S_star_solution(S, K, T, r, q, sigma,-1)
bawput = baw(S, S_star, K, T, r, q, sigma, -1) 
print(f"The BS  price for an European put option is: {bsput:.4f}")
print(f"The BAW price for an American put option is:", bawput, "S*=", S_star)

#plot_boundary(S, K, r, q, sigma)

#print("S_star=", S_star)
#plot_boundary(S, K, r, q, sigma, 1)
#plot_boundary(S, K, r, q, sigma, -1)

plot_BSBAWS(S, K, r, q, sigma, -1)
plot_BSBAWS(S, K, r, q, sigma, 1)
plt.show()