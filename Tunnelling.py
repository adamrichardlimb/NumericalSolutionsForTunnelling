#-- Begin defining modules to import.

import numpy as npy
import matplotlib.pyplot as plt
import sympy

#-- Cease defining modules to import.



#-- Begin defining exact barrier.

def Exact(x, V0, L):
    return V0 / ( (npy.cosh( (x / L) ))**2.0 )

#-- Cease defining exact barrier.



#-- Begin defining exact transmission coefficient.
#-- Thank you to Kendra for giving me her function
#-- in order to prevent a runtime error.

def Exact_Transmission(E, V0, L):
    
    k0 = npy.sqrt(E)
    
    term_1 = npy.sinh(npy.pi*k0*L)**(2)  

    Kappa = npy.sqrt(V0)

    if (Kappa**2.0) * (L**2.0) < (0.25):
        term_2 = ( npy.cos(npy.pi*(npy.sqrt(0.25-(Kappa**(2)*L**(2)) ))) )**2  
        Transmission = (term_1) / (term_1 + term_2)
        
    else: 
        term_3 =sympy.cosh(npy.pi * (npy.sqrt( (Kappa**(2) *L**(2) ) - 0.25)))**2
        Transmission = (term_1) / (term_1 + term_3)
    
    return Transmission

#-- Cease defining exact transmission coefficient.



#-- Begin defining comparison potential.

def Model(x, V0, L):
    return V0 / ( 1 + (x**2.0 / ( 2 * (L**2.0) ) ) )**2.0

#-- Cease defining comparison potential.



#-- Begin defining potential plotter.

def Potential_Plotter(rangex, E, V0, L):
    
    rangex = npy.arange(- rangex, rangex, 0.1)
    position = []
    exact_potential = []
    model_potential = []
    
    for pos in rangex:
        
        exact_potential.append( Exact( pos, V0, L ) )
        model_potential.append( Model( pos, V0, L ) )
        position.append( pos )
        
        
    plt.plot(position, exact_potential, label = 'Exact Potential')
    
    plt.plot(position, model_potential, label = 'Model Potential')
    
    plt.xlabel("x")
    
    plt.ylabel("Potential")
    
    plt.legend()
    
#-- Cease defining potential plotter.



#-- Begin defining derivative function.

def Derivatives(n, x, y, L, V0, E):
    
    dy = npy.zeros(n + 1, float)
    
    Wavenumber = npy.sqrt( E )
    
    z = Wavenumber * x
    
    Lambda = Wavenumber * L
    
    dy[1] = y[2]
    
    dy[2] = ((V0/E) / (1+(z**2)/(2*Lambda**2))**(2.0) - 1) * y[1]
    
    return dy

#-- Cease defining derivative function.
    


#-- Begin defining RK4 function.

def Rk4(n, x, y, h, L, V0, E):  #Runge-Kutta order 4 algorithm 
    
    y0 = y[:]
    
    k1=Derivatives(n, x, y, L, V0, E)
    
    for i in range(1,n+1):
        
        y[i]=y0[i]+0.5*h*k1[i]
        
        k2=Derivatives(n, x+0.5*h, y, L, V0, E)
        
    for i in range(1,n+1):
        
        y[i]=y0[i]+h*(0.2071067811*k1[i]+0.2928932188*k2[i])
        
        k3=Derivatives(n, x+0.5*h, y, L, V0, E)
        
    for i in range(1,n+1):
        
        y[i]=y0[i]-h*(0.7071067811*k2[i]-1.7071067811*k3[i])
        
        k4=Derivatives(n, x+h, y, L, V0, E)
        
    for i in range(1,n+1):
        
        a=k1[i]+0.5857864376*k2[i]+3.4142135623*k3[i]+k4[i]
        
        y[i]=y0[i]+0.16666666667*h*a
         
    x+=h
    
    return(x, y)

#-- Cease defining RK4 function.



#-- Begin defining tunnelling function.

def Tunnelling(E, V0, L, rangex):
    
    xmax = rangex
    xmin = - rangex
    N = (xmax*10)
    x = xmax
    k0 = npy.sqrt(E)
    z = k0 * x
    dx = (xmax - xmin) / N
    
    T = 1

    Psi1 = [0.0 for i in range(0, N+1)]  #wavefunction values
    Psi2 = [0.0 for i in range(0, N+1)] 
    Psi3 = [0.0 for i in range(0, N+1)]

    y = [0, T*npy.cos(z), -T*k0*npy.sin(z) ] #real part of wavefunction
    
    trans = [0.0 for i in range(0, N+1)]
    
    etrans = Exact_Transmission(E, V0, L)
    
    exact_trans = [etrans for i in range(xmin, xmax, 1)]
    
    position = [i for i in range(xmin, xmax, 1)]
    
    for j in range(0,N+1):
    
        Psi1[N-j] = y[1] 
        
        (x,y) = Rk4(2, x, y, -dx, L, V0, E)
        
    y_re = y
    y = [0, T*npy.sin(z), T*k0*npy.cos(z)]
    x = xmax
    
    for j in range(0, N+1):
    
        Psi2[N-j]=y[1] 
        
        (x,y) = Rk4(2, x, y, -dx, L, V0, E)
        
    y_im = y
        
    for j in range(0, N+1):
        
        Double_I = 0.25*((y_re[1]+(y_im[2]/k0))**(2) + (y_im[1]- (y_re[2]/k0))**(2))
        
        trans[N - j] = (T**2.0) / Double_I

    for i in range(0,N):
    
        Psi3[i] = Psi1[i] + Psi2[i]

    #for j in range(0, N):
        #plt.plot([xmin+j*dx, xmin+j*dx+dx], [trans[j], trans[j+1]], color = 'red')
        #plt.plot([xmin+j*dx, xmin+j*dx+dx], [Psi3[j], Psi3[j+1]], color = 'green')
        
    #plt.plot(position, exact_trans, color = 'purple')
    
    print(trans[1])
    print(exact_trans[1])
    
    Potential_Plotter(rangex, E, V0, L)
    #print(Exact_Transmission(E, V0, L))
    #print(trans[2])
    
    #print(trans)
    #print(exact_trans)

Tunnelling(9, 10, 8, 50)}
