import numpy as np
import math

from integrate import integrate_function

def volume():
	r = 1

	#geometric defining function for a sphere of radius 1
	def f(x,y,z):
		return math.sqrt(x*x + y*y + z*z) - r

	#limits of integration
	limit  = [-1, 0] #[-np.inf, 0] if min is not known
	#domain size to integrate over
	size   = np.array([2, 2, 2], dtype = np.float64)
	#2^depth levelsets used in 1D integral
	depth  = 10
	#grid resolution
	res    = 32
	#stencil genus. With float64 data maximum allowed is 2 (due to shared memory limitations)
	genus  = 2
	#quadrature scheme to use
	quad   = 'simpsons'
	#use adaptive or fixed quardature scheme (fixed is faster for small grids)
	scheme = 'fixed'
	#true volume of a unit sphere
	true = 4/3*np.pi*(r**3)

	integral = integrate_function(f = f,
								  limit = limit,
								  size = size,
								  depth = depth,
								  res = res,
								  genus = genus,
								  quad = quad,
								  scheme = scheme)

	print("Computed: %s"%integral)
	print("True: %s"%true)

def trivariate():
	from integrate import compile_f, compile_g, compile_levelsets, compile_grad

	r = 1

	def f(x,y,z):
		return math.sqrt(x*x + y*y + z*z) - r
		# return x*x + y*y + z*z - r

	#limits of integration
	limit  = [-1, 0]
	#domain size to integrate over
	size   = np.array([2*r, 2*r, 2*r], dtype = np.float64)
	#2^depth levelsets used in 1D integral
	depth  = 10
	#grid resolution
	res    = 32
	#stencil genus. With float64 data maximum allowed is 2 (due to shared memory limitations)
	genus  = 2
	#quadrature scheme to use
	quad   = 'simpsons'
	#use adaptive or fixed quardature scheme (fixed is faster for small grids)
	scheme = 'fixed'
	#true volume of a unit sphere
	true = 4/3*np.pi*(r**3)

	#compile kernels with given inputs
	f    = compile_f(f)
	grad = compile_grad(genus)

	#precomile all integrand kernels to save on evaluation time
	compiled = []
	for i in range(4):
		compiled.append([])
		for j in range(4):
			compiled[-1].append([])
			for k in range(4):
				if i + j + k <= 3:
					def g(x,y,z):
						return x**i + y**j + z**k

					g        = compile_g(g)
					levelsets = compile_levelsets(genus, quad, g, scheme)

					compiled[-1][-1].append((g, levelsets))

	import sympy as sp
	r, t, p = sp.symbols('r t p')

	x = r*sp.sin(p)*sp.cos(t)
	y = r*sp.sin(p)*sp.sin(t)
	z = r*sp.cos(p)

	for i in range(4):
		for j in range(4):
			for k in range(4):
				if i + j + k <= 3:
					true = sp.integrate((x**i + y**j + z**k)*r**2*sp.sin(p),
										(r, 0, 1),
										(t, 0, 2*np.pi),
										(p, 0, np.pi))

					g, levelset = compiled[i][j][k]
					integral = integrate_function(f = f,
												  g = g,
												  grad = grad,
												  levelset = levelset,
												  limit = limit,
												  size = size,
												  depth = depth,
												  res = res,
												  genus = genus,
												  quad = quad,
												  scheme = scheme)
					print("Computed: %s"%integral)
					print("True: %s"%true)
					print()

def saye():
	def f(x,y,z):
		return math.cos(x)*math.sin(y) + math.cos(y)*math.sin(z) + math.cos(z)*math.sin(x)

	def g(x,y,z):
		#integrand is unbounded, prevent integrand contributions outside domain
		boole = (abs(x) <= L)*(abs(y) <= L)*(abs(z) <= L/2.)
		return boole*math.log((x**2 + y**2 + z**2)/(L**2) + 3/8)

	L = 4.25

	#limits of integration
	limit  = [-np.inf, 0] #[-1.5, 0]
	#domain size to integrate over
	size = np.array([2*L, 2*L, L], dtype = np.float64)
	#2^depth levelsets used in 1D integral
	depth  = 10
	#grid resolution
	res    = 32
	#stencil genus. With float64 data maximum allowed is 2 (due to shared memory limitations)
	genus  = 2
	#quadrature scheme to use
	quad   = 'simpsons'
	#use adaptive or fixed quardature scheme (fixed is faster for small grids)
	scheme = 'fixed'
	#true volume computed in saye's paper
	true = 6.26192376

	integral = integrate_function(f = f,
								  g = g,
								  limit = limit,
								  size = size,
								  depth = depth,
								  res = res,
								  genus = genus,
								  quad = quad,
								  scheme = scheme)

	print("Computed: %s"%integral)
	print("True: %s"%true)
	print()

# volume()
# trivariate()
saye()
