import numpy as np
from numba import cuda, float64
import math
import time
import matplotlib.pyplot as plt

from reductions import sum_gpu, min_gpu, max_gpu

#order: 2,4,6,8
central = [np.array(                      [-1/2, 0, 1/2],                       dtype = np.float64),
		   np.array(                [1/12, -2/3, 0, 2/3, -1/12],                dtype = np.float64),
		   np.array(       [-1/60 ,  3/20, -3/4, 0, 3/4, -3/20, 1/60],          dtype = np.float64),
		   np.array([1/280, -4/105,  1/5 , -4/5, 0, 4/5, -1/5 , 4/105, -1/280], dtype = np.float64)]

#maximum number of evaluation threads per block
ethreads = 512

#error threshold for levelset contribution
EPS = 1e-14

#levelset kernel block size
_threads = np.array([8,8,8], dtype = np.int64)

#maximum chunk shape (determined by available memory on GPU)
# chunkshape = np.array([512, 512, 512], dtype = np.int64)
chunkshape = np.array([256, 256, 256], dtype = np.int64)
# chunkshape = np.array([128, 128, 128], dtype = np.int64)
# chunkshape = np.array([64, 64, 64], dtype = np.int64)

@cuda.jit(device = True)
def __flatten3d(i, j, k, shape):
	return i + j*shape[0] + k*shape[0]*shape[1]

@cuda.jit(device = True)
def __unflatten3d(n, shape):
	i = n%shape[0]
	j = (n//shape[0])%shape[1]
	k = n//(shape[0]*shape[1])

	return (i,j,k)

@cuda.jit('b1(f8,f8)', device = True)
def __chi(f, levelset):
	return f <= levelset

def setup_levelset_kernel(shape):
	global _threads

	#initilize with center block
	blocks  = [shape//_threads]
	threads = [_threads]
	offsets = [np.zeros(3, dtype = np.int64)]

	#amount sticking out in each dimension
	remainder = shape - (shape//_threads)*_threads

	#edge and face blocks (any outer blocks not divisible by 8 in every dimension)
	for n,p in enumerate([[1,2],[0,2],[0,1]]):
		#planes
		pthreads = np.array(threads[0])
		pblocks  = np.ones(3, dtype = np.int64)
		poffset  = np.zeros(3, dtype = np.int64)

		pthreads[n] = remainder[n]
		pblocks[p]  = blocks[0][p]
		poffset[n]  = blocks[0][n]*8

		blocks.append(pblocks)
		threads.append(pthreads)
		offsets.append(poffset)

		#edges
		ethreads = np.array(threads[0])
		eblocks  = np.ones(3, dtype = np.int64)
		eoffset  = np.zeros(3, dtype = np.int64)

		ethreads[p] = remainder[p]
		eblocks[n]  = blocks[0][n]
		eoffset[p]  = blocks[0][p]*8

		blocks.append(eblocks)
		threads.append(ethreads)
		offsets.append(eoffset)

	#corner block (remainder block not divisible by 8 in any dim)
	blocks.append(np.ones(3, dtype = np.int64))
	threads.append(remainder)
	offsets.append(blocks[0]*8)

	return blocks, threads, offsets

def compute_levelset(kernels, kargs, blocks, threads, offsets, shape):
	#Kernel break down 
	#	Note: bottom plane hidden behind front facing plane/edge
	#        __________________
	#       /             /  / |
	#      /             /  /  |
	#     /   center    /  /  e|
	#    /             /  /  n |
	#   /____________ /__/  a  |
	#  /____________ /__/| l   |
	# |              |  ||p    |
	# |              |e ||     |
	# |              |d ||    /| 
	# |    plane     |g ||   /e/ 
	# |              |e ||  /g/  
	# |              |  || /d/
	# |              |  ||/e/
	# |______________|__|/ /
	# |     edge     |  ||/
	# |______________|__|/ <- corner

	streams = [cuda.stream()]

	if np.prod(threads[0]) and np.prod(blocks[0]):
		d_offset = cuda.to_device(offsets[0], stream = streams[-1])
		kernels[0][(*blocks[0], ), (*threads[0], ), streams[-1]](*kargs)

	for b, t, o in zip(blocks[1:], threads[1:], offsets[1:]):
		if np.prod(t) and np.prod(b):
			streams.append(cuda.stream())

			d_offset = cuda.to_device(o, stream = streams[-1])

			kernels[1][(*b, ), (*t, ), streams[-1]](*kargs, d_offset)

	#sync all streams to make sure all kernels finished before moving on
	for stream in streams:
		stream.synchronize()

	#only sum from d_integral the contributions that have been modofied from this block (prevents the array from needing to be zeroed)
	return sum_gpu(kargs[0], np.prod(shape))

def integrate_fixed(kernels, kargs, shape, res, depth, fmin, fmax):
	blocks, threads, offsets = setup_levelset_kernel(shape)

	#add fixed kernel specific args
	d_limit = cuda.to_device(np.array([fmin, fmax], dtype = np.float64))
	kargs.append(d_limit)
	kargs.append(depth)

	h = 1/(res**3)
	return compute_levelset(kernels, kargs, blocks, threads, offsets, shape)*h #*((fmax - fmin)/2)

def adaptive_simpsons(kernels, kargs, shape, res, max_depth, fmin, fmax, err):
	#notation from https://www.youtube.com/watch?v=gYVYOtmx-Ms

	# y
	# |          __
	# |      ___/  \__
	# |  ___/  |     |\
	# | /|     |     | \
	# |/ |     |     |
	# |  |     |     |
	# |  |     |     |
	# |__|_____|_____|____ x
	#    a     c     b
	#    |-----------| = h
	#          R
	#   subdivide R into R1 and R2
	# y
	# |          __
	# |      ___/ |\__
	# |  ___/  |  |  |\
	# | /|  |  |  |  | \
	# |/ |  |  |  |  |
	# |  |  |  |  |  |
	# |  |  |  |  |  |
	# |__|__|__|__|__|____ x
	#    a  d  c  e  b
	#    |-----|-----| = h
	#       R1    R2

	#compute kernel info only once for each levelset kernel
	blocks, threads, offsets = setup_levelset_kernel(shape)


	levels = []


	h = 1/(res**3)

	fmid = (fmin + fmax)/2.

	Ia = compute_levelset(kernels, [*kargs, fmin], blocks, threads, offsets, shape)*h
	Ic = compute_levelset(kernels, [*kargs, fmid], blocks, threads, offsets, shape)*h
	Ib = compute_levelset(kernels, [*kargs, fmax], blocks, threads, offsets, shape)*h

	#Note:
		#Once depth_flag has been tripped the quad will not subdivide by all remaining quad points at that max depth with still be accumulated

	quadrature = [[[fmin, fmid, fmax], [Ia, Ic, Ib], 0]]
	depth_flag = False
	integral   = 0
	while quadrature:
		quad, I, depth = quadrature.pop(0)

		a,  c,  b  = quad
		Ia, Ic, Ib = I

		levels.append(a)
		levels.append(c)
		levels.append(b)

		##########
		# print(depth + 1, a, b)
		##########

		I1 = (b - a)/6.*(Ia + 4.*Ic + Ib)

		#first run depth check before computing sub interval integral
		if depth >= max_depth - 1:
			#this is a breadth first algorithm so once the first interval gets to max depth all remaining intervals are also at max depth
			depth_flag = True
			#accumulate integral contribution
			integral += I1

		#if max depth has not been acheived sub divide range and compute sub integrals
		if not depth_flag:
			d = (a + c)/2.
			e = (c + b)/2.

			Id = compute_levelset(kernels, [*kargs, d], blocks, threads, offsets, shape)*h
			Ie = compute_levelset(kernels, [*kargs, e], blocks, threads, offsets, shape)*h

			I2 = (c - a)/6.*(Ia + 4.*Id + 2.*Ic + 4.*Ie + Ib)

			#the error threshold has been reached. Accumulate contribution and move to next interval
			if abs(I2 - I1) <= 15*err:
				#Richardson Extrapolation for Simpson's rule
				integral += (16*I2 - I1)/15

				#Richardson Extrapolation for Trap
				#(4*I2 - I1)/3

			#append sub intervals to quadrature tree to compute at lower resolution
			else:
				quadrature.append([[a, d, c], [Ia, Id, Ic], depth + 1])
				quadrature.append([[c, e, b], [Ic, Ie, Ib], depth + 1])

	# if depth_flag:
		# print('You reached maximum depth before reaching the provided error tolerance')

	return integral, levels

def integrate_adaptive(kernels, kargs, shape, res, depth, fmin, fmax, err, quad):
	if quad == 'simpsons':
		return adaptive_simpsons(kernels, kargs, shape, res, depth, fmin, fmax, err)
	# elif quad == 'trap':
	# 	return

def compile_levelsets_fixed(genus, quad, g):
	#compile kernel with genus as a constant and use the specified quad function
	global central

	constant = central[genus - 1]

	sh_shape  = _threads
	sh_rshape = sh_shape + 2*genus
	sh_size   = np.prod(sh_rshape)

	sh_l = sh_shape[0]
	sh_w = sh_shape[1]
	sh_h = sh_shape[2]

	sh_rl = sh_rshape[0]
	sh_rw = sh_rshape[1]
	sh_rh = sh_rshape[2]

	if quad == "simpsons":
		@cuda.jit('f8(i8,i8)', device = True)
		def __quad(element, total):
			#coeff: (1 4 2 .. 4 2 4 1)/3
			#outputs quadrature coeff based on locationn along span
				#first term alternates between 2 and 4
				#the second term subtracts 1 if on the start or end of the span

			half = total//2
			return ((2*(element%2) + 2) - max(abs(element - half) - half + 1, 0))/3.

	elif quad == 'trap':
		__quad = None

	if g is None:

		@cuda.jit('void(f8[:],f8[:],f8[:,:],i8[:],i8[:],f8[:],i8)')
		def __compute_levelsets_inner(integral, f, grad, rshape, shape, limit, depth):
			i,j,k = cuda.grid(3)

			#set shared mem size to maximum possible block size
			shared = cuda.shared.array(shape = (sh_size,), dtype = float64)
			#constant array filled with stencil values
			stencil = cuda.const.array_like(constant)

			#shared memory indicies
			sh_i = cuda.threadIdx.x + genus
			sh_j = cuda.threadIdx.y + genus
			sh_k = cuda.threadIdx.z + genus

			#flattened integral index
			n    = __flatten3d(i, j, k, shape)
			#flattened function index
			r_n  = __flatten3d(i + genus, j + genus, k + genus, rshape)
			#flattened shared memory index
			sh_n = sh_i + sh_j*sh_rl + sh_k*sh_rl*sh_rw

			#Load shared memory
			#center
			shared[sh_n] = f[r_n]

			#NOTE:
				#for shared memory loads if kernel block is smaller in a dim than the radius/genus it will miss loads
			#######################
			#genus offsets in 1d (for function)
			rady = genus*rshape[0]
			radz = genus*rshape[0]*rshape[1]

			#genus offsets in 1d (for shared memory)
			sh_rady = genus*sh_rl
			sh_radz = genus*sh_rl*sh_rw

			#faces
			if cuda.threadIdx.x < genus:
				shared[sh_n - genus] = f[r_n - genus]
			if cuda.threadIdx.x > cuda.blockDim.x - 1 - genus:
				shared[sh_n + genus] = f[r_n + genus]
			if cuda.threadIdx.y < genus:
				shared[sh_n - sh_rady] = f[r_n - rady]
			if cuda.threadIdx.y > cuda.blockDim.y - 1 - genus:
				shared[sh_n + sh_rady] = f[r_n + rady]
			if cuda.threadIdx.z < genus:
				shared[sh_n - sh_radz] = f[r_n - radz]
			if cuda.threadIdx.z > cuda.blockDim.z - 1 - genus:
				shared[sh_n + sh_radz] = f[r_n + radz]
			#######################

			#sync threads in block after loading shared memory
			cuda.syncthreads()

			#contribution to the total integral from this point
			contributions = 0

			#precompute denominator (it is constant over each levelset contribution)
			den = grad[n][0]*grad[n][0] + grad[n][1]*grad[n][1] + grad[n][2]*grad[n][2]

			#check if the denom isn't zero
			if den > EPS:
				#find min and max neighbors. Use neighbors to determine span of quad points that contribute to current point
				minimum = shared[sh_n]
				maximum = shared[sh_n]
				#need to index at 1 with genus when finding max/min (starting at 0 means you just check shared[sh_n] 12 times)
				for r in range(1, genus + 1):
					#min i neighbors
					minimum = min(shared[sh_n + r], minimum)
					minimum = min(shared[sh_n - r], minimum)
					#min j neighbors
					minimum = min(shared[sh_n + r*sh_rl], minimum)
					minimum = min(shared[sh_n - r*sh_rl], minimum)
					#min k neighbors
					minimum = min(shared[sh_n + r*sh_rl*sh_rw], minimum)
					minimum = min(shared[sh_n - r*sh_rl*sh_rw], minimum)

					#max i neighbors
					maximum = max(shared[sh_n + r], maximum)
					maximum = max(shared[sh_n - r], maximum)
					#max j neighbors
					maximum = max(shared[sh_n + r*sh_rl], maximum)
					maximum = max(shared[sh_n - r*sh_rl], maximum)
					#max k neighbors
					maximum = max(shared[sh_n + r*sh_rl*sh_rw], maximum)
					maximum = max(shared[sh_n - r*sh_rl*sh_rw], maximum)

				#span of function values between limits of integration
				span    = limit[1] - limit[0]
				#spacing of quad elements within span
				spacing = math.pow(2, -depth)
				#total number of quad points
				total   = (1 << depth) + 1
				#integer loction of min neighbor along span (clipped at lower and upper limit)
				start   = max(min(int((minimum - limit[0])/(span*spacing)), total), 0)
				#integer loction of max neighbor along span (clipped at lower and upper limit)
				stop    = max(min(int((maximum - limit[0])/(span*spacing)) + 1, total), 0)

				#number of levelsets to loop over
				for l in range(start,stop):
					#current levelset value
					levelset = l*spacing*span + limit[0]
					#corresponding quad element coeff
					quad = __quad(l, total)*spacing*span

					dchidx = 0.
					dchidy = 0.
					dchidz = 0.

					for s in range(2*genus + 1):
						c = s - genus

						dchidx += __chi(shared[sh_n + c],
										levelset)*stencil[s]

						dchidy += __chi(shared[sh_n + c*sh_rl],
										levelset)*stencil[s]

						dchidz += __chi(shared[sh_n + c*sh_rl*sh_rw],
										levelset)*stencil[s]

					num = (dchidx*grad[n][0] + dchidy*grad[n][1] + dchidz*grad[n][2])
					contributions += -quad*num/den
			integral[n] = contributions

		@cuda.jit('void(f8[:],f8[:],f8[:,:],i8[:],i8[:],f8[:],i8,i8[:])')
		def __compute_levelsets_outer(integral, f, grad, rshape, shape, limit, depth, offset):
			#######################
			#Identical to __compute_levelsets_inner with an offset and a bounds check
			#######################

			i,j,k = cuda.grid(3)

			#offset kernel indicies to position in function
			i += offset[0]
			j += offset[1]
			k += offset[2]

			#run a kernel bounds check on the outer kernel blocks
			if i >= shape[0] or j >= shape[1] or k >= shape[2]:
				return

			shared = cuda.shared.array(shape = (sh_size,), dtype = float64)
			stencil = cuda.const.array_like(constant)

			sh_i = cuda.threadIdx.x + genus
			sh_j = cuda.threadIdx.y + genus
			sh_k = cuda.threadIdx.z + genus

			n    = __flatten3d(i, j, k, shape)
			r_n  = __flatten3d(i + genus, j + genus, k + genus, rshape)
			sh_n = sh_i + sh_j*sh_rl + sh_k*sh_rl*sh_rw

			shared[sh_n] = f[r_n]

			rady = (genus*rshape[0])//genus
			radz = (genus*rshape[0]*rshape[1])//genus

			sh_rady = (genus*sh_rl)//genus
			sh_radz = (genus*sh_rl*sh_rw)//genus

			#slower shared memory loads but prevents misses if a dim is smaller than rad/genus
			if cuda.threadIdx.x < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r] = f[r_n - r]
			if cuda.threadIdx.x > cuda.blockDim.x - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r] = f[r_n + r]
			if cuda.threadIdx.y < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_rady] = f[r_n - r*rady]
			if cuda.threadIdx.y > cuda.blockDim.y - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_rady] = f[r_n + r*rady]
			if cuda.threadIdx.z < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_radz] = f[r_n - r*radz]
			if cuda.threadIdx.z > cuda.blockDim.z - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_radz] = f[r_n + r*radz]

			cuda.syncthreads()

			contributions = 0

			den = grad[n][0]*grad[n][0] + grad[n][1]*grad[n][1] + grad[n][2]*grad[n][2]

			if den > EPS:

				minimum = shared[sh_n]
				maximum = shared[sh_n]
				for r in range(1, genus + 1):
					minimum = min(shared[sh_n + r], minimum)
					minimum = min(shared[sh_n - r], minimum)
					minimum = min(shared[sh_n + r*sh_rl], minimum)
					minimum = min(shared[sh_n - r*sh_rl], minimum)
					minimum = min(shared[sh_n + r*sh_rl*sh_rw], minimum)
					minimum = min(shared[sh_n - r*sh_rl*sh_rw], minimum)

					maximum = max(shared[sh_n + r], maximum)
					maximum = max(shared[sh_n - r], maximum)
					maximum = max(shared[sh_n + r*sh_rl], maximum)
					maximum = max(shared[sh_n - r*sh_rl], maximum)
					maximum = max(shared[sh_n + r*sh_rl*sh_rw], maximum)
					maximum = max(shared[sh_n - r*sh_rl*sh_rw], maximum)

				span    = limit[1] - limit[0]
				spacing = math.pow(2, -depth)
				total   = (1 << depth) + 1
				start   = min(max(int((minimum - limit[0])/(span*spacing)), 0), total)
				stop    = max(min(int((maximum - limit[0])/(span*spacing)) + 1, total), 0)

				for l in range(start,stop):
					levelset = l*spacing*span + limit[0]
					quad = __quad(l, total)*spacing*span

					dchidx = 0.
					dchidy = 0.
					dchidz = 0.

					for s in range(2*genus + 1):
						c = s - genus

						dchidx += __chi(shared[sh_n + c],
										levelset)*stencil[s]

						dchidy += __chi(shared[sh_n + c*sh_rl],
										levelset)*stencil[s]

						dchidz += __chi(shared[sh_n + c*sh_rl*sh_rw],
										levelset)*stencil[s]

					num = (dchidx*grad[n][0] + dchidy*grad[n][1] + dchidz*grad[n][2])
					contributions += -quad*num/den
			integral[n] = contributions

		return __compute_levelsets_inner, __compute_levelsets_outer

	else:

		@cuda.jit('void(f8[:],f8[:],f8[:,:],i8[:],i8[:],f8[:],f8[:],i8)')
		def __compute_levelsets_inner_integrand(integral, f, grad, rshape, shape, g, limit, depth):
			#######################
			#Identical to __compute_levelsets_inner except with the integrand included
			#######################

			i,j,k = cuda.grid(3)

			shared = cuda.shared.array(shape = (sh_size,), dtype = float64)
			stencil = cuda.const.array_like(constant)

			sh_i = cuda.threadIdx.x + genus
			sh_j = cuda.threadIdx.y + genus
			sh_k = cuda.threadIdx.z + genus

			n    = __flatten3d(i, j, k, shape)
			r_n  = __flatten3d(i + genus, j + genus, k + genus, rshape)
			sh_n = sh_i + sh_j*sh_rl + sh_k*sh_rl*sh_rw

			shared[sh_n] = f[r_n]

			rady = genus*rshape[0]
			radz = genus*rshape[0]*rshape[1]

			sh_rady = genus*sh_rl
			sh_radz = genus*sh_rl*sh_rw

			#faces
			if cuda.threadIdx.x < genus:
				shared[sh_n - genus] = f[r_n - genus]
			if cuda.threadIdx.x > cuda.blockDim.x - 1 - genus:
				shared[sh_n + genus] = f[r_n + genus]
			if cuda.threadIdx.y < genus:
				shared[sh_n - sh_rady] = f[r_n - rady]
			if cuda.threadIdx.y > cuda.blockDim.y - 1 - genus:
				shared[sh_n + sh_rady] = f[r_n + rady]
			if cuda.threadIdx.z < genus:
				shared[sh_n - sh_radz] = f[r_n - radz]
			if cuda.threadIdx.z > cuda.blockDim.z - 1 - genus:
				shared[sh_n + sh_radz] = f[r_n + radz]

			cuda.syncthreads()

			contributions = 0

			den = grad[n][0]*grad[n][0] + grad[n][1]*grad[n][1] + grad[n][2]*grad[n][2]

			if den > EPS:
				minimum = shared[sh_n]
				maximum = shared[sh_n]
				for r in range(1, genus + 1):
					minimum = min(shared[sh_n + r], minimum)
					minimum = min(shared[sh_n - r], minimum)
					minimum = min(shared[sh_n + r*sh_rl], minimum)
					minimum = min(shared[sh_n - r*sh_rl], minimum)
					minimum = min(shared[sh_n + r*sh_rl*sh_rw], minimum)
					minimum = min(shared[sh_n - r*sh_rl*sh_rw], minimum)

					maximum = max(shared[sh_n + r], maximum)
					maximum = max(shared[sh_n - r], maximum)
					maximum = max(shared[sh_n + r*sh_rl], maximum)
					maximum = max(shared[sh_n - r*sh_rl], maximum)
					maximum = max(shared[sh_n + r*sh_rl*sh_rw], maximum)
					maximum = max(shared[sh_n - r*sh_rl*sh_rw], maximum)

				span    = limit[1] - limit[0]
				spacing = math.pow(2, -depth)
				total   = (1 << depth) + 1
				start   = max(min(int((minimum - limit[0])/(span*spacing)), total), 0)
				stop    = max(min(int((maximum - limit[0])/(span*spacing)) + 1, total), 0)

				for l in range(start,stop):
					levelset = l*spacing*span + limit[0]
					quad = __quad(l, total)*spacing*span

					dchidx = 0.
					dchidy = 0.
					dchidz = 0.

					for s in range(2*genus + 1):
						c = s - genus

						dchidx += __chi(shared[sh_n + c],
										levelset)*stencil[s]

						dchidy += __chi(shared[sh_n + c*sh_rl],
										levelset)*stencil[s]

						dchidz += __chi(shared[sh_n + c*sh_rl*sh_rw],
										levelset)*stencil[s]

					num = (dchidx*grad[n][0] + dchidy*grad[n][1] + dchidz*grad[n][2])
					contributions += -quad*num/den
			integral[n] = contributions*g[n]

		@cuda.jit('void(f8[:],f8[:],f8[:,:],i8[:],i8[:],f8[:],f8[:],i8,i8[:])')
		def __compute_levelsets_outer_integrand(integral, f, grad, rshape, shape, g, limit, depth, offset):
			#######################
			#Identical to __compute_levelsets_inner_integrand except with offset and bounds check
			#######################

			i,j,k = cuda.grid(3)

			i += offset[0]
			j += offset[1]
			k += offset[2]
			if i >= shape[0] or j >= shape[1] or k >= shape[2]:
				return

			shared = cuda.shared.array(shape = (sh_size,), dtype = float64)
			stencil = cuda.const.array_like(constant)

			sh_i = cuda.threadIdx.x + genus
			sh_j = cuda.threadIdx.y + genus
			sh_k = cuda.threadIdx.z + genus

			n    = __flatten3d(i, j, k, shape)
			r_n  = __flatten3d(i + genus, j + genus, k + genus, rshape)
			sh_n = sh_i + sh_j*sh_rl + sh_k*sh_rl*sh_rw

			shared[sh_n] = f[r_n]

			rady = (genus*rshape[0])//genus
			radz = (genus*rshape[0]*rshape[1])//genus

			sh_rady = (genus*sh_rl)//genus
			sh_radz = (genus*sh_rl*sh_rw)//genus

			if cuda.threadIdx.x < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r] = f[r_n - r]
			if cuda.threadIdx.x > cuda.blockDim.x - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r] = f[r_n + r]
			if cuda.threadIdx.y < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_rady] = f[r_n - r*rady]
			if cuda.threadIdx.y > cuda.blockDim.y - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_rady] = f[r_n + r*rady]
			if cuda.threadIdx.z < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_radz] = f[r_n - r*radz]
			if cuda.threadIdx.z > cuda.blockDim.z - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_radz] = f[r_n + r*radz]

			cuda.syncthreads()

			contributions = 0

			den = grad[n][0]*grad[n][0] + grad[n][1]*grad[n][1] + grad[n][2]*grad[n][2]

			if den > EPS:

				minimum = shared[sh_n]
				maximum = shared[sh_n]
				for r in range(1, genus + 1):
					minimum = min(shared[sh_n + r], minimum)
					minimum = min(shared[sh_n - r], minimum)
					minimum = min(shared[sh_n + r*sh_rl], minimum)
					minimum = min(shared[sh_n - r*sh_rl], minimum)
					minimum = min(shared[sh_n + r*sh_rl*sh_rw], minimum)
					minimum = min(shared[sh_n - r*sh_rl*sh_rw], minimum)

					maximum = max(shared[sh_n + r], maximum)
					maximum = max(shared[sh_n - r], maximum)
					maximum = max(shared[sh_n + r*sh_rl], maximum)
					maximum = max(shared[sh_n - r*sh_rl], maximum)
					maximum = max(shared[sh_n + r*sh_rl*sh_rw], maximum)
					maximum = max(shared[sh_n - r*sh_rl*sh_rw], maximum)

				span    = limit[1] - limit[0]
				spacing = math.pow(2, -depth)
				total   = (1 << depth) + 1
				start   = min(max(int((minimum - limit[0])/(span*spacing)), 0), total)
				stop    = max(min(int((maximum - limit[0])/(span*spacing)) + 1, total), 0)

				for l in range(start,stop):
					levelset = l*spacing*span + limit[0]
					quad = __quad(l, total)*spacing*span

					dchidx = 0.
					dchidy = 0.
					dchidz = 0.

					for s in range(2*genus + 1):
						c = s - genus

						dchidx += __chi(shared[sh_n + c],
										levelset)*stencil[s]

						dchidy += __chi(shared[sh_n + c*sh_rl],
										levelset)*stencil[s]

						dchidz += __chi(shared[sh_n + c*sh_rl*sh_rw],
										levelset)*stencil[s]

					num = (dchidx*grad[n][0] + dchidy*grad[n][1] + dchidz*grad[n][2])
					contributions += -quad*num/den
			integral[n] = contributions*g[n]

		return __compute_levelsets_inner_integrand, __compute_levelsets_outer_integrand

def compile_levelsets_adaptive(genus, g):
	#compile kernel with genus as a constant and use the specified quad function
	global central

	constant = central[genus - 1]

	sh_shape  = _threads
	sh_rshape = sh_shape + 2*genus
	sh_size   = np.prod(sh_rshape)

	sh_l = sh_shape[0]
	sh_w = sh_shape[1]
	sh_h = sh_shape[2]

	sh_rl = sh_rshape[0]
	sh_rw = sh_rshape[1]
	sh_rh = sh_rshape[2]

	if g is None:

		@cuda.jit('void(f8[:],f8[:],f8[:,:],i8[:],i8[:],f8)')
		def __compute_levelsets_inner(integral, f, grad, rshape, shape, levelset):
			#######################
			#Very similar to __compute_levelsets_inner (fixed). This only computes one levelset
			#######################

			i,j,k = cuda.grid(3)

			shared = cuda.shared.array(shape = (sh_size,), dtype = float64)
			stencil = cuda.const.array_like(constant)

			sh_i = cuda.threadIdx.x + genus
			sh_j = cuda.threadIdx.y + genus
			sh_k = cuda.threadIdx.z + genus

			n    = __flatten3d(i, j, k, shape)
			r_n  = __flatten3d(i + genus, j + genus, k + genus, rshape)
			sh_n = sh_i + sh_j*sh_rl + sh_k*sh_rl*sh_rw

			shared[sh_n] = f[r_n]

			rady = (genus*rshape[0])//genus
			radz = (genus*rshape[0]*rshape[1])//genus
	 
			sh_rady = (genus*sh_rl)//genus
			sh_radz = (genus*sh_rl*sh_rw)//genus

			if cuda.threadIdx.x < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r] = f[r_n - r]
			if cuda.threadIdx.x > cuda.blockDim.x - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r] = f[r_n + r]
			if cuda.threadIdx.y < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_rady] = f[r_n - r*rady]
			if cuda.threadIdx.y > cuda.blockDim.y - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_rady] = f[r_n + r*rady]
			if cuda.threadIdx.z < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_radz] = f[r_n - r*radz]
			if cuda.threadIdx.z > cuda.blockDim.z - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_radz] = f[r_n + r*radz]

			cuda.syncthreads()

			den = grad[n][0]*grad[n][0] + grad[n][1]*grad[n][1] + grad[n][2]*grad[n][2]
			if den > EPS:

				dchidx = 0.
				dchidy = 0.
				dchidz = 0.

				for s in range(2*genus + 1):
					c = s - genus

					dchidx += __chi(shared[sh_n + c],
									levelset)*stencil[s]

					dchidy += __chi(shared[sh_n + c*sh_rl],
									levelset)*stencil[s]

					dchidz += __chi(shared[sh_n + c*sh_rl*sh_rw],
									levelset)*stencil[s]

				num = (dchidx*grad[n][0] + dchidy*grad[n][1] + dchidz*grad[n][2])
				integral[n] = -num/den

		@cuda.jit('void(f8[:],f8[:],f8[:,:],i8[:],i8[:],f8,i8[:])')
		def __compute_levelsets_outer(integral, f, grad, rshape, shape, levelset, offset):
			i,j,k = cuda.grid(3)

			i += offset[0]
			j += offset[1]
			k += offset[2]

			if i >= shape[0] or j >= shape[1] or k >= shape[2]:
				return

			shared = cuda.shared.array(shape = (sh_size,), dtype = float64)
			stencil = cuda.const.array_like(constant)

			sh_i = cuda.threadIdx.x + genus
			sh_j = cuda.threadIdx.y + genus
			sh_k = cuda.threadIdx.z + genus

			n    = __flatten3d(i, j, k, shape)
			r_n  = __flatten3d(i + genus, j + genus, k + genus, rshape)
			sh_n = sh_i + sh_j*sh_rl + sh_k*sh_rl*sh_rw

			shared[sh_n] = f[r_n]

			rady = (genus*rshape[0])//genus
			radz = (genus*rshape[0]*rshape[1])//genus
	 
			sh_rady = (genus*sh_rl)//genus
			sh_radz = (genus*sh_rl*sh_rw)//genus

			if cuda.threadIdx.x < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r] = f[r_n - r]
			if cuda.threadIdx.x > cuda.blockDim.x - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r] = f[r_n + r]
			if cuda.threadIdx.y < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_rady] = f[r_n - r*rady]
			if cuda.threadIdx.y > cuda.blockDim.y - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_rady] = f[r_n + r*rady]
			if cuda.threadIdx.z < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_radz] = f[r_n - r*radz]
			if cuda.threadIdx.z > cuda.blockDim.z - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_radz] = f[r_n + r*radz]

			cuda.syncthreads()

			den = grad[n][0]*grad[n][0] + grad[n][1]*grad[n][1] + grad[n][2]*grad[n][2]
			if den > EPS:

				dchidx = 0.
				dchidy = 0.
				dchidz = 0.

				for s in range(2*genus + 1):
					c = s - genus

					dchidx += __chi(shared[sh_n + c],
									levelset)*stencil[s]

					dchidy += __chi(shared[sh_n + c*sh_rl],
									levelset)*stencil[s]

					dchidz += __chi(shared[sh_n + c*sh_rl*sh_rw],
									levelset)*stencil[s]

				num = (dchidx*grad[n][0] + dchidy*grad[n][1] + dchidz*grad[n][2])
				integral[n] = -num/den

		return __compute_levelsets_inner, __compute_levelsets_outer

	else:

		@cuda.jit('void(f8[:],f8[:],f8[:,:],i8[:],i8[:],f8[:],f8)')
		def __compute_levelsets_inner_integrand(integral, f, grad, rshape, shape, g, levelset):
			i,j,k = cuda.grid(3)

			shared = cuda.shared.array(shape = (sh_size,), dtype = float64)
			stencil = cuda.const.array_like(constant)

			sh_i = cuda.threadIdx.x + genus
			sh_j = cuda.threadIdx.y + genus
			sh_k = cuda.threadIdx.z + genus

			n    = __flatten3d(i, j, k, shape)
			r_n  = __flatten3d(i + genus, j + genus, k + genus, rshape)
			sh_n = sh_i + sh_j*sh_rl + sh_k*sh_rl*sh_rw

			shared[sh_n] = f[r_n]

			rady = (genus*rshape[0])//genus
			radz = (genus*rshape[0]*rshape[1])//genus
	 
			sh_rady = (genus*sh_rl)//genus
			sh_radz = (genus*sh_rl*sh_rw)//genus

			if cuda.threadIdx.x < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r] = f[r_n - r]
			if cuda.threadIdx.x > cuda.blockDim.x - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r] = f[r_n + r]
			if cuda.threadIdx.y < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_rady] = f[r_n - r*rady]
			if cuda.threadIdx.y > cuda.blockDim.y - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_rady] = f[r_n + r*rady]
			if cuda.threadIdx.z < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_radz] = f[r_n - r*radz]
			if cuda.threadIdx.z > cuda.blockDim.z - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_radz] = f[r_n + r*radz]

			cuda.syncthreads()

			den = grad[n][0]*grad[n][0] + grad[n][1]*grad[n][1] + grad[n][2]*grad[n][2]
			if den > EPS:

				dchidx = 0.
				dchidy = 0.
				dchidz = 0.

				for s in range(2*genus + 1):
					c = s - genus

					dchidx += __chi(shared[sh_n + c],
									levelset)*stencil[s]

					dchidy += __chi(shared[sh_n + c*sh_rl],
									levelset)*stencil[s]

					dchidz += __chi(shared[sh_n + c*sh_rl*sh_rw],
									levelset)*stencil[s]

				num = (dchidx*grad[n][0] + dchidy*grad[n][1] + dchidz*grad[n][2])
				integral[n] = -num/den*g[n]

		@cuda.jit('void(f8[:],f8[:],f8[:,:],i8[:],i8[:],f8[:],f8,i8[:])')
		def __compute_levelsets_outer_integrand(integral, f, grad, rshape, shape, g, levelset, offset):
			i,j,k = cuda.grid(3)

			i += offset[0]
			j += offset[1]
			k += offset[2]

			if i >= shape[0] or j >= shape[1] or k >= shape[2]:
				return

			shared = cuda.shared.array(shape = (sh_size,), dtype = float64)
			stencil = cuda.const.array_like(constant)

			sh_i = cuda.threadIdx.x + genus
			sh_j = cuda.threadIdx.y + genus
			sh_k = cuda.threadIdx.z + genus

			n    = __flatten3d(i, j, k, shape)
			r_n  = __flatten3d(i + genus, j + genus, k + genus, rshape)
			sh_n = sh_i + sh_j*sh_rl + sh_k*sh_rl*sh_rw

			shared[sh_n] = f[r_n]

			rady = (genus*rshape[0])//genus
			radz = (genus*rshape[0]*rshape[1])//genus
	 
			sh_rady = (genus*sh_rl)//genus
			sh_radz = (genus*sh_rl*sh_rw)//genus

			if cuda.threadIdx.x < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r] = f[r_n - r]
			if cuda.threadIdx.x > cuda.blockDim.x - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r] = f[r_n + r]
			if cuda.threadIdx.y < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_rady] = f[r_n - r*rady]
			if cuda.threadIdx.y > cuda.blockDim.y - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_rady] = f[r_n + r*rady]
			if cuda.threadIdx.z < 1:
				for r in range(1, genus + 1):
					shared[sh_n - r*sh_radz] = f[r_n - r*radz]
			if cuda.threadIdx.z > cuda.blockDim.z - 2:
				for r in range(1, genus + 1):
					shared[sh_n + r*sh_radz] = f[r_n + r*radz]

			cuda.syncthreads()

			den = grad[n][0]*grad[n][0] + grad[n][1]*grad[n][1] + grad[n][2]*grad[n][2]
			if den > EPS:

				dchidx = 0.
				dchidy = 0.
				dchidz = 0.

				for s in range(2*genus + 1):
					c = s - genus

					dchidx += __chi(shared[sh_n + c],
									levelset)*stencil[s]

					dchidy += __chi(shared[sh_n + c*sh_rl],
									levelset)*stencil[s]

					dchidz += __chi(shared[sh_n + c*sh_rl*sh_rw],
									levelset)*stencil[s]

				num = (dchidx*grad[n][0] + dchidy*grad[n][1] + dchidz*grad[n][2])
				integral[n] = -num/den*g[n]

		return __compute_levelsets_inner_integrand, __compute_levelsets_outer_integrand

def compile_levelsets(genus, quad, g, levelsets):
	kernels = None

	if levelsets == 'fixed':
		kernels = compile_levelsets_fixed(genus, quad, g)
	elif levelsets == 'adaptive':
		kernels = compile_levelsets_adaptive(genus, g)

	return kernels

def compute_grad(kernel, d_grad, d_f, d_rshape, d_shape):
	threads = min(ethreads, d_grad.shape[0])
	blocks  = int(max((d_grad.shape[0] + threads - 1)//threads, 1))

	kernel[blocks, threads](d_grad, d_f, d_rshape, d_shape)

def compile_grad(genus):
	global central

	cconstant = central[genus - 1]

	@cuda.jit('void(f8[:,:],f8[:],i8[:],i8[:])')
	def __compute_grad(grad, f, rshape, shape):
		offset = cuda.grid(1)
		stride = cuda.gridDim.x * cuda.blockDim.x

		cstencil = cuda.const.array_like(cconstant)

		for n in range(offset, shape[0]*shape[1]*shape[2], stride):
			i,j,k = __unflatten3d(n, shape)

			dfdx = 0
			dfdy = 0
			dfdz = 0

			for s in range(2*genus + 1):
				c = s - genus

				dfdx += f[__flatten3d(i + genus + c,
									  j + genus,
									  k + genus,
									  rshape)]*cstencil[s]

				dfdy += f[__flatten3d(i + genus,
									  j + genus + c,
									  k + genus,
									  rshape)]*cstencil[s]

				dfdz += f[__flatten3d(i + genus,
									  j + genus,
									  k + genus + c,
									  rshape)]*cstencil[s]

			grad[n][0] = dfdx
			grad[n][1] = dfdy
			grad[n][2] = dfdz

	return __compute_grad

def compute_fg(kernel, d_fg, d_T, d_shape, res):
	if kernel is None:
		return None

	threads = min(ethreads, d_fg.size)
	blocks  = int(max((d_fg.size + threads - 1)//threads, 1))

	kernel[blocks, threads](d_fg, d_shape, d_T, np.float64(res))

def compile_f(function):
	return _compile_fg(function)

def compile_g(function):
	return _compile_fg(function)

def _compile_fg(function):
	if function is None:
		return None

	function = cuda.jit('f8(f8,f8,f8)', device = True)(function)

	@cuda.jit('void(f8[:],i8[:],f8[:],f8)')
	def __compute_fg(f, shape, T, res):
		offset = cuda.grid(1)
		stride = cuda.gridDim.x * cuda.blockDim.x

		for n in range(offset, shape[0]*shape[1]*shape[2], stride):
			i,j,k = __unflatten3d(n, shape)

			x = (i + 0.5 - shape[0]/2.)/res
			y = (j + 0.5 - shape[1]/2.)/res
			z = (k + 0.5 - shape[2]/2.)/res
			w = 1.

			a = T[0]*x + T[1]*y + T[2]*z  + T[3]*w
			b = T[4]*x + T[5]*y + T[6]*z  + T[7]*w
			c = T[8]*x + T[9]*y + T[10]*z + T[11]*w

			f[n] = function(a,b,c)

	return __compute_fg

def map_to_quad(f, limit, depth, add = False):
	span    = limit[1] - limit[0]
	spacing = math.pow(2, -depth)
	total   = (1 << depth) + 1

	#for max
	if add:
		qf = max(min(int((f - limit[0])/(span*spacing)) + 1, total), 0)*spacing*span + limit[0]
	#for min
	else:
		qf = max(min(int((f - limit[0])/(span*spacing))    , total), 0)*spacing*span + limit[0]

	return qf

def chunk_domain(shape, cshape, res, o):
	#generates subdomains and function offsets from original domain
	chunks = []
	Tn     = []

	#number of chunks in each dimension
	count = (shape + cshape - 1)//cshape

	#no function offset was provided
	if o is None:
		o = np.zeros(3)

	offset = np.empty(3, dtype = np.int64)
	span   = np.empty((3,2), dtype = np.float64)
	c      = np.empty(3, dtype = np.float64)
	for x in range(count[0]):
		for y in range(count[1]):
			for z in range(count[2]):
				offset[0] = x + 1
				offset[1] = y + 1
				offset[2] = z + 1

				nmax = np.minimum(shape, offset*cshape)

				#get span of each subdomain
				for n in range(3):
					span[n] = [(offset[n] - 1)*cshape[n], nmax[n]]

				#shift subdomain so original domain is centered at (0,0,0)
				for n in range(3):
					span[n] -= shape[n]/2

				#find center point of chunk
				for n in range(3):
					c[n] = (span[n][1] + span[n][0])/2

				#offset function by subdomain offset
				T = np.array([[1,0,0,-(c[0] + o[0])/res],
							  [0,1,0,-(c[1] + o[1])/res],
							  [0,0,1,-(c[2] + o[2])/res],
							  [0,0,0, 1                ]], dtype = np.float64)

				chunks.append(nmax - (offset - 1)*cshape)
				Tn.append(T)

	return chunks, Tn

def integrate_function(size = [1,1,1], limit = [-np.inf, 0], f = None, g = None, grad = None, levelset = None, depth = 6, res = 16, genus = 2, quad = 'simpsons', scheme = 'fixed', bounds = 'global', err = 1e-6, offset = None):
	global chunkshape

	#NOTE:
		#bounds = 'local'
			#The error dependent on the chunk size. Because of this it might be undesirable
			#However it is faster as the function does not need to evaluated twice

	if limit[0] >= limit[1]:
		#print(the first limit must be lower than the second)
		return 0

	size = np.array(size, dtype = np.float64)

	fkernel    = f
	gkernel    = g
	gradkernel = grad
	lkernels   = levelset

	if not isinstance(fkernel, cuda.compiler.CUDAKernel):
		print('Compiling f on the GPU')
		fkernel = compile_f(f)
	if not isinstance(gkernel, cuda.compiler.CUDAKernel):
		print('Compiling g on the GPU')
		gkernel = compile_g(g)
	if not isinstance(gradkernel, cuda.compiler.CUDAKernel):
		print('Compiling grad on the GPU')
		gradkernel = compile_grad(genus)
	if not isinstance(lkernels, cuda.compiler.CUDAKernel):
		print('Compiling levelsets on the GPU')
		lkernels = compile_levelsets(genus, quad, gkernel, scheme)

	print('Integration has started')

	start = time.time()

	#shape of entire domain
	shape = (np.ceil(size*res)).astype(np.int64)

	#chunk the domain here. Needed if domain size cannot fit into GPU memory
	chunks, Tn = chunk_domain(shape, chunkshape, res, offset)

	#generate device arrays for largest chunk and reuse for all other chunks
	largest = chunks[0]
	for chunk in chunks[1:]:
		if np.prod(chunk) > np.prod(largest):
			largest = chunk

	#dilate function domain by the radius of the central diff stencil (for gradients) so it fits over entire domain
	d_f         = cuda.device_array((np.prod(largest + 2*genus), ), dtype = np.float64)
	d_grad      = cuda.device_array((np.prod(largest), 3), dtype = np.float64)
	d_integral  = cuda.device_array((np.prod(largest),  ), dtype = np.float64)
	d_g         = None
	#only generate integrand array if an integrand function is given
	if g is not None:
		d_g = cuda.device_array((np.prod(largest),  ), dtype = np.float64)
	# cuda.pinned_array(shape, dtype=np.float)

	#sets limits if global bounds are selected and +-inf is given by finding the global min/max
	fmins = []
	fmaxs = []
	if bounds == 'global':
		#if there is more than one chunk and the limits are not provided need to search for domain's global min/max
		if len(chunks) > 1 and (limit[0] == -np.inf or limit[1] == np.inf):
			for n in range(len(chunks)):
				chunk = chunks[n]

				#get current transform and move it to the gpu
				T   = Tn[n]
				d_T = cuda.to_device(T.flatten())

				#add radius around chunk shape and move that to gpu
				rchunk   = chunk + 2*genus
				d_rchunk = cuda.to_device(rchunk)

				compute_fg(fkernel, d_f, d_T, d_rchunk,  res)

				#find local min of each chunk
				if limit[0] == -np.inf:
					fmins.append(min_gpu(d_f, np.prod(rchunk)))
				#find local max of each chunk
				if limit[1] == np.inf:
					fmaxs.append(max_gpu(d_f, np.prod(rchunk)))

			#find global min and set lower bound of integration
			limit[0] = min(fmins) if fmins else limit[0]
			#find global max and set upper bound of integration
			limit[1] = min(fmaxs) if fmaxs else limit[1]

	integral = 0
	#iterate over each domain chunk and compute the itegral over subdomain
	for n in range(len(chunks)):
		print(f'integrating chunk {n + 1}/{len(chunks)}')

		fmin = limit[0]
		fmax = limit[1]

		#get current chunk shape and move it to the gpu
		chunk   = chunks[n]
		d_chunk = cuda.to_device(chunk)

		#get current transform and move it to the gpu
		T   = Tn[n]
		d_T = cuda.to_device(T.flatten())

		#add radius around chunk shape (to fit stencil) and move to gpu
		rchunk   = chunk + 2*genus
		d_rchunk = cuda.to_device(rchunk)

		compute_fg(fkernel, d_f, d_T, d_rchunk,  res)

		compute_fg(gkernel, d_g, d_T, d_chunk, res)

		compute_grad(gradkernel, d_grad, d_f, d_rchunk, d_chunk)

		if bounds == 'global':
			#if there is only one chunk and bounds are not given find min/max of already evaluated function
			if len(chunks) == 1:
				#find global min
				if limit[0] == -np.inf:
					limit[0] = min_gpu(d_f, np.prod(rchunk))
				#find global min
				if limit[1] == np.inf:
					limit[1] = max_gpu(d_f, np.prod(rchunk))
				fmin = limit[0]
				fmax = limit[1]

			#if there is more than one chunk, check if there are preevaluated fmins and/or fmaxs from finding global min/max
			elif (fmins or fmaxs) and scheme == 'adaptive':
				#map local min/max to a quad point within limits of integration
				if fmins:
					fmin = max(fmin, map_to_quad(fmins[n], limit, depth))
				if fmaxs:
					fmax = min(fmax, map_to_quad(fmaxs[n], limit, depth, add = True))

		#if local find min/max after each chunk and adjust limits to be as conservative as possible
		elif bounds == 'local':
			fmin = max(fmin, min_gpu(d_f, np.prod(rchunk)))
			fmax = min(fmax, max_gpu(d_f, np.prod(rchunk)))

		kargs = [d_integral, d_f, d_grad, d_rchunk, d_chunk]
		if g is not None:
			kargs.append(d_g)

		if limit[0] >= limit[1]:
			return 0
		elif fmax < limit[0]:
			integral += 0
		elif fmin > limit[1]:
			integral += 0

		elif scheme == 'fixed':
			integral += integrate_fixed(lkernels, kargs, chunk, res, depth, fmin, fmax)

		elif scheme == 'adaptive':
			integral += integrate_adaptive(lkernels, kargs, chunk, res, depth, fmin, fmax, err, quad)

	end = time.time()
	# print(end - start)

	return integral
