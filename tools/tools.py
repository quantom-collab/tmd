# import cPickle 
try:
    import cPickle
except:
    import _pickle as cPickle
import sys, os
import zlib
import torch
import gc
from torch import Tensor
from scipy.spatial import Delaunay

def checkdir(path):
  if not os.path.exists(path): 
    os.makedirs(path)

def tex(x):
  return r'$\mathrm{'+x+'}$'

def save(data,name):  
  compressed=zlib.compress(cPickle.dumps(data))
  f=open(name,"wb")
  try:
      f.writelines(compressed)
  except:
      f.write(compressed)
  f.close()

def load(name): 
  compressed=open(name,"rb").read()
  data=cPickle.loads(zlib.decompress(compressed))
  return data

def load2(name): 
  compressed=open(name,"rb").read()
  data=cPickle.loads(compressed)
  return data

def isnumeric(value):
  try:
    int(value)
    return True
  except:
    return False

def ERR(msg):
  print(msg)
  sys.exit()

def lprint(msg):
  sys.stdout.write('\r')
  sys.stdout.write(msg)
  sys.stdout.flush()


## Written by Zhite Yu
## Generalizing Nobuo's interpolation code to arbitrary dimension
def interpD(x_: Tensor, xp_: Tensor, fp_: Tensor, dim=-1) -> Tensor:
    """
    Simple linear interpolation along the dimension 'dim' (the last one by default).
    :param x_: gridpoints to interpolate to (not necessarily sorted)
    :param xp_: gridpoints to interpolate from (sorted grid)
    :param fp_: values of the function (at grid points x_) to interpolate (from xp_ to x_)
    x_, xp_, and fp_ are of the same dimension.
    xp_ and fp_ have the same shape.
    The size of x_ may only differ from the xp_ and fp_ in the 'dim' dimension.
    """
    # change the dim to the last dimension
    x, xp, fp = x_, xp_, fp_
    if dim != -1 or dim != x.dim() - 1:
        x, xp, fp = x_.transpose(dim, -1), xp_.transpose(dim, -1), fp_.transpose(dim, -1)
    # slope for each segment (the index corresponds to the left endpoint at each segment)
    slopes = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1])
    # intercepts for each linear segment: which takes y = k * x + b in [x_i, x_{i+1})
    intercepts = fp[..., :-1] - (slopes * xp[..., :-1])
    # get the indices for the x table to be interpolated
    indicies = torch.sum(torch.ge(x[..., :, None], xp[..., None, :]), dim=-1) - 1
    indicies = torch.clamp(indicies, 0, slopes.shape[-1] - 1)
    f_interp = torch.gather(slopes, -1, indicies) * x + torch.gather(intercepts, -1, indicies)
    # change the dim back
    if dim != -1 or dim != x.dim() - 1:
        f_interp = f_interp.transpose(dim, -1)
    return f_interp

#--get memory of tensor in Gigabytes
def mem(T):
    mem = T.element_size()*T.nelement()/1024**3
    print('%5.4f GB'%mem)
    return mem

#--get total memory used
def totmem(message=None):
    if message != None: print(message)
    print("torch.cuda.memory_allocated: %4.3f GB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %4.3f GB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %4.3f GB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

#--clears all objects from GPU memory (not working?)
def clear_mem():
    objs = gc.get_objects()
    for obj in objs: del obj
    torch.cuda.empty_cache()

#--interpolation matrices written by Daniel Adamiak
#--interpolate from x0 to x1
#--x0 must be a 1D array
#--x1 can have any shape
def interpolate(x0,x1,kind,dev,dtype=torch.float64):
    x1_shape = x1.shape
    x1 = x1.flatten()
    L = torch.zeros((x1.shape[0],x0.shape[0]),dtype=dtype).to(dev)

    if kind=='linear':
        indices = torch.searchsorted(x0,x1)
        i  = torch.clamp(indices-1,0,len(x0)-2)
        i1 = i+1
        weights = (x1-x0[i])/(x0[i1]-x0[i])
        for row in range(len(x1)):
            L[row, i[row]]  = 1 - weights[row]
            L[row, i1[row]] = weights[row]

    if kind=='cubic':
        for i, x in enumerate(x1):
            # Find the index i such that x0[i] <= x1_point <= x0[i+1]
            i_index = torch.searchsorted(x0, x, right=True)-1
            i_index = torch.clamp(i_index,0,len(x0)-2)

            # Compute interpolation weights
            a, b, c, d = 1,1,1,1
            if i_index>0:xLm = x0[i_index-1]
            xL  = x0[i_index]
            xR  = x0[i_index+1]
            if i_index<len(x0) - 2:xRp = x0[i_index+2]

            # Populate the row in L
            if i_index == 0:
                # Brute force cubic interpolation for the first grid interval
                x_0=x0[0]
                x_1=x0[1]
                x_2=x0[2]
                x_3=x0[3]

                a0 = x_1*x_2*x_3/((x_1-x_0)*(x_2-x_0)*(x_3-x_0))
                a1 = x_0*x_2*x_3/((x_0-x_1)*(x_1-x_2)*(x_1-x_3))
                a2 = x_0*x_1*x_3/((x_0-x_2)*(x_2-x_1)*(x_2-x_3))
                a3 = x_0*x_1*x_2/((x_0-x_3)*(x_3-x_1)*(x_3-x_2))

                b0 = (x_1*x_2+x_1*x_3+x_2*x_3)/((x_0-x_1)*(x_0-x_2)*(x_0-x_3))
                b1 = (x_0*x_2+x_0*x_3+x_2*x_3)/((x_1-x_0)*(x_1-x_2)*(x_1-x_3))
                b2 = (x_0*x_1+x_0*x_3+x_1*x_3)/((x_2-x_0)*(x_2-x_1)*(x_2-x_3))
                b3 = (x_0*x_1+x_0*x_2+x_1*x_2)/((x_3-x_0)*(x_3-x_1)*(x_3-x_2))    

                c0 = (x_1+x_2+x_3)/((x_1-x_0)*(x_2-x_0)*(x_3-x_0))
                c1 = (x_0+x_2+x_3)/((x_0-x_1)*(x_2-x_1)*(x_3-x_1))
                c2 = (x_0+x_1+x_3)/((x_0-x_2)*(x_1-x_2)*(x_3-x_2))
                c3 = (x_0+x_1+x_2)/((x_0-x_3)*(x_1-x_3)*(x_2-x_3))

                d0 = 1/((x_0-x_1)*(x_0-x_2)*(x_0-x_3))
                d1 = 1/((x_1-x_0)*(x_1-x_2)*(x_1-x_3))
                d2 = 1/((x_2-x_0)*(x_2-x_1)*(x_2-x_3))
                d3 = 1/((x_3-x_0)*(x_3-x_1)*(x_3-x_2))
                
                p0 = a0 + b0*x + c0*x**2 + d0*x**3
                p1 = a1 + b1*x + c1*x**2 + d1*x**3
                p2 = a2 + b2*x + c2*x**2 + d2*x**3
                p3 = a3 + b3*x + c3*x**2 + d3*x**3
                #populate
                L[i, i_index:i_index + 4] = torch.tensor([p0, p1, p2, p3])
            elif i_index == len(x0) - 2:
              # Brute force cubic interpolation for the first grid interval
                x_0=x0[-4]
                x_1=x0[-3]
                x_2=x0[-2]
                x_3=x0[-1]

                a0 = x_1*x_2*x_3/((x_1-x_0)*(x_2-x_0)*(x_3-x_0))
                a1 = x_0*x_2*x_3/((x_0-x_1)*(x_1-x_2)*(x_1-x_3))
                a2 = x_0*x_1*x_3/((x_0-x_2)*(x_2-x_1)*(x_2-x_3))
                a3 = x_0*x_1*x_2/((x_0-x_3)*(x_3-x_1)*(x_3-x_2))

                b0 = (x_1*x_2+x_1*x_3+x_2*x_3)/((x_0-x_1)*(x_0-x_2)*(x_0-x_3))
                b1 = (x_0*x_2+x_0*x_3+x_2*x_3)/((x_1-x_0)*(x_1-x_2)*(x_1-x_3))
                b2 = (x_0*x_1+x_0*x_3+x_1*x_3)/((x_2-x_0)*(x_2-x_1)*(x_2-x_3))
                b3 = (x_0*x_1+x_0*x_2+x_1*x_2)/((x_3-x_0)*(x_3-x_1)*(x_3-x_2))    

                c0 = (x_1+x_2+x_3)/((x_1-x_0)*(x_2-x_0)*(x_3-x_0))
                c1 = (x_0+x_2+x_3)/((x_0-x_1)*(x_2-x_1)*(x_3-x_1))
                c2 = (x_0+x_1+x_3)/((x_0-x_2)*(x_1-x_2)*(x_3-x_2))
                c3 = (x_0+x_1+x_2)/((x_0-x_3)*(x_1-x_3)*(x_2-x_3))

                d0 = 1/((x_0-x_1)*(x_0-x_2)*(x_0-x_3))
                d1 = 1/((x_1-x_0)*(x_1-x_2)*(x_1-x_3))
                d2 = 1/((x_2-x_0)*(x_2-x_1)*(x_2-x_3))
                d3 = 1/((x_3-x_0)*(x_3-x_1)*(x_3-x_2))
                
                p0 = a0 + b0*x + c0*x**2 + d0*x**3
                p1 = a1 + b1*x + c1*x**2 + d1*x**3
                p2 = a2 + b2*x + c2*x**2 + d2*x**3
                p3 = a3 + b3*x + c3*x**2 + d3*x**3
                #populate
                L[i, i_index-2:i_index + 2] = torch.tensor([p0, p1, p2, p3])   
            else:
                #weights
                a = -(x-xL)/(xR-xLm) + (x-xL)**2/(xR-xL)/(xR-xLm) - (x-xL)**2*(x-xR)/(xR-xL)**2/(xR-xLm)
                b = 1 - (x-xL)**2/(xR-xL)**2 - (x-xL)**2*(x-xR)/(xR-xL)**2/(xRp-xL) + 2*(x-xL)**2*(x-xR)/(xR-xL)**3
                c = (x-xL)/(xR-xLm) + (x-xL)**2/(xR-xL)**2 - (x-xL)**2/(xR-xL)/(xR-xLm) + (x-xL)**2*(x-xR)/(xR-xL)**2/(xR-xLm) -  2*(x-xL)**2*(x-xR)/(xR-xL)**3
                d = (x-xL)**2*(x-xR)/(xR-xL)**2/(xRp-xL)
                #populate
                L[i, i_index - 1:i_index + 3] = torch.tensor([a, b, c, d])

    if kind=='cubic-turbo':
        def coeff_start(xt): #compute interpolation coefficients near starting boundary
            # Brute force cubic interpolation for the first grid interval
            x  =x1[xt] 
            x_0=x0[0]
            x_1=x0[1]
            x_2=x0[2]
            x_3=x0[3]

            a0 = x_1*x_2*x_3/((x_1-x_0)*(x_2-x_0)*(x_3-x_0))
            a1 = x_0*x_2*x_3/((x_0-x_1)*(x_1-x_2)*(x_1-x_3))
            a2 = x_0*x_1*x_3/((x_0-x_2)*(x_2-x_1)*(x_2-x_3))
            a3 = x_0*x_1*x_2/((x_0-x_3)*(x_3-x_1)*(x_3-x_2))

            b0 = (x_1*x_2+x_1*x_3+x_2*x_3)/((x_0-x_1)*(x_0-x_2)*(x_0-x_3))
            b1 = (x_0*x_2+x_0*x_3+x_2*x_3)/((x_1-x_0)*(x_1-x_2)*(x_1-x_3))
            b2 = (x_0*x_1+x_0*x_3+x_1*x_3)/((x_2-x_0)*(x_2-x_1)*(x_2-x_3))
            b3 = (x_0*x_1+x_0*x_2+x_1*x_2)/((x_3-x_0)*(x_3-x_1)*(x_3-x_2))    

            c0 = (x_1+x_2+x_3)/((x_1-x_0)*(x_2-x_0)*(x_3-x_0))
            c1 = (x_0+x_2+x_3)/((x_0-x_1)*(x_2-x_1)*(x_3-x_1))
            c2 = (x_0+x_1+x_3)/((x_0-x_2)*(x_1-x_2)*(x_3-x_2))
            c3 = (x_0+x_1+x_2)/((x_0-x_3)*(x_1-x_3)*(x_2-x_3))

            d0 = 1/((x_0-x_1)*(x_0-x_2)*(x_0-x_3))
            d1 = 1/((x_1-x_0)*(x_1-x_2)*(x_1-x_3))
            d2 = 1/((x_2-x_0)*(x_2-x_1)*(x_2-x_3))
            d3 = 1/((x_3-x_0)*(x_3-x_1)*(x_3-x_2))

            p0 = a0 + b0*x + c0*x**2 + d0*x**3
            p1 = a1 + b1*x + c1*x**2 + d1*x**3
            p2 = a2 + b2*x + c2*x**2 + d2*x**3
            p3 = a3 + b3*x + c3*x**2 + d3*x**3


            return torch.stack((p0, p1, p2, p3)).transpose(0,1).to(dev)
        
        def coeff_end(xt):
            # Brute force cubic interpolation for the last grid interval
            x  =x1[xt]
            x_0=x0[-4]
            x_1=x0[-3]
            x_2=x0[-2]
            x_3=x0[-1]

            a0 = x_1*x_2*x_3/((x_1-x_0)*(x_2-x_0)*(x_3-x_0))
            a1 = x_0*x_2*x_3/((x_0-x_1)*(x_1-x_2)*(x_1-x_3))
            a2 = x_0*x_1*x_3/((x_0-x_2)*(x_2-x_1)*(x_2-x_3))
            a3 = x_0*x_1*x_2/((x_0-x_3)*(x_3-x_1)*(x_3-x_2))

            b0 = (x_1*x_2+x_1*x_3+x_2*x_3)/((x_0-x_1)*(x_0-x_2)*(x_0-x_3))
            b1 = (x_0*x_2+x_0*x_3+x_2*x_3)/((x_1-x_0)*(x_1-x_2)*(x_1-x_3))
            b2 = (x_0*x_1+x_0*x_3+x_1*x_3)/((x_2-x_0)*(x_2-x_1)*(x_2-x_3))
            b3 = (x_0*x_1+x_0*x_2+x_1*x_2)/((x_3-x_0)*(x_3-x_1)*(x_3-x_2))    

            c0 = (x_1+x_2+x_3)/((x_1-x_0)*(x_2-x_0)*(x_3-x_0))
            c1 = (x_0+x_2+x_3)/((x_0-x_1)*(x_2-x_1)*(x_3-x_1))
            c2 = (x_0+x_1+x_3)/((x_0-x_2)*(x_1-x_2)*(x_3-x_2))
            c3 = (x_0+x_1+x_2)/((x_0-x_3)*(x_1-x_3)*(x_2-x_3))

            d0 = 1/((x_0-x_1)*(x_0-x_2)*(x_0-x_3))
            d1 = 1/((x_1-x_0)*(x_1-x_2)*(x_1-x_3))
            d2 = 1/((x_2-x_0)*(x_2-x_1)*(x_2-x_3))
            d3 = 1/((x_3-x_0)*(x_3-x_1)*(x_3-x_2))

            p0 = a0 + b0*x + c0*x**2 + d0*x**3
            p1 = a1 + b1*x + c1*x**2 + d1*x**3
            p2 = a2 + b2*x + c2*x**2 + d2*x**3
            p3 = a3 + b3*x + c3*x**2 + d3*x**3
            #populate
            return torch.stack((p0, p1, p2, p3)).transpose(0,1)

        def coeff_middle(xt, xd):
            # Initialize the output tensor with zeros
            result = torch.zeros(len(xt), 4, dtype=dtype).to(dev)

            # Get the target x and x values in the domain
            x = x1[xt]
            xd_mask = torch.logical_and(xd >= 0, xd < len(x0)-2)
            xL = torch.zeros_like(xd, dtype=dtype)
            xL[xd_mask] = x0[xd[xd_mask]]
            xR = torch.zeros_like(xd, dtype=dtype)
            xR[xd_mask] = x0[(xd + 1)[xd_mask]]
            xLm = torch.zeros_like(xd, dtype=dtype)
            xLm[xd_mask] = x0[(xd - 1)[xd_mask]]
            xRp = torch.zeros_like(xd, dtype=dtype)
            xRp[xd_mask] = x0[(xd + 2)[xd_mask]]

            # Compute weights where they are well-defined
            valid_mask = torch.logical_and(xd != 0, xd != len(x0) - 2)
            valid_indices = torch.nonzero(valid_mask).squeeze(1)

            result[valid_indices, 0] = (-(x - xL) / (xR - xLm) + ((x - xL) ** 2) / ((xR - xL) * (xR - xLm)) - ((x - xL) ** 2) * (x - xR) / ((xR - xL) ** 2) / (xR - xLm))[valid_indices]
            result[valid_indices, 1] =( 1 - ((x - xL) ** 2) / ((xR - xL) ** 2) - ((x - xL) ** 2) * (x - xR) / ((xR - xL) ** 2) / (xRp - xL) + 2 * ((x - xL) ** 2) * (x - xR) / ((xR - xL) ** 3))[valid_indices]
            result[valid_indices, 2] = ((x - xL) / (xR - xLm) + ((x - xL) ** 2) / ((xR - xL) ** 2) - ((x - xL) ** 2) / ((xR - xL) * (xR - xLm)) + ((x - xL) ** 2) * (x - xR) / ((xR - xL) ** 2) / (xR - xLm) -  2 * ((x - xL) ** 2) * (x - xR) / ((xR - xL) ** 3))[valid_indices]
            result[valid_indices, 3] = (((x - xL) ** 2) * (x - xR) / ((xR - xL) ** 2) / (xRp - xL))[valid_indices]

            return result

        t_index = torch.arange(len(x1)).to(dev)
        d_index = torch.searchsorted(x0, x1, right=True)-1
        d_index = torch.clamp(d_index,0,len(x0)-2)

        c_i = coeff_start(t_index)
        c_f = coeff_end(t_index)
        c_m = coeff_middle(t_index,d_index)

        condition_i = (d_index == 0)
        condition_m = ((d_index > 0) & (d_index < len(x0) - 2))
        condition_f = (d_index == (len(x0) - 2))
        d_slice = torch.tensor([[i, i+1, i+2, i+3] for i in d_index[condition_i]], dtype=dtype)         

        for p in range(4):
            L[condition_i, d_index[condition_i]+p] = c_i[condition_i,p]
            L[condition_m, d_index[condition_m]-1+p] = c_m[condition_m,p]
            L[condition_f, d_index[condition_f]-2+p] = c_f[condition_f,p]

    L = L.reshape(*x1_shape,x0.shape[0])    

    return L




