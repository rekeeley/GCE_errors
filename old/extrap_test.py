from scipy.interpolate import interp1d
from scipy import arange, array, exp
import matplotlib.pyplot as plt

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

x = arange(-10,10)
y = x**2
f_i = interp1d(x, y)
f_x = extrap1d(f_i)

x2 = arange(-20,50)

print f_x([9,12])

print exp(-9/3.)

print exp(-12/3.)

print f_x([5])

plt.plot(x,y)
plt.plot(x2,f_x(x2))
plt.savefig('tests.png')
