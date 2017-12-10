# Neural-Trans-Funcs

Neuro transfer functions

requires numpy

example usage:
```
from neurotrans import HardLims
p = [1,-5,3,-99,0]
hardlims = HardLims()
a = hardLims(p)
print(a)
>>> [1,-1,1,-1,1]

#to get derivative
b = hardlims.derivative(p)
print(b)

>>> [1,1,1,1,1]
```

support currently for:
- hardlims/derivative
- purelin /derivative
- hardlim / derivative
- logSig/ derivative


