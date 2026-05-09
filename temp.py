import numpy as np

s_len =5
a = np.random.randint(0,16,size=4)
i = np.array([np.arange(s, s + s_len) for s in a])

print(a)
print(i)