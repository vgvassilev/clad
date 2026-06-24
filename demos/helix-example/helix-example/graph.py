
import matplotlib.pyplot as plt
import sys

fig = plt.figure()
ax = plt.axes(projection="3d")

x, y, z = [], [], []

for line in sys.stdin:
    if line.strip() == "end":
        ax.plot3D(x, y, z)
        x.clear()
        y.clear()
        z.clear()
    else:
        a, b, c = [float(i) for i in line.split()]
        x.append(a)
        y.append(b)
        z.append(c)

ax.set_box_aspect([1,1,1])
plt.show()


