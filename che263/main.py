# Numpy (data import, manipulation, export)
import numpy as np
# Matplotlib (create trends)
import matplotlib.pyplot as plt

data_file = np.loadtxt("data_file.txt", delimiter=',')

time = data_file[ : , 0]
time = time - time[0]

sensors = data_file[ : , 1:5]

avg = np.mean(sensors, axis=1)

my_data = np.vstack((time, sensors.T, avg)).T
np.savetxt("my_data.csv", my_data, delimiter=',')
timeM=time/60
plt.plot(timeM, sensors[:,1], 'r.')
plt.plot(timeM, avg, 'b.')
plt.legend(["Sensor 2", "Average"])
plt.xlabel("Time (minutes)")
plt.ylabel("Value")
plt.savefig("plot.png")
plt.show()