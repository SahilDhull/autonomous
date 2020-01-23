import numpy as np
import math

grid_points = []

x1 = 500.0
x2 = 0

y1 = 0.0
y2 = -40.0

w = 3.6

x_step = 5.0
y_step = 0.4

r_curv = 20.0
x_ctr1 = 0.0
y_ctr1 = -20.0
x_ctr2 = 500.0
y_ctr2 = -20.0
st = math.floor((math.pi*r_curv)/x_step)		# steps to take in the curved part

# 1st part
for i in np.arange(x1,x2,-x_step):
	gp = []
	for j in np.arange(y1+w,y1-w,-y_step):
		gp.append([i,y1+round(j,2)])
	grid_points.append(gp)


# 2nd part
for i in range(st):
	gp = []
	theta = i*x_step/r_curv
	x_cur = x_ctr1 - r_curv*math.sin(theta)
	y_cur = y_ctr1 + r_curv*math.cos(theta)
	for j in np.arange(y1+w,y1-w,-y_step):
		gp.append([x_cur+j*math.sin(theta),y_cur-j*math.cos(theta)])
	grid_points.append(gp)


# 3rd part
for i in np.arange(x2,x1,x_step):
	gp = []
	for j in np.arange(y1+w,y1-w,-y_step):
		gp.append([i,y2+round(j,2)])
	grid_points.append(gp)

# 4th part
for i in range(st):
	gp = []
	theta = i*x_step/r_curv
	x_cur = x_ctr2 + r_curv*math.sin(theta)
	y_cur = y_ctr2 - r_curv*math.cos(theta)
	for j in np.arange(y1+w,y1-w,-y_step):
		gp.append([x_cur+j*math.sin(theta),y_cur-j*math.cos(theta)])
	grid_points.append(gp)

# print(grid_points[0][9])

travel_path = []
total_steps = 1000

for i in range(total_steps):
	travel_path.append(grid_points[i%len(grid_points)][9])

# print (travel_path)