import numpy as np
import math
import copy
import dubins
import shapely.geometry as geom

inf = 1e9


# def get_line_segment(target_path,segment_ind):
#     """Returns the indexed segment as list, as vector and its angle and length."""
#     line_segment = geom.LineString(self.target_path.coords[segment_ind:segment_ind + 2])
#     line_segment_as_list = target_path[segment_ind:segment_ind+2]
#     line_segment_as_vector = [line_segment_as_list[1][0] - line_segment_as_list[0][0],
#                               line_segment_as_list[1][1] - line_segment_as_list[0][1]]
#     segment_angle = math.atan2(line_segment_as_vector[0], line_segment_as_vector[1])
#     segment_length = math.sqrt(line_segment_as_vector[0]**2 + line_segment_as_vector[1]**2)
#     return line_segment_as_list, line_segment_as_vector, segment_angle, segment_length

def RadiusofCurvature(start_pt, end_pt, turn_radius=10.0, step_size=1.0):
    """Generate points along a Dubins path connecting start point to end point.
    Format for input / output points: (x, y, angle)"""
    min_turn_radius = min(0.1, turn_radius)
    satisfied = False
    configurations = [start_pt, end_pt]
    while not satisfied:
        dubins_path = dubins.shortest_path(start_pt, end_pt, turn_radius)
        configurations, _ = dubins_path.sample_many(step_size)
        cex_found = False
        for configuration in configurations:
            if not (min(start_pt[0], end_pt[0]) - 0.1 <= configuration[0] <= max(start_pt[0], end_pt[0]) + 0.1 and
                    min(start_pt[1], end_pt[1]) - 0.1 <= configuration[1] <= max(start_pt[1], end_pt[1]) + 0.1):
                cex_found = True
                break
        satisfied = not cex_found
        if cex_found:
            # Decrease radius until finding a satisfying result.
            # We could do a binary search but that requires a termination condition.
            turn_radius = turn_radius*0.9
            if turn_radius < min_turn_radius:
                break
    if not satisfied:
        return 0.1
    return turn_radius



# def convert_point_for_dubins_computation(point_coordinates, angle):
#     """Converts the format and angle to the format accepted by the dubins compuations."""
#     dub_angle = math.pi/2 - angle
#     return point_coordinates[0], point_coordinates[1], dub_angle

# def smoothen_the_path(target_path, turn_radius=10.0, step_size=1.0):
#     """Converts linear segments to Dubins path where necessary."""
#     c=0.0
#     if target_path is not None:
#         # First, connect turns with Dubins paths:
#         num_original_points = len(target_path)
#         for pt_ind in range(num_original_points - 3):
#             (init_line_segment_as_list, line_segment_as_vector, cur_angle, segment_length) = \
#                 get_line_segment(target_path,pt_ind)
#             (line_segment_as_list, line_segment_as_vector, next_angle, segment_length) = \
#                 get_line_segment(target_path,pt_ind + 1)
#             angle_diff = next_angle - cur_angle
#             if abs(angle_diff) > math.pi / 60.0:
#                 (line_segment_as_list, line_segment_as_vector, end_angle, segment_length) = \
#                     get_line_segment(target_path,pt_ind + 2)
#                 # If there is little angle difference or if the segment length is short,
#                 # don't bother creating a Dubins path for this segments.
#                 if (abs(next_angle - end_angle) > math.pi/60.0 and
#                         np.linalg.norm(np.array(line_segment_as_list[0]) -
#                                        np.array(init_line_segment_as_list[0])) > 3.0):
#                     start_pt = convert_point_for_dubins_computation(
#                         point_coordinates=init_line_segment_as_list[1], angle=cur_angle)
#                     end_pt = convert_point_for_dubins_computation(
#                         point_coordinates=line_segment_as_list[0], angle=end_angle)
#                     turn_radius = generate_dubins_path(start_pt, end_pt,
#                                                                              turn_radius=turn_radius,
#                                                                              step_size=step_size)
#                     c = c + 1.0/turn_radius
#     return c


def cost(c1, pt1,pt2):
	r = RadiusofCurvature(pt1,pt2)
	return c1 + math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2) + 10.0/r + 10.0*abs(pt2[0])


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
		gp.append([i,y1+round(j,2),math.pi/2.0])
	grid_points.append(gp)


# 2nd part
for i in range(st):
	gp = []
	theta = i*x_step/r_curv
	x_cur = x_ctr1 - r_curv*math.sin(theta)
	y_cur = y_ctr1 + r_curv*math.cos(theta)
	for j in np.arange(y1+w,y1-w,-y_step):
		gp.append([round(x_cur+j*math.sin(theta),2),round(y_cur-j*math.cos(theta),2),math.pi/2.0-theta])
	grid_points.append(gp)


# 3rd part
for i in np.arange(x2,x1,x_step):
	gp = []
	for j in np.arange(y1+w,y1-w,-y_step):
		gp.append([i,y2+round(j,2),-math.pi/2.0])
	grid_points.append(gp)

# 4th part
for i in range(st):
	gp = []
	theta = i*x_step/r_curv
	x_cur = x_ctr2 + r_curv*math.sin(theta)
	y_cur = y_ctr2 - r_curv*math.cos(theta)
	for j in np.arange(y1+w,y1-w,-y_step):
		gp.append([round(x_cur+j*math.sin(theta),2),round(y_cur-j*math.cos(theta),2),-math.pi/2.0 - theta])
	grid_points.append(gp)

#-----------Solve the circularity problem with theta------------------------
# print(grid_points[0][9])

travel_path = []
total_steps = 1000


p = []
c = []
X = round(2*w/y_step)
Y = len(grid_points)

for j in range(Y):
	k = []
	f = []
	for i in range(X):
		k.append(inf)
		f.append((-1,-1))
	c.append(k)
	p.append(f)

c[0][9] = 0.0


for i in  range(Y-1):
	for j in range(X):
		m1 = max(0,j-3)
		m2 = min(X-1,j+3)
		for k in range(m1,m2+1):
			cur_cost = 0;
			cur_cost = cost(c[i][j],grid_points[i][j],grid_points[i+1][k])
			if(c[i+1][k] > cur_cost):
				c[i+1][k] = cur_cost
				p[i+1][k] = (i,j)

i= Y-1
j = 9
while(p[i][j]!=(-1,-1)):
	travel_path = [[grid_points[i][j][0],grid_points[i][j][1]]] + travel_path
	(i,j) = p[i][j]

print (travel_path)