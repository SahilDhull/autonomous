##Remaining tasks: -ve value of velocity, maxlimit of velocity, 0-acceleration
import numpy as np
import math
import copy
import dubins
import shapely.geometry as geom
import threading
from statistics import median 


#Change radius of curvature for 0.9
from vel_acc_to_throttle import *

inf = 1e9
No_of_threads = 11

acc= [-10.0,-5.0, -3.0, -1.0, 0.01, 1.0, 2.0]

total_distance = 150.0
grid_points = []

actual_vel = {} #key = (i,j,v,t)
actual_tim = {} #key = (i,j,v,t)
prev_acc = {} #key = (i,j,v,t)
c = {} #key = (j,v,t)
p = {} #key = (i,j,v,t)
velocities = []
times= []


#Used for updation across different layers
temp_tim = {}
temp_vel = {}
temp_acc = {}
temp_c = {}
temp_p = {}
#-----------------------------------------


y_step = 0.9
x_step = 5.0
w = 3.6
obs_initial_pos = [450.0,0.0]
obs_vel = 5.0
corner_local_coords = [[-1.1, 2.5], [1.1, 2.5], [-1.1, -2.5], [1.1, -2.5]]
Radius_of_road = 20.0
    
def RadiusofCurvature(start_pt, end_pt, turn_radius=20.0, step_size=1.0):
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


def rotate_point_ccw(point, theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), point)

def check_colliding(pt2):
    obstacle_position = [obs_initial_pos[0] - obs_vel*pt2[3],obs_initial_pos[1]]
    car_corner_pos = []
    for local_coord in corner_local_coords:
        car_corner_pos.append([pt2[0][0]+local_coord[1],pt2[0][1]+local_coord[0]])


    obs_corner_pos = []
    for local_coord in corner_local_coords:
        # rotated_local_coord = \
        #     rotate_point_ccw(point=np.transpose(np.array(local_coord)),
        #                      rotation_angle=-detected_objects[obj_ind].object_yaw_angle)
        rotated_local_coord = \
            rotate_point_ccw(point=np.transpose(np.array(local_coord)),
                             theta=0.0)
        
        obs_corner_pos.append([obstacle_position[0] + rotated_local_coord[1],
                             obstacle_position[1] + rotated_local_coord[0]])


    collision = 0
    for pos in car_corner_pos:
        if (pos[0]>=obs_corner_pos[2][0] and pos[0]<=obs_corner_pos[0][0] and pos[1]<=obs_corner_pos[1][1] and pos[1]>=obs_corner_pos[0][1]): 
            collision=1
            break
    return collision    

def cost(c1, pt1,pt2, off=0.0):
    # print(pt1)
    # print(pt2)
    # r = RadiusofCurvature(pt1[0],pt2[0])
    R={}
    R[(5,0)] = inf
    # For straight line only

    deltay = abs(pt2[0][1]-pt1[0][1])
    deltax = abs(pt2[0][0]-pt1[0][0])
    temp = (deltax,deltay)
    if(temp in R):
        r = R[temp]
    else:
        r = RadiusofCurvature(pt1[0],pt2[0])
        if(r>=20):
            r=inf
        R[temp] = r

    static_cost =  c1 + math.sqrt((pt2[0][0]-pt1[0][0])**2 + (pt2[0][1]-pt1[0][1])**2) + 10.0/r + 20.0*abs(off)
    dynamic_cost = 15*(pt2[3]-pt1[3]) + (pt2[2]**2)*0.0 + 0.0*(pt2[1]**2) + 1.0*(((pt2[1]-pt1[1])/(pt2[3]-pt1[3]))**2) + 10*(((pt2[2])**2)/r)
    
    return static_cost + dynamic_cost + check_colliding(pt2)*inf


def Grid1(cur_pt,dist_to_cover):
    global grid_points
    x1 = round(cur_pt[0],2)
    x2 = max(x1-dist_to_cover,0) ##path to travel in first part of the road
    for i in np.arange(x1,x2,-x_step):
        gp = []
        for j in np.arange(w,-w,-y_step):
            gp.append([i,round(j,2),math.pi])
        grid_points.append(gp)
    return dist_to_cover - (x1-x2)

def Grid2(cur_pt,dist_to_cover):
    global grid_points
    theta_covered = math.atan(abs(cur_pt[0])/(Radius_of_road+cur_pt[1]))
    if(theta_covered<0):
        theta_covered = theta_covered + math.pi
    theta_to_cover = dist_to_cover/Radius_of_road
    final_theta = min(theta_covered + theta_to_cover,math.pi)
    for theta in np.arange(theta_covered,final_theta+0.00001,x_step/Radius_of_road):
        gp = []
        for j in np.arange(Radius_of_road+w,Radius_of_road-w,-y_step):
            x_coord = round(-j*math.sin(theta),2)
            y_coord = round(-Radius_of_road+j*math.cos(theta),2)
            gp.append([x_coord,y_coord,math.pi+theta])
        grid_points.append(gp)
    return (theta_covered + theta_to_cover - final_theta)*Radius_of_road 

def Grid3(cur_pt,dist_to_cover):
    global grid_points
    x1 = round(cur_pt[0],2)
    x2 = min(x1+dist_to_cover,500.0) ##path to travel in first part of the road
    for i in np.arange(x1,x2,x_step):
        gp = []
        for j in np.arange(-2*Radius_of_road + w,-2*Radius_of_road-w,-y_step):
            gp.append([i,round(j,2),0.0])
        grid_points.append(gp)
    return (dist_to_cover - (x2-x1))    


def Grid4(cur_pt,dist_to_cover):
    global grid_points
    theta_covered = math.atan(abs(cur_pt[0]-500.0)/(-Radius_of_road-cur_pt[1]))
    if(theta_covered<0):
        theta_covered = theta_covered + math.pi
    theta_to_cover = dist_to_cover/Radius_of_road
    final_theta = min(theta_covered + theta_to_cover,math.pi)
    for theta in np.arange(theta_covered,final_theta+0.0000001,x_step/Radius_of_road):
        gp = []
        for j in np.arange(Radius_of_road+w,Radius_of_road-w,-y_step):
            x_coord = round(500.0+j*math.sin(theta),2)
            y_coord = round(-Radius_of_road-j*math.cos(theta),2)
            gp.append([x_coord,y_coord,theta])
        grid_points.append(gp)
    return (theta_covered + theta_to_cover - final_theta)*Radius_of_road 


def calculate_grid(cur_pt,dist_to_cover):
    global grid_points
    grid_points = []
    if(cur_pt[0]>0 and cur_pt[0]<=500 and cur_pt[1]>-20.0):  ##check in first part of the road
        remaining_dist = Grid1(cur_pt,dist_to_cover)
        if(remaining_dist > 0):
            remaining_dist = Grid2([0.0,0.0],remaining_dist)
        if(remaining_dist > 0):
            remaining_dist = Grid3([0.0,-2*Radius_of_road],remaining_dist)
        if(remaining_dist > 0):
            remaining_dist = Grid4([500.0,-2*Radius_of_road],remaining_dist)
    elif(cur_pt[0]<=0):
        remaining_dist = Grid2(cur_pt,dist_to_cover)
        if(remaining_dist>0):
            remaining_dist = Grid3([0.0,-2*Radius_of_road],remaining_dist)
        if(remaining_dist > 0):
            remaining_dist = Grid4([500.0,-2*Radius_of_road],remaining_dist)
    elif(cur_pt[0]>=0 and cur_pt[0]<500 and cur_pt[1]<-20.0):
        remaining_dist = Grid3(cur_pt,dist_to_cover)
        if(remaining_dist > 0):
            remaining_dist = Grid4([500.0,-2*Radius_of_road],remaining_dist)
        if(remaining_dist > 0):
            remaining_dist = Grid1([500.0,0.0],remaining_dist)
    else:
        remaining_dist = Grid4([500.0,-2*Radius_of_road],dist_to_cover)    
        if(remaining_dist > 0):
            remaining_dist = Grid1([500.0,0.0],remaining_dist)



def computeTargetPath(cur_pt, dist_to_cover):
    
    calculate_grid(cur_pt,dist_to_cover)
    global grid_points
    global c
    global p
    global actual_vel
    global actual_tim
    global prev_acc
       
    # print(grid_points)

    ##########change from here
    X = round(2*w/y_step)
    Y = len(grid_points)
    


    ind2 = -1
    min_dist = inf
    for j in range(X):
        cur_dist = (grid_points[0][j][1]-cur_pt[1])**2 + (grid_points[0][j][0]-cur_pt[0])**2
        if(cur_dist < min_dist):
            min_dist = cur_dist
            ind2 = j

    #Initialisation
    i3 = math.ceil(cur_pt[3])
    i4 = math.ceil(cur_pt[4])
    c[(ind2,i3,i4)] = 0.0
    p[(0,ind2,i3,i4)] = -1
    actual_tim[(0,ind2,i3,i4)] = cur_pt[4]
    actual_vel[(0,ind2,i3,i4)] = cur_pt[3]
    prev_acc[(0,ind2,i3,i4)] = cur_pt[2]


    global velocities
    global times
    global temp_vel
    global temp_c
    global temp_tim
    global temp_p
    global temp_acc

    cf = inf
    final_pos = -1
    
    for i in  range(Y-1):
        t0= threading.Thread(target=parallel_func, args=(0,i,X,))
        t1= threading.Thread(target=parallel_func, args=(1,i,X,))
        t2= threading.Thread(target=parallel_func, args=(2,i,X,))
        t3= threading.Thread(target=parallel_func, args=(3,i,X,))
        t4= threading.Thread(target=parallel_func, args=(4,i,X,))
        t5= threading.Thread(target=parallel_func, args=(5,i,X,))
        t6= threading.Thread(target=parallel_func, args=(6,i,X,))
        t7= threading.Thread(target=parallel_func, args=(7,i,X,))
        t8= threading.Thread(target=parallel_func, args=(8,i,X,))
        t9= threading.Thread(target=parallel_func, args=(9,i,X,))
        t10= threading.Thread(target=parallel_func, args=(10,i,X,))
        t0.start()
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()
        t7.start()
        t8.start()
        t9.start()
        t10.start()
        t0.join()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        t6.join()
        t7.join()
        t8.join()
        t9.join()
        t10.join()

        # print(velocities)
        # print(" ")
        v_m = median(velocities)
        t_m = median(times)
        v_min = v_m-5
        v_max = v_m+5
        t_max = t_m+5
        t_min = t_m-5

        # print(c)
        c = {}
        for (j,v,t) in temp_c:
            ind_v = math.ceil(v)
            if(v > v_max):
                ind_v = inf
            if(v < v_min):
                ind_v = v_min
            ind_t = math.ceil(t)
            if(t > t_max):
                ind_t = inf
            if(t < t_min):
                ind_t = t_min
            
            if ((j,ind_v,ind_t) not in c) or (c[(j,ind_v,ind_t)] > temp_c[(j,v,t)] ):
                c[(j,ind_v,ind_t)] = temp_c[(j,v,t)]
                p[(i+1,j,ind_v,ind_t)] = temp_p[(i+1,j,v,t)]
                actual_vel[(i+1,j,ind_v,ind_t)] = temp_vel[(i+1,j,v,t)]
                actual_tim[(i+1,j,ind_v,ind_t)] = temp_tim[(i+1,j,v,t)]
                prev_acc[(i+1,j,ind_v,ind_t)] = temp_acc[(i+1,j,v,t)]
                if(i==Y-2) and (cf>c[(j,ind_v,ind_t)]):
                    cf = c[(j,ind_v,ind_t)]
                    final_pos = (i+1,j,ind_v,ind_t)




        velocities = []
        times = []
        temp_c = {}
        temp_vel = {}
        temp_acc = {}
        temp_p = {}
        temp_tim = {}



    travel_path = []
    (i,j,ind2,ind3) = final_pos
    while ( (p[(i,j,ind2,ind3)]) != -1 ):
        travel_path = [[float(grid_points[i][j][0]),float(grid_points[i][j][1]),prev_acc[(i,j,ind2,ind3)],actual_vel[(i,j,ind2,ind3)],actual_tim[(i,j,ind2,ind3)]]] + travel_path
        (i,j,ind2,ind3) = (p[(i,j,ind2,ind3)])
    
    return travel_path


    

def parallel_func(ind4,i,X):
    global c
    global p
    global actual_vel
    global actual_tim
    global prev_acc


    global temp_c
    global temp_p
    global temp_acc
    global temp_vel
    global temp_tim

    global velocities
    global times
    while(ind4 < len(acc)):
        for (j,ind2,ind3) in c:
            m1 = max(0,j-2)
            m2 = min(X-1,j+2)
            for k in range(m1,m2+1):
                a_f = acc[ind4]
                cur_cost = 0
                v_f = ( (actual_vel[(i,j,ind2,ind3)]**2) +2*a_f*x_step)
                if(v_f < 0):
                    v_f =0
                    continue
                else:
                    v_f = v_f ** 0.5
                v_f = round(v_f,4)
                
                ind5 = math.ceil(v_f)
                if v_f == actual_vel[(i,j,ind2,ind3)]:
                    t_f = x_step/v_f + actual_tim[(i,j,ind2,ind3)]
                else: 
                    t_f = (v_f-actual_vel[(i,j,ind2,ind3)])/a_f + actual_tim[(i,j,ind2,ind3)]
                t_f = round(t_f,2)         
                ind6 = math.ceil(t_f)
                
                
                cur_cost = cost(c[(j,ind2,ind3)],(grid_points[i][j],prev_acc[(i,j,ind2,ind3)],actual_vel[(i,j,ind2,ind3)],actual_tim[(i,j,ind2,ind3)]),(grid_points[i+1][k],a_f,v_f,t_f),off=abs(w-k*y_step))
                if(cur_cost > inf):
                    continue
                
                if( (k,ind5,ind6) not in temp_c) or (temp_c[(k,ind5,ind6)] > cur_cost):
                    temp_tim[(i+1,k,ind5,ind6)] = t_f
                    temp_c[(k,ind5,ind6)] = cur_cost
                    temp_vel[(i+1,k,ind5,ind6)] = v_f
                    temp_acc[(i+1,k,ind5,ind6)] = a_f
                    temp_p[(i+1,k,ind5,ind6)] = (i,j,ind2,ind3)
                    velocities.append(v_f)
                    times.append(t_f)
                

        ind4+=No_of_threads





total_distance_covered = 0
# cur_pt = [16.77,0.0,0.5,34.45,26.0]
cur_pt =  [500.0, 0.0, 0.0, 0.0, 0.0]
# cur_pt = [80.0, 0.0, 2.0, 21.9088, 10.97]

path = [cur_pt]
while(total_distance_covered < 100):
    path = path + computeTargetPath(cur_pt,300)
    # print("path=====================")
    # print(path)
    total_distance_covered = 150 + total_distance_covered
    cur_pt = path[-1]
    actual_vel = {}
    actual_tim = {}
    prev_acc = {}
    c = {}
    p = {}
    # print(cur_pt)
    # print(path)



output = path
print(output)
print(" ")
target_path = []
# v = []
t = []
# a= []
throttle = []
prev = -1

for i in output:
    target_path.append([i[0],i[1]])
    # a.append(i[2])
    # v.append(i[3])
    t.append(i[4])
    if(prev == -1):
        prev = (i[3],i[2])
    else:
        throttle.append(throttle_value( (i[3]+prev[0])/2.0,(i[2]+prev[1])/2.0))
        prev = (i[3],i[2])
        
print(throttle)
print(" ")
print(target_path)
# print(" ")
# print(a)
# print(" ")
# print(v)
print(" ")
print(t)

