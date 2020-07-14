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


lock = threading.Lock()
inf = 1e9
No_of_threads = 11

acc= {}
acc[0] = [-1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
acc[1] = [-1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
acc[2] = [-1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
acc[3] = [-1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
acc[4] = [-1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
acc[5] = [-1.0, 0.0, 1.0, 2.0, 4.0]
acc[6] = [-1.0, 0.0, 1.0, 2.0, 4.0]
acc[7] = [-1.0, 0.0, 1.0, 2.0, 4.0]
acc[8] = [-1.0, 0.0, 1.0, 2.0, 4.0]
acc[9] =  [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]
acc[10] = [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]
acc[11] = [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]
acc[12] = [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]
acc[13] = [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]
acc[14] = [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]
acc[15] = [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]
acc[16] = [-5.0, -3.0, -1.0, 0.0, 1.5, 3.0]
acc[17] = [-5.0, -3.0, -1.0, 0.0, 1.5, 3.0]
acc[18] = [-5.0, -3.0, -1.0, 0.0, 1.5, 3.0]
acc[19] = [-5.0, -3.0, -1.0, 0.0, 1.5, 3.0]
acc[20] = [-5.0, -3.0, -1.0, 0.0, 1.5, 3.0]
acc[21] = [-5.0, -3.0, -1.0, 0.0, 1.5]
acc[22] = [-5.0, -3.0, -1.2, 0.0, 1.5]
acc[23] = [-5.0, -3.0, -1.2, 0.0, 1.0]
acc[24] = [-5.0, -3.0, -1.2, 0.0, 1.0]
acc[25] = [-5.0, -3.0, -1.2, 0.0, 1.0]
acc[26] = [-5.0, -3.0, -1.3, 0.0, 1.0]
acc[27] = [-5.0, -3.0, -1.4, 0.0, 1.0]
acc[28] = [-5.0, -3.0, -1.4, 0.0, 0.5]
acc[29] = [-5.0, -3.0, -1.4, 0.0, 0.5]
acc[30] = [-5.0, -3.0, -1.5, 0.0]


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

temp_theta = {}
cur_theta = {}
#-----------------------------------------


y_step = 0.4
x_step = 5.0
w = 3.6
obs_initial_pos = [-450.0,0.0]
obs_vel = 10.0

corner_local_coords = [[-2.5, 1.1], [-2.5, -1.1], [2.5, 1.1], [2.5, -1.1]]

Radius_of_road = 20.0
    
def RadiusofCurvature(start_pt, end_pt, turn_radius=30.0, step_size=1.0):
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
            turn_radius = turn_radius*0.8
            if turn_radius < min_turn_radius:
                break
    if not satisfied:
        return 0.1
    return turn_radius


def rotate_point_cw(point, theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.dot(np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]), point)

def ObsPosition(t):
    Total_time = (1000.0 + 2*math.pi*20.0)/obs_vel
    t = t - Total_time * int(t/Total_time)
    offset = t * obs_vel
    if( obs_initial_pos[0] - offset >=0):
        return [obs_initial_pos[0]-offset,obs_initial_pos[1], math.pi]
    elif( obs_initial_pos[0] - (offset - math.pi * Radius_of_road) >=0 ):
        turned_theta = (offset - obs_initial_pos[0])/Radius_of_road
        return [-Radius_of_road*math.sin(turned_theta), -Radius_of_road + Radius_of_road*math.cos(turned_theta), math.pi + turned_theta]
    elif( offset <= obs_initial_pos[0] + 500.0 + Radius_of_road*math.pi):
        return [500.0 -offset +obs_initial_pos[0] +Radius_of_road*math.pi, -2*Radius_of_road, 0.0]
    elif(offset <= 2*Radius_of_road*math.pi + obs_initial_pos[0]+500.0):
        turned_theta = (offset - Radius_of_road*math.pi - 500.0 - obs_initial_pos[0])/Radius_of_road
        return [500.0+Radius_of_road*math.sin(turned_theta),-Radius_of_road- Radius_of_road*math.cos(turned_theta), turned_theta]
    else:
        return [1000.0 - offset + 2*Radius_of_road*math.pi + obs_initial_pos[0], 0.0, math.pi]




def check_colliding(pt2):
    turned = pt2[0][2]-pt2[4]
    obstacle_position = [obs_initial_pos[0] - obs_vel*pt2[3],obs_initial_pos[1]]
    # obstacle_position =  ObsPosition(pt2[3])
    car_corner_pos = []
    for local_coord in corner_local_coords:
        rotated_local_coord = \
            rotate_point_cw(point=np.transpose(np.array(local_coord)),
                             theta=turned)
        
        car_corner_pos.append([pt2[0][0]+rotated_local_coord[0],pt2[0][1]+rotated_local_coord[1]])

    # print(car_corner_pos)

    obs_corner_pos = []
    for local_coord in corner_local_coords:
        # rotated_local_coord = \
        #     rotate_point_ccw(point=np.transpose(np.array(local_coord)),
        #                      rotation_angle=-detected_objects[obj_ind].object_yaw_angle)
        rotated_local_coord = \
            rotate_point_cw(point=np.transpose(np.array(local_coord)),
                             theta=0.0)
        
        obs_corner_pos.append([obstacle_position[0] + rotated_local_coord[0],
                             obstacle_position[1] + rotated_local_coord[1]])

    # print(obs_corner_pos)

    collision = 0
    for dx in np.arange(-pt2[2],pt2[2],4.9):
        for pos in car_corner_pos:
            x = pos[0] + dx
            if (x>=obs_corner_pos[0][0] and x<=obs_corner_pos[2][0] and pos[1]<=obs_corner_pos[0][1] and pos[1]>=obs_corner_pos[1][1]): 
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
        r = RadiusofCurvature([pt1[0][0],pt1[0][1],pt1[4]],[pt2[0][0],pt2[0][1],pt2[4]])
        if(r==30):
            r=inf
        R[temp] = r

    obstacle_position = [obs_initial_pos[0] - obs_vel*pt2[3],obs_initial_pos[1]]
    
    static_cost =  c1 + math.sqrt((pt2[0][0]-pt1[0][0])**2 + (pt2[0][1]-pt1[0][1])**2) + 10.0/r + 1.0*abs(off) + 0.1*math.exp(-0.1*math.sqrt((pt2[0][0]-obstacle_position[0])**2 + (pt2[0][1]-obstacle_position[1])**2))

    dynamic_cost = 15.0*(pt2[3]-pt1[3]) + (pt2[2]**2)*0.0 + 0.0*(pt2[1]**2) + 1.7e-10*(((pt2[1]-pt1[1])/(pt2[3]-pt1[3]))**2) + 1.0*(((pt2[2])**2)/r)
    
    return static_cost + dynamic_cost + check_colliding(pt2)*inf

    #off = 1 or 0.5
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
    global cur_theta
       
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
    cur_theta[(ind2,i3,i4)] = cur_pt[5]

    global velocities
    global times
    global temp_vel
    global temp_c
    global temp_tim
    global temp_p
    global temp_acc
    global temp_theta


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
                cur_theta[(j,ind_v,ind_t)] = temp_theta[(j,v,t)]
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
        temp_theta = {}



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
    global temp_theta

    global velocities
    global times
    global lock
                
    for (j,ind2,ind3) in c:

        v_i = math.ceil(actual_vel[(i,j,ind2,ind3)])
        if(ind4 < len(acc[v_i])):
            m1 = max(0,j)
            m2 = min(X-1,j)
            for k in range(m1,m2+1):
                a_f = acc[v_i][ind4]
                cur_cost = 0
                v_f = ( (actual_vel[(i,j,ind2,ind3)]**2) +2*a_f*x_step)
                if(v_f < 0):
                    continue
                else:
                    v_f = v_f ** 0.5
                if(v_f > 30):
                    continue
                v_f = round(v_f,4)
                
                ind5 = math.ceil(v_f)
                if v_f == actual_vel[(i,j,ind2,ind3)]:
                    t_f = x_step/v_f + actual_tim[(i,j,ind2,ind3)]
                else: 
                    t_f = (v_f-actual_vel[(i,j,ind2,ind3)])/a_f + actual_tim[(i,j,ind2,ind3)]
                t_f = round(t_f,2)         
                ind6 = math.ceil(t_f)
                
                x1 = grid_points[i][j][0]
                y1 = grid_points[i][j][1]
                x2 = grid_points[i+1][k][0]
                y2 = grid_points[i+1][k][1]
                if(x2 < x1):
                    if(y2 >= y1):
                        curtheta = math.pi + math.atan((y2-y1)/(x2-x1))
                    else:
                        curtheta = math.pi + math.atan((y2-y1)/(x2-x1))
                elif(x2 > x1):
                    if(y2 >= y1):
                        curtheta = math.atan((y2-y1)/(x2-x1))
                    else:
                        curtheta = 2*math.pi + math.atan((y2-y1)/(x2-x1))
                else:
                    if(y2>y1):
                        curtheta = math.pi/2.0
                    else:
                        curtheta = 1.5*math.pi



                # curtheta = grid_points[i+1][k][2] - math.atan((k-j)*y_step/x_step)
                
                cur_cost = cost(c[(j,ind2,ind3)],(grid_points[i][j],prev_acc[(i,j,ind2,ind3)],actual_vel[(i,j,ind2,ind3)],actual_tim[(i,j,ind2,ind3)], cur_theta[(j,ind2,ind3)]),(grid_points[i+1][k],a_f,v_f,t_f,curtheta),off=abs(w-k*y_step))
                if(cur_cost > inf):
                    continue
                velocities.append(v_f)
                times.append(t_f)
                lock.acquire(True)
                if( (k,ind5,ind6) not in temp_c) or (temp_c[(k,ind5,ind6)] > cur_cost):
                    temp_tim[(i+1,k,ind5,ind6)] = t_f
                    temp_c[(k,ind5,ind6)] = cur_cost
                    temp_vel[(i+1,k,ind5,ind6)] = v_f
                    temp_acc[(i+1,k,ind5,ind6)] = a_f
                    temp_p[(i+1,k,ind5,ind6)] = (i,j,ind2,ind3)
                    temp_theta[(k,ind5,ind6)] = curtheta
                lock.release()
                





total_distance_covered = 0
# cur_pt = [16.77,0.0,0.5,34.45,26.0, math.pi]
cur_pt =  [500.0, 0.0, 0.0, 0.0, 0.0, math.pi]
# cur_pt = [[405.0, 0.0, math.pi], 1.5, 16.583, 8.9, math.pi]
# c = check_colliding(cur_pt)
# print(c)


path = [cur_pt]
while(total_distance_covered < 50):
    path = path + computeTargetPath(cur_pt,1200)
    # print("path=====================")
    # print(path)
    total_distance_covered = 100 + total_distance_covered
    cur_pt = path[-1]
    actual_vel = {}
    actual_tim = {}
    prev_acc = {}
    c = {}
    p = {}
    # print(cur_pt)
    # print(path)



output = path
# output = [[500.0, 0.0, 0.0, 0.0, 0.0, 3.141592653589793], [495.0, 0.0, 4.0, 6.3246, 1.58], [490.0, 0.0, 4.0, 8.9443, 2.23], [485.0, 0.0, 4.0, 10.9545, 2.73], [480.0, 0.0, 4.0, 12.6492, 3.15], [475.0, 0.0, 4.0, 14.1422, 3.52], [470.0, 0.0, 4.0, 15.492, 3.86], [465.0, 0.0, 3.0, 16.4317, 4.17], [460.0, 0.0, 3.0, 17.3205, 4.47], [455.0, 0.0, 3.0, 18.1659, 4.75], [450.0, 0.0, 3.0, 18.9737, 5.02], [445.0, 0.0, 3.0, 19.7485, 5.28], [440.0, 0.0, 3.0, 20.494, 5.53], [435.0, 0.4, 1.5, 20.8568, 5.77], [430.0, 0.8, 1.5, 21.2133, 6.01], [425.0, 1.6, 1.5, 21.564, 6.24], [420.0, 2.4, 1.5, 21.909, 6.47], [415.0, 2.4, 1.5, 22.2487, 6.7], [410.0, 1.2, 1.0, 22.4723, 6.92], [405.0, 0.0, 1.0, 22.6937, 7.14], [400.0, 0.0, 1.0, 22.913, 7.36], [395.0, 0.0, 1.0, 23.1302, 7.58], [390.0, 0.0, 1.0, 23.3454, 7.8], [385.0, 0.0, 1.0, 23.5586, 8.01], [380.0, 0.0, 1.0, 23.7699, 8.22], [375.0, 0.0, 1.0, 23.9793, 8.43], [370.0, 0.0, 1.0, 24.1869, 8.64], [365.0, 0.0, 1.0, 24.3927, 8.85], [360.0, 0.0, 1.0, 24.5968, 9.05], [355.0, 0.0, 1.0, 24.7992, 9.25], [350.0, 0.0, 1.0, 25.0, 9.45], [345.0, 0.0, 1.0, 25.1992, 9.65], [340.0, 0.0, 1.0, 25.3968, 9.85], [335.0, 0.0, 1.0, 25.5929, 10.05], [330.0, 0.0, 1.0, 25.7875, 10.24], [325.0, 0.0, 1.0, 25.9807, 10.43], [320.0, 0.0, 1.0, 26.1724, 10.62], [315.0, 0.0, 1.0, 26.3627, 10.81], [310.0, 0.0, 1.0, 26.5517, 11.0], [305.0, 0.0, 1.0, 26.7393, 11.19], [300.0, 0.0, 1.0, 26.9256, 11.38], [295.0, 0.0, 1.0, 27.1107, 11.57], [290.0, 0.0, 0.5, 27.2028, 11.75], [285.0, 0.0, 0.5, 27.2945, 11.93], [280.0, 0.0, 0.5, 27.3859, 12.11], [275.0, 0.0, 0.5, 27.477, 12.29], [270.0, 0.0, 0.5, 27.5678, 12.47], [265.0, 0.0, 0.5, 27.6583, 12.65], [260.0, 0.0, 0.5, 27.7485, 12.83], [255.0, 0.0, 0.5, 27.8384, 13.01], [250.0, 0.0, 0.5, 27.9281, 13.19], [245.0, 0.0, 0.5, 28.0175, 13.37], [240.0, 0.0, 0.5, 28.1066, 13.55], [235.0, 0.0, 0.5, 28.1954, 13.73], [230.0, 0.0, 0.5, 28.2839, 13.91], [225.0, 0.0, 0.5, 28.3722, 14.09], [220.0, 0.0, 0.5, 28.4602, 14.27], [215.0, 0.0, 0.5, 28.5479, 14.45], [210.0, 0.0, 0.5, 28.6353, 14.62], [205.0, 0.0, 0.5, 28.7225, 14.79]]
# output = [[500.0, 0.0, 0.0, 0.0, 0.0, 3.141592653589793], [495.0, 0.0, 4.0, 6.3246, 1.58], [490.0, 0.0, 4.0, 8.9443, 2.23], [485.0, 0.0, 4.0, 10.9545, 2.73], [480.0, 0.0, 4.0, 12.6492, 3.15], [475.0, 0.0, -3.0, 11.4019, 3.57], [470.0, 0.0, 1.0, 11.8323, 4.0], [465.0, 0.0, 2.0, 12.6492, 4.41], [460.0, 0.0, -1.0, 12.2475, 4.81], [455.0, 0.0, 4.0, 13.7841, 5.19], [450.0, 0.0, -3.0, 12.6492, 5.57], [445.0, 0.0, -1.0, 12.2475, 5.97], [440.0, 0.0, -3.0, 10.9545, 6.4], [435.0, 0.4, -3.0, 9.4869, 6.89], [430.0, 0.8, -3.0, 7.746, 7.47], [425.0, 1.2, 0.0, 7.746, 8.12], [420.0, 1.6, 1.0, 8.3666, 8.74], [415.0, 2.0, 4.0, 10.4881, 9.27], [410.0, 2.4, 4.0, 12.2475, 9.71], [405.0, 2.4, 1.0, 12.6492, 10.11], [400.0, 2.4, -3.0, 11.4019, 10.53], [395.0, 2.4, 1.0, 11.8323, 10.96], [390.0, 2.4, -3.0, 10.4882, 11.41], [385.0, 2.4, -3.0, 8.9444, 11.92], [380.0, 2.0, -3.0, 7.0712, 12.54], [375.0, 1.6, 0.0, 7.0712, 13.25], [370.0, 1.2, 4.0, 9.4869, 13.85], [365.0, 0.8, 4.0, 11.4018, 14.33], [360.0, 0.4, 2.0, 12.2475, 14.75], [355.0, 0.0, 4.0, 13.7841, 15.13], [350.0, 0.0, 2.0, 14.4914, 15.48], [345.0, 0.0, 4.0, 15.8114, 15.81], [340.0, 0.0, 1.5, 16.2788, 16.12], [335.0, 0.0, 1.5, 16.7332, 16.42], [330.0, 0.0, 1.5, 17.1756, 16.71], [325.0, 0.0, 3.0, 18.0278, 16.99], [320.0, 0.0, 1.5, 18.4391, 17.26], [315.0, 0.0, -1.0, 18.1659, 17.53], [310.0, 0.0, 1.5, 18.5742, 17.8], [305.0, 0.0, 3.0, 19.3649, 18.06], [300.0, 0.0, 3.0, 20.1246, 18.31], [295.0, 0.0, 1.5, 20.4939, 18.56], [290.0, 0.0, 0.0, 20.4939, 18.8], [285.0, 0.0, 0.0, 20.4939, 19.04], [280.0, 0.0, 0.0, 20.4939, 19.28], [275.0, 0.0, 0.0, 20.4939, 19.52], [270.0, 0.0, 0.0, 20.4939, 19.76], [265.0, 0.0, 0.0, 20.4939, 20.0], [260.0, 0.0, 0.0, 20.4939, 20.24], [255.0, 0.0, 0.0, 20.4939, 20.48], [250.0, 0.0, 0.0, 20.4939, 20.72], [245.0, 0.0, 0.0, 20.4939, 20.96], [240.0, 0.0, 0.0, 20.4939, 21.2], [235.0, 0.0, 0.0, 20.4939, 21.44], [230.0, 0.0, 0.0, 20.4939, 21.68], [225.0, 0.0, 0.0, 20.4939, 21.92], [220.0, 0.0, 0.0, 20.4939, 22.16], [215.0, 0.0, 0.0, 20.4939, 22.4], [210.0, 0.0, 0.0, 20.4939, 22.64], [205.0, 0.0, 0.0, 20.4939, 22.88]]
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
    
    if(prev == -1):
        prev = (i[3],i[2])
    else:
        t.append(i[4])
        throttle.append(throttle_value( (i[3]+prev[0])/2.0,i[2]))
        prev = (i[3],i[2])
        
print(throttle)
print(" ")
print(target_path)
print(" ")
print(t)



# r= RadiusofCurvature([5.0,0.0,math.pi],[0.0,0.25,math.pi])
# print(r)