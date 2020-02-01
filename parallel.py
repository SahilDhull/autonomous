import numpy as np
import math
import copy
import dubins
import shapely.geometry as geom
import threading



inf = 1e9
No_of_threads = 11
acc= [-2.5,-2.0,-1.5,-1.0,-0.5,0.01,0.5,1.0,1.5,2.0,2.5]
vel = [v for v in range(10)]
tim = [p/4.0 for p in range(10)]


actual_vel = []


actual_tim = []


prev_acc = []
grid_points = []
    


c = []
p = []

obs_initial_pos = [450.0,0.0]
obs_vel = 5.0
corner_local_coords = [[-1.5, 2.0], [1.5, 2.0], [-1.5, -2.0], [1.5, -2.0]]
                
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
    R[0] = inf
    R[0.4] = 14.58
    R[0.8] = 7.48
    R[1.2] = 5.08
    # For straight line only
    r = R[round(abs(pt2[0][1]-pt1[0][1]),1)]
    static_cost =  c1 + math.sqrt((pt2[0][0]-pt1[0][0])**2 + (pt2[0][1]-pt1[0][1])**2) + 10.0/r + 10.0*abs(pt2[0][1])
    dynamic_cost = 10*(pt2[3]-pt1[3]) + (pt2[2]**2)*0.0 + 0.0*(pt2[1]**2) + 0.1*(((pt2[1]-pt1[1])/(pt2[3]-pt1[3]))**2) + 0.1*(((pt2[2])**2)/r)
    
    return static_cost + dynamic_cost + check_colliding(pt2)*inf

def computeTargetPath(cur_pt):
    x1 = round(cur_pt[0],2)
    w = 3.6
    x_step = 5.0
    y_step = 0.4
    r_curv = 20.0
    x_ctr1 = 0.0 #(x)center of first circle 
    y_ctr1 = -20.0#(y)center of first circle
    x_ctr2 = 500.0
    y_ctr2 = -20.0

    global grid_points
    global c
    global p
    global actual_vel
    global actual_tim
    global prev_acc
        
    if(x1>-1000.0 and cur_pt[1]> (-20.0) ):
        x2 = x1-150
        # 1st part
        y1 = 0.0
        for i in np.arange(x1,x2,-x_step):
            gp = []
            for j in np.arange(y1+w,y1-w,-y_step):
                gp.append([i,y1+round(j,2),math.pi])
            grid_points.append(gp)

        
        X = round(2*w/y_step)
        Y = len(grid_points)

        for j in range(Y):
            k1 = []
            f1 = []
            for i in range(X):
                k2 = []
                f2 = []
                for v in vel:
                    k3 = []
                    f3 = []
                    for t in tim:
                        k3.append(inf)
                        f3.append((-1,-1,-1,-1))
                    k2.append(k3)
                    f2.append(f3)
                k1.append(k2)
                f1.append(f2)
            c.append(k1)
            p.append(f1)


    for i in range(Y):
        v1 = []
        for j in range(X):
            v2 = []
            for ti in tim:
                v3 = [v for v in range(10)]
                v2.append(v3)
            v1.append(v2)
        actual_vel.append(v1)

    for i in range(Y):
        t1 = []
        for j in range(X):
            t2 = []
            for vi in vel:
                t3 = [p/4.0 for p in range(10)]
                t2.append(t3)
            t1.append(t2)
        actual_tim.append(t1)

    for i in range(Y):
        a1 =  []
        for j in range(X):
            a2 = []
            for vi in vel:
                a3 = []
                for ti in tim:
                    a3.append(0.0)
                a2.append(a3)
            a1.append(a2)
        prev_acc.append(a1)

    # print(c[0][0][0][9][0])
    y1 = cur_pt[1]
    ind2 = 0
    if (y1 >= 0.0):
        ind2 = 9-round(y1/0.4,0)
    else:
        ind2 = 9+round((-y1)/0.4,0)
    ind2 = int(ind2)
    c[0][ind2][0][0] = 0.0

    
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

    i = Y-2
    cf = inf
    final_pos = (-1,-1,-1,-1)
    for j in range(X):
        for ind1,v in enumerate(vel):
            for ind2,t in enumerate(tim):
                if(cf>c[i+1][j][ind1][ind2]):
                    cf =c[i+1][j][ind1][ind2]
                    final_pos = (i+1,j,ind1,ind2)
            
    print(final_pos)
    
    travel_path = []
    (i,j,ind2,ind3) = final_pos
    # # print(p[i][j][ind1][ind2][ind3])
    while ( (p[i][j][ind2][ind3]) !=(-1,-1,-1,-1) ):
        travel_path = [[float(grid_points[i][j][0]),float(grid_points[i][j][1]),prev_acc[i][j][ind2][ind3],actual_vel[i][j][ind3][ind2],actual_tim[i][j][ind2][ind3]]] + travel_path
        (i,j,ind2,ind3) = (p[i][j][ind2][ind3])
    print(travel_path)


    

def parallel_func(ind4,i,X):
    global c
    global p
    global actual_vel
    global actual_tim
    global prev_acc


    while(ind4 < len(acc)):
        for j in range(X):
            for ind2,v in enumerate(vel):
                for ind3,t in enumerate(tim):
                    m1 = max(0,j-3)
                    m2 = min(X-1,j+3)
                    for k in range(m1,m2+1):
                        a_f = acc[ind4]
                        cur_cost = 0
                        v_f = ( (actual_vel[i][j][ind3][ind2]**2) +10*a_f)
                        if(v_f < 0):
                            v_f =0
                            continue
                        else:
                            v_f = v_f ** 0.5
                            
                        ind5 = math.floor(v_f)
                        ind5 = min(ind5,9)
                        if v_f == actual_vel[i][j][ind3][ind2]:
                            t_f = 5.0/v_f + actual_tim[i][j][ind2][ind3]
                        else: 
                            t_f = (v_f-actual_vel[i][j][ind3][ind2])/a_f + actual_tim[i][j][ind2][ind3]
                        ind6 = math.floor(t_f*4)
                        ind6 = min(ind6,9)

                        # print (str(i)+" "+str(k)+" "+str(ind4)+" "+str(ind5)+" "+str(ind6))
                        cur_cost = cost(c[i][j][ind2][ind3],(grid_points[i][j],prev_acc[i][j][ind2][ind3],actual_vel[i][j][ind3][ind2],actual_tim[i][j][ind2][ind3]),(grid_points[i+1][k],a_f,v_f,t_f))
                        if(c[i+1][k][ind5][ind6] > cur_cost):
                            c[i+1][k][ind5][ind6] = cur_cost
                            prev_acc[i+1][k][ind5][ind6] = a_f
                            actual_vel[i+1][k][ind6][ind5] = v_f
                            actual_tim[i+1][k][ind5][ind6] = t_f
                            p[i+1][k][ind5][ind6] = (i,j,ind2,ind3)
                            # if i==Y-2 and cf > cur_cost:
                            #     cf =cur_cost
                            #     final_pos = (i+1,k,ind5,ind6)

        ind4+=No_of_threads



computeTargetPath([500.0,0.0])
