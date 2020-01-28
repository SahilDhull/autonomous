import numpy as np
import math
import copy
import dubins
import shapely.geometry as geom
import threading

inf = 1e9


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


def cost(c1, pt1,pt2, off=0.0):
    print(pt1)
    print(pt2)
    # r = RadiusofCurvature(pt1[0],pt2[0])
    R={}
    R[0] = inf
    R[0.4] = 14.58
    R[0.8] = 7.48
    R[1.2] = 5.08
    # For straight line only
    r = R[round(abs(pt2[0][1]-pt1[0][1]),1)]
    static_cost =  c1 + math.sqrt((pt2[0][0]-pt1[0][0])**2 + (pt2[0][1]-pt1[0][1])**2) + 10.0/r + 10.0*abs(off)
    dynamic_cost = 10*(pt2[3]-pt1[3]) + pt2[2]**2 + 0.1*(pt2[1]**2) + 0.1*(((pt2[1]-pt1[1])/(pt2[3]-pt1[3]))**2) + 0.1*(((pt2[2])**2)/r)
    return static_cost + dynamic_cost 

def computeTargetPath(cur_pt):
    grid_points = []
    x1 = round(cur_pt[0],2)
    w = 3.6
    x_step = 5.0
    y_step = 0.4
    r_curv = 20.0
    x_ctr1 = 0.0 #(x)center of first circle 
    y_ctr1 = -20.0#(y)center of first circle
    x_ctr2 = 500.0
    y_ctr2 = -20.0

    
    acc= [-2.5,-2.0,-1.5,-1.0,-0.5,0.01,0.5,1.0,1.5,2.0,2.5]
    vel = [v for v in range(10)]
    actual_vel = [v for v in range(10)]
    tim = [p/4.0 for p in range(10)]
    actual_tim = [p/4.0 for p in range(10)]

    if(x1>-1000.0 and cur_pt[1]> (-20.0) ):
        x2 = x1-30
        # 1st part
        y1 = 0.0
        for i in np.arange(x1,x2,-x_step):
            gp = []
            for j in np.arange(y1+w,y1-w,-y_step):
                gp.append([i,y1+round(j,2),math.pi])
            grid_points.append(gp)

        p = []
        c = []
        X = round(2*w/y_step)
        Y = len(grid_points)

        for j in range(Y):
            k = []
            f = []
            for i in range(X):
                k1 = []
                f1 = []
                for ind,a in enumerate(acc):
                    k2 = []
                    f2 = []
                    for v in vel:
                        k3 = []
                        f3 = []
                        for t in tim:
                            k3.append(inf)
                            f3.append((-1,-1,-1,-1,-1))
                        k2.append(k3)
                        f2.append(f3)
                    k1.append(k2)
                    f1.append(f2)
                k.append(k1)
                f.append(f1)
            c.append(k)
            p.append(f)

    # print(c[0][0][0][9][0])
    y1 = cur_pt[1]
    ind2 = 0
    if (y1 >= 0.0):
        ind2 = 9-round(y1/0.4,0)
    else:
        ind2 = 9+round((-y1)/0.4,0)
    ind2 = int(ind2)
    c[0][ind2][0][0][0] = 0.0

    
    final_pos = [-1,-1,-1,-1,-1]
    cf = inf
    for i in  range(Y-1):
        for j in range(X):
            for ind1,a in enumerate(acc):
                for ind2,v in enumerate(vel):
                    for ind3,t in enumerate(tim):
                        m1 = max(0,j-3)
                        m2 = min(X-1,j+3)
                        for k in range(m1,m2+1):
                            for ind4,a_f in enumerate(acc):
                                cur_cost = 0
                                v_f = ( (actual_vel[ind2]**2) +10*a_f)
                                if(v_f < 0):
                                    v_f =0
                                    continue
                                else:
                                    v_f = v_f ** 0.5
                                
                                ind5 = math.floor(v_f)
                                ind5 = min(ind5,9)
                                if v_f == actual_vel[ind2]:
                                    t_f = 5.0/v_f + actual_tim[ind3]
                                else: 
                                    t_f = (v_f-actual_vel[ind2])/a_f + actual_tim[ind3]
                                ind6 = math.floor(t_f*4)
                                ind6 = min(ind6,9)

                                # print (str(i)+" "+str(k)+" "+str(ind4)+" "+str(ind5)+" "+str(ind6))
                                cur_cost = cost(c[i][j][ind1][ind2][ind3],(grid_points[i][j],a,actual_vel[ind2],actual_tim[ind3]),(grid_points[i+1][k],a_f,v_f,t_f))
                                if(c[i+1][k][ind4][ind5][ind6] > cur_cost):
                                    c[i+1][k][ind4][ind5][ind6] = cur_cost
                                    actual_vel[ind5] = v_f
                                    actual_tim[ind6] = t_f
                                    p[i+1][k][ind4][ind5][ind6] = (i,j,ind1,ind2,ind3)
                                    if i==Y-2 and cf > cur_cost:
                                        cf =cur_cost
                                        final_pos = (i+1,k,ind4,ind5,ind6)

    print(final_pos)
    
    travel_path = []
    (i,j,ind1,ind2,ind3) = final_pos
    # # print(p[i][j][ind1][ind2][ind3])
    while ( (p[i][j][ind1][ind2][ind3]) !=(-1,-1,-1,-1,-1) ):
        travel_path = [[float(grid_points[i][j][0]),float(grid_points[i][j][1]),a[ind1],actual_vel[ind2],actual_tim[ind3]]] + travel_path
        (i,j,ind1,ind2,ind3) = (p[i][j][ind1][ind2][ind3])
    print(travel_path)

    
    # global suboptimalPath
    # suboptimalPath = travel_path

    # if(self.path_following_tools.target_path != None):
    #     cur_target_path = list(self.path_following_tools.target_path.coords)
    #     cur_path_details = self.path_following_tools.path_details
        
    #     self.path_following_tools.future_starting_point = (cur_target_path[-3][0],cur_target_path[-3][1])
        
    #     suboptimalPath = [[cur_target_path[-2][0],cur_target_path[-2][1]]] + [[cur_target_path[-1][0],cur_target_path[-1][1]]] + suboptimalPath
    #     for pt in suboptimalPath:
    #         self.path_following_tools.add_future_point_to_path(pt)

    #     self.path_following_tools.smoothen_the_future_path()
    #     self.path_following_tools.populate_the_future_path_with_details()
         
    #     cur_target_path = cur_target_path[:-3] + list(self.path_following_tools.future_target_path.coords)
    #     self.path_following_tools.future_target_path = geom.LineString(cur_target_path)
    #     cur_path_details = cur_path_details[:-3] + self.path_following_tools.future_path_details
    #     self.path_following_tools.future_path_details = cur_path_details
    
computeTargetPath([500.0,0.0])