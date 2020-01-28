import numpy as np
import math
import copy
import dubins
import shapely.geometry as geom

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
    r = RadiusofCurvature(pt1,pt2)
    return c1 + math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2) + 10.0/r + 10.0*abs(off)


def computeTargetPath():
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
    st = math.floor((math.pi*r_curv)/x_step)        # steps to take in the curved part

    # 1st part
    for i in np.arange(x1,x2,-x_step):
        gp = []
        for j in np.arange(y1+w,y1-w,-y_step):
            gp.append([i,y1+round(j,2),math.pi])
        grid_points.append(gp)


    # 2nd part
    for i in range(st):
        gp = []
        theta = i*x_step/r_curv
        x_cur = x_ctr1 - r_curv*math.sin(theta)
        y_cur = y_ctr1 + r_curv*math.cos(theta)
        for j in np.arange(y1+w,y1-w,-y_step):
            gp.append([round(x_cur+j*math.sin(theta),2),round(y_cur-j*math.cos(theta),2),math.pi+theta])
        grid_points.append(gp)


    # 3rd part
    for i in np.arange(x2,x1,x_step):
        gp = []
        for j in np.arange(y1+w,y1-w,-y_step):
            gp.append([i,y2+round(j,2),0.0])
        grid_points.append(gp)

    # 4th part
    for i in range(st):
        gp = []
        theta = i*x_step/r_curv
        x_cur = x_ctr2 + r_curv*math.sin(theta)
        y_cur = y_ctr2 - r_curv*math.cos(theta)
        for j in np.arange(y1+w,y1-w,-y_step):
            gp.append([round(x_cur+j*math.sin(theta),2),round(y_cur-j*math.cos(theta),2),theta])
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
                cur_cost = cost(c[i][j],grid_points[i][j],grid_points[i+1][k],abs(k-9)*0.4)
                if(c[i+1][k] > cur_cost):
                    c[i+1][k] = cur_cost
                    p[i+1][k] = (i,j)

    i= Y-1
    j = 9
    # print(type(grid_points[0][0][0]))
    
    while(p[i][j]!=(-1,-1)):
        travel_path = [[float(grid_points[i][j][0]),float(grid_points[i][j][1])]] + travel_path
        (i,j) = p[i][j]

    print(travel_path)

computeTargetPath()