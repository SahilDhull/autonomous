import dubins
import numpy as np
import math

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



print(RadiusofCurvature([5.0,0.0,math.pi],[0.0,-0.4,math.pi]))