"""Defines SupervisorControls class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import math
import numpy as np
from controller import Supervisor, Robot, Node, Field
# Above imported "controller" package is from Webots.


class SupervisorControls(Supervisor):
    """SupervisorControls class inherits the Supervisor class defined in
    Webots Python controller interface,
    and carries direct supervisor actions for controlling the simulation."""
    DEBUG_MODE = 1
    # If we add _init_() function, Webots will give error.

    def init(self, params):
        self.roots_children = self.getRoot().getField("children")

    def set_default_time_step(self, time_step):
        """Sets the default (basic) time step for the world.
        WorldInfo must have the DEF name WORLDINFO."""
        newtimestep = 0
        worldinfo = self.getFromDef('WORLDINFO')
        if worldinfo is not None:
            timestepfield = worldinfo.getField('basicTimeStep')
            timestepfield.setSFFloat(time_step)
            newtimestep = timestepfield.getSFFloat()
        return newtimestep


    def initialize_creating_simulation_environment(self):
        """Initialize the simulation environment creation process.
        (Just sets the simulation mode to PAUSE)"""
        self.set_simulation_mode(self.SIMULATION_MODE_PAUSE)

    def pause_simulation(self):
        """Pauses the simulation."""
        self.set_simulation_mode(self.SIMULATION_MODE_PAUSE)

    def finalize_creating_simulation_environment(self):
        """Finalizes the simulation environment creation process.
        (Just steps the simulation)"""
        self.step_simulation(0)

    def enable_receiver(self, receiver_name, period_ms):
        """Enables the receiver."""
        receiver = self.getReceiver(receiver_name)
        if receiver is not None:
            receiver.enable(period_ms)

    def get_receiver(self, receiver_name):
        """Returns the receiver object."""
        receiver = self.getReceiver(receiver_name)
        return receiver

    def enable_emitter(self, emitter_name, period_ms):
        """Enables the given emitter."""
        emitter = self.getEmitter(emitter_name)
        if emitter is not None:
            emitter.enable(period_ms)

    def get_emitter(self, emitter_name):
        """Returns the emitter object."""
        emitter = self.getEmitter(emitter_name)
        return emitter

    def get_obj_node(self, obj):
        """Returns the object node using the def_name field of the object"""
        return self.getFromDef(obj.def_name)

    def get_obj_from_def(self, def_name):
        """Returns the object node corresponding to the def_name"""
        return self.getFromDef(def_name)

    def get_obj_field(self, obj, field_name):
        """Returns the requested field."""
        return obj.node.getField(field_name)

    def add_obj_to_sim_from_string(self, object_string):
        """Adds the given string as an object to the simulation."""
        self.roots_children.importMFNodeFromString(-1, object_string)
        self.step(0)

    def step_simulation(self, simulation_step_size_ms):
        """Steps the simulation."""
        self.step(simulation_step_size_ms)

    def set_simulation_mode(self, mode):
        """Sets the simulation mode."""
        self.simulationSetMode(mode)

    def set_label(self, label_id, label, xpos, ypos, size, color, transparency, font):
        self.setLabel(label_id, label, xpos, ypos, size, color, transparency, font)

    def start_movie_recording(self, file_name, width, height, codec, quality, acceleration, caption):
        self.movieStartRecording(file_name, width, height, codec, quality, acceleration, caption)

    def stop_movie_recording(self):
        self.movieStopRecording()

    def wait_until_movie_is_ready(self):
        return self.movieIsReady()

    def revert_simulation(self):
        """Reverts the simulation. Before reloading the world,
        we will first change simulation state to "RUN"
        so that supervisor starts automatically again after revert."""
        self.set_simulation_mode(self.SIMULATION_MODE_FAST)
        self.step_simulation(0)
        self.simulationRevert()

    def get_time(self):
        """Get Simulation Time"""
        return self.getTime()

    def set_obj_position_3D(self, obj, position):
        """Sets the position of the object in 3D."""
        obj.translation.setSFVec3f(position)

    def get_obj_position_3D(self, obj):
        """Gets the position of the object in 3D."""
        pos = obj.translation.getSFVec3f()
        return pos

    def set_obj_velocity(self, obj, velocity):
        """Sets the velocity of the object."""
        obj.node.setVelocity(velocity)

    def get_obj_velocity(self, obj):
        """Gets the current velocity of the object."""
        vel = obj.node.getVelocity()
        return vel

    def set_obj_rotation(self, obj, rotation):
        """Sets the rotation of the object."""
        obj.rotation.setSFRotation(rotation)

    def get_obj_rotation(self, obj):
        """Gets the rotation of the object."""
        return obj.rotation.getSFRotation()

    def get_node_orientation(self, node):
        """Gets the orientation of the node."""
        rot_matrix_3d = node.getOrientation()  # Columns are in order x,y,z
        return rot_matrix_3d

    def get_obj_orientation(self, obj, axis=None):
        """Gets the orientation of the object around the given axis.
        If no axis is given, return the 3d rotation matrix."""
        rot_matrix_3d = obj.node.getOrientation()  # Columns are in order x,y,z
        if axis not in ['x', 'y', 'z']:
            ret_val = rot_matrix_3d
        else:
            # See Webots forum for more details.
            if axis == 'x':
                ind1 = 5
                ind2 = 8
            elif axis == 'y':
                ind1 = 2
                ind2 = 8
            else:  # axis == 'z':
                ind1 = 1
                ind2 = 4
            angle = math.atan2(rot_matrix_3d[ind1], rot_matrix_3d[ind2])
            if axis == 'y':
                angle = -angle
            while angle > math.pi:
                angle -= 2.0*math.pi
            while angle < -math.pi:
                angle += 2.0*math.pi
            ret_val = angle
        return ret_val

    def get_obj_orientation_vector(self, obj, local_principle_vector):
        """Returns a vector which is pointing the direction of the object orientation.
        local_principle_vector is a unit vector which point the head of the object in its local coordinates."""
        rot_matrix_3d = np.array(obj.node.getOrientation())  # Columns are in order x,y,z
        rot_matrix_3d.shape = (3, 3)
        principle_vector = np.array(local_principle_vector)
        principle_vector.shape = (3, 1)
        return np.dot(rot_matrix_3d, principle_vector)

    def reset_obj_physics(self, obj):
        """Resets the physics for the given simulation object."""
        obj.node.resetPhysics()
