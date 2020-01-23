"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import os

#Rename webots dataset files
extension = '.png'
base_id = 7482
increment_id = 0
for i in range(1, 7481):
    if i%3 != 0:
        new_file_id = int(base_id + increment_id)
        # rename from cur_file_id to new_file_id
        cur_file_name = '{0:06d}'.format(int(i))
        cur_file_name = cur_file_name + extension
        new_file_name = '{0:06d}'.format(int(new_file_id))
        new_file_name = new_file_name + extension
        os.rename(cur_file_name, new_file_name)
        increment_id += 3
