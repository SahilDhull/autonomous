import os

#Rename webots dataset files
extension = '.png'
new_file_id = int(14960)
for i in range(1, 4987):
    cur_file_id = int(i*3)
    new_file_id += 1
    if new_file_id % 3 == 0:
        new_file_id += 1
    # rename from cur_file_id to new_file_id
    cur_file_name = '{0:06d}'.format(int(cur_file_id))
    cur_file_name = cur_file_name + extension
    new_file_name = '{0:06d}'.format(int(new_file_id))
    new_file_name = new_file_name + extension
    os.rename(cur_file_name, new_file_name)
