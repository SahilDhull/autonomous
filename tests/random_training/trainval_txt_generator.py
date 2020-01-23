f = open('train_val.txt', 'w')
for i in range(0, 22441):
    f.write('{0:06d}\n'.format(i))
f.close()
