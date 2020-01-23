c = []
p = []
inf = 1e9

def cost(c1, x1, x2, y):
	return c1 + abs(x2-x1) + 1


X = 19
Y = 11

for i in range(X):
	k = []
	f = []
	for j in range(Y):
		k.append(inf)
		f.append((-1,-1))
	c.append(k)
	p.append(f)

c[9][0] = 0.0

for j in range(1,Y):
	for i in range(X):
		m1 = max(0,i-3)
		m2 = min(X-1,i+3)
		_min = inf
		for k in range(m1,m2):
			_min = min(_min,cost(c[k][j-1],k,i,j))
			if _min<c[i][j]:
				c[i][j] = _min
				p[i][j] = (k,j-1)
		# c[i][j] = _min

# print(c[9][10])

static_path = [[-1000.0, 0.0]]
y_start = 300.0

temp = (9,10)
while(temp!=(-1,-1)):
	# print(temp)
	static_path = [[y_start - temp[1]*10, (9-temp[0])*0.4]] + static_path
	temp = p[temp[0]][temp[1]]

# static_path = [[400.0, 0.0]] + static_path

# print(static_path)