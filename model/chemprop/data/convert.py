rd = open("./test.csv", "r")
wt = open("./test_.csv", "w")
x = rd.readline()
for x in rd.readlines():
    y = x.split(",")
    wt.write(y[0] + "\n")
