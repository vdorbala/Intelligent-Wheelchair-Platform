mean = 0.0508461651

stdev = 0.3333695309

file = open("values1.csv", "w+")

f = open("captured15_n.csv","r+")

lis = [line.split() for line in f]
imgarr = []
valarr = []

for i,x in enumerate(lis):
    # imgno,val1,val2,val3,val4,val5 = x[0].split(',')
    imgno, val1 = x[0].split(',')
    imgarr.append(imgno)
    valarr.append(val1)

for i in range(len(valarr)):
	value = float((float(valarr[i])-mean))/float(stdev)
	file.write("{}\n".format(value))x`