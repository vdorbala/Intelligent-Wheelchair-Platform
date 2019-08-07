import visdom
from visdom import server
import re
import csv

server.download_scripts(
        proxies={
            'http': "http://proxy.iiit.ac.in:8080",
            'https': "http://proxy.iiit.ac.in:8080",
        },
)


f = open("values.csv","r")

with f:

    reader = csv.reader(f)

    for row in reader:
        for e in row:
            if isinstance(int(e), str) == 1:
                print(e)

# m = re.search(r'Train+\d*',f)
# z = m.group(0)
# print (z)
# numbers = re.search(r"[-+]?\d*\.\d+|\d+\$Test", f)


# print (numbers)

# win = viz.line(
#     Y=np.array([1]),
#     X=np.array([2]),
#     opts=dict(
#         fillarea=False,
#         showlegend=False,
#         width=800,
#         height=800,
#         xlabel='Epoch',
#         ylabel='Test and train Loss',
#         title='Loss plot',
#         # marginleft=30,
#         # marginright=30,
#         # marginbottom=80,
#         # margintop=30,
#     ),
# )

# viz.line(
#     X=np.array([1,2]),
#     Y=np.array([3,2]),
#     win=win,
#     name='Trainloss',
#     update='insert'
# )