import ANFIS as anf
import membershipfunction as msf
import numpy as np

path_train = "/Users/egorslipchenko/Documents/LAB_6/TrainDataSet"
path_test = "/Users/egorslipchenko/Documents/LAB_6/TestDataSet"
epoch = 100

training_set = np.loadtxt(path_train, usecols=[0, 1, 2, 3])
X = training_set[:, :3]
Y = training_set[:, 3]

mf = [[['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
       ['gaussmf', {'mean': -4., 'sigma': 10.}], ['gaussmf', {'mean': -7., 'sigma': 7.}]],
      [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
       ['gaussmf', {'mean': -4., 'sigma': 10.}], ['gaussmf', {'mean': -7., 'sigma': 7.}]],
      [['gaussmf', {'mean': 1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}],
       ['gaussmf', {'mean': -2., 'sigma': 10.}], ['gaussmf', {'mean': -10.5, 'sigma': 5.}]]]

mfc = msf.MemFuncs(mf)
anfis = anf.ANFIS(X, Y, mfc)

print("\t\t\t\t\n\n\nBEFORE TRAIN")

test_set = np.loadtxt(path_train, usecols=[0, 1, 2])
test = test_set[:, :3]
res = anf.predict(anfis, test)

for r, y in zip(res,Y):
    print(f"value:{r} would be:{y}")


print("\n\n\nTrain...")
anfis.trainHybridJangOffLine(epoch)


print("\t\t\t\t\n\n\nAFTER TRAIN")
res = anf.predict(anfis, test)
for r, y in zip(res,Y):
    print(f"value:{r} would be:{y}")

print("\n\n\nSearch awaitable value:")

test_set = np.loadtxt(path_test, usecols=[0, 1, 2])
test = test_set[:, :3]
res = anf.predict(anfis, test)
for r in res:
    print(f"value:{r}")


anfis.plotErrors()

# value:[-16.13820867]
# value:[-12.90432]
# value:[0.24060622]