





usercount = 0
TP = 0
FN = 0
FP = 0
check = 0




T = [1, 5, 7, 9,  11]
P = [1, 0, 7, 6, 8]
for Titem in T:
    for j in range(5):
        if Titem == P[j]:
            TP = TP + 1
            check = 1
    if check == 0: FN = FN +1
    check = 0
check = 0
for Pitem in P:
    for j in range(5):
        if Pitem == T[j]:
            check = 1
    if check == 0: FP = FP +1
    check = 0

print(TP, FN, FP)
print('P:',(TP)/(TP+FP), '   R:',(TP)/(TP+FN))