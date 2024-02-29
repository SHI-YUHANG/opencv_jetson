from re import I


i = 0.998
x = 0
def test(x):
    z =  i ** x
    c = (1 - z) * 100
    print("第" + str(x) + "包抽到的概率是:" + str(round(c,2)) + "%")

test(430)