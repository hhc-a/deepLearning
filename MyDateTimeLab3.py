
from MyDate import MyDate
#
class MyDateTime(MyDate):
    def __init__(self, y, m, d, h=0, mi=0, s=0):
        super().__init__(y,m,d)
        self.h = h
        self.mi = mi
        self.s = s
    #
    def print(self):
        super().print()
        print(f'{self.h:02d}:{self.mi:02d}:{self.s:02d}')

dt1 = MyDateTime(2026,3,4,11,58,30)
dt1.print()