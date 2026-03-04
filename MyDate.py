class MyDate:
    def __init__( self, year=2000, month=1, day=1):
        self.year = year
        self.month = month
        self.day = day
    def setDate( self, y, m, d):
        self.year = y
        self.month = m
        self.day = d
    def print(self):
        print(f'Date: {self.year}/{self.month:02d}/{self.day:02d}')
#
d1 = MyDate(2026,3,4)
#d1.y = 2026
#d1.m = 3
#d1.d = 4
d1.print()
#print(f'Date: {d1.year}/{d1.month:02d}/{d1.day:02d}')
#
d2 = MyDate()
print(f'Date: {d2.year}/{d2.month:02d}/{d2.day:02d}')
d2.setDate( 2000, 12, 30)
print(f'Date: {d2.year}/{d2.month:02d}/{d2.day:02d}')
'''
y = 2026
m = 3
d = 4

print(f'Date: {y}/{m:02d}/{d:02d}')
'''