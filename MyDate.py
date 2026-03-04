class MyDate:
    def __init__( self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

d1 = MyDate(2026,3,4)
#d1.y = 2026
#d1.m = 3
#d1.d = 4

print(f'Date: {d1.year}/{d1.month:02d}/{d1.day:02d}')

'''
y = 2026
m = 3
d = 4

print(f'Date: {y}/{m:02d}/{d:02d}')
'''