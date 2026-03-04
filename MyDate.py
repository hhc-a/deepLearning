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