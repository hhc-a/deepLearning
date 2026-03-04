'''
n = int(input("Input a number: "))
#print(n)
def isPrimer(n):
    for i in range( 2, n, 1):
        #print(i)
        if n%i == 0:
            #print(f"{n} is not a primer.")
            return False
            #break;
    else:
        #print(f"{n} is not a primer.")
        return True
#
n = int(input("Input a number: "))
#print(n)
for i in range( 2, n, 1):
        print(f"{n} is not a primer.")
else:
    print(f"{n} is not a primer.")
#
'''
def isPrimer(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
for i in range(1, 101):
    if isPrimer(i):
        print(i, end=" ")