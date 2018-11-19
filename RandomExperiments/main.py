# List Comprehension Examples

val01 = [1,2,3,4,5,6,7,8,9,10]
print(val01)
val01 = [(x+1) for x in val01]
print(val01)
# <oldList> = [ <operation for each element> for <element Variable> in <oldList>

celsius = [39.2, 36.5, 37.3, 37.8]
fahrenheit = [((float(9)/5)*x + 32) for x in celsius]
# <oldList> = [ <operation for each element> for <element Variable> in <oldList>
print(fahrenheit)


noprimes = [j for i in range(2, 8) for j in range(i*2, 100, i)]
# list = [<x1> for <x2> in <firstList> for <x1> in <secondList>]
print(noprimes)

# primes = [x for x in range(2, 8) if x not in noprimes]

val01 = range(0,5)
val02 = ("a", "b")
val01 = [print(number, letter) for number in val01 for letter in val02]
# for each value in val01 we iterate through val02 and execute the first operation (print)
'''
0 a
0 b
1 a
1 b
2 a
2 b
3 a
3 b
4 a
4 b
'''
