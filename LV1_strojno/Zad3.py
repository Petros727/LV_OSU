numbers = []
while True:
        number = input("Unesite broj: ")
        if number == "Done":
            break
        try:
            numbers.append(float(number))
        except:
            print("niste unjeli broj") 
print(len(numbers))
print(sum(numbers)/len(numbers))
print(min(numbers))
print(max(numbers))
numbers.sort()
print(numbers)
   
