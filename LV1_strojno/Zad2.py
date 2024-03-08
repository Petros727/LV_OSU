try:
    number = float(input("Unesite broj u intervalu [0.0 1.0]: "))
    if number >= 0.0 and number <= 1.0:
        if number >= 0.9:
            print("Vaša ocjena je A")
        elif  number >= 0.8:
            print("Vaša ocjena je B")
        elif  number >= 0.7:
            print("Vaša ocjena je C")
        elif number >= 0.6:
            print("Vaša ocjena je D")
        elif number < 0.6:
            print("Vaša ocjena je F")         
    else:
        print("unjeli ste broj koji se ne nalazi u intervalu [0.0 1.0]")              
except:
    print("Niste unjeli broj pokušajte ponovo")