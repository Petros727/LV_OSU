file = open('SMSSpamCollection.txt')

spamCounter = 0
hamCounter = 0
exclamationSpamMark = 0
ham_words_counter = 0
spam_words_counter = 0

for line in file:
    if(line.startswith("ham")):
        hamCounter += 1
        ham_words_counter += len(line.split()[1::])
    if(line.startswith("spam")):
        spamCounter += 1
        spam_words_counter += len(line.split()[1::])
    if(line.endswith("!")):
        exclamationSpamMark += 1
        
           
print (float(ham_words_counter/hamCounter))
print(float(spam_words_counter/spamCounter))
print(exclamationSpamMark)