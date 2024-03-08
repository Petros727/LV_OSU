file = open('song (2).txt')
uniqueWords = {}
count= 0

for line in file:
    words = line.rstrip().split()
    for word in words:
        if word not in uniqueWords:
                uniqueWords[word] = 1
        else:
                uniqueWords[word] +=1     
print(uniqueWords)
for word,value in uniqueWords.items():
    if value == 1:
        count += 1
        print(word)
print(count)