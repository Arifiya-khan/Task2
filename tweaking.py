import os
accuracy = os.system("cat /task3/accuracy.txt")
if accuracy < 95:
     print("Error")
else:
    print("Good going :)")
    exit()