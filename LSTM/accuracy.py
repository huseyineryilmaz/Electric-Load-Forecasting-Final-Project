import numpy as np


data = np.loadtxt("2005-2006_Predictions.txt")


counter = 0
counter2 = 0
counter3 = 0
counter4 = 0
counter5 = 0
for i in range(len(data)):
	if(data[i][0]-data[i][1] >= 1000 or data[i][1]-data[i][0] >= 1000):
		counter = counter + 1
	if(data[i][0]-data[i][1] >= 750 or data[i][1]-data[i][0] >= 750):
		counter2 = counter2 + 1
	if(data[i][0]-data[i][1] >= 500 or data[i][1]-data[i][0] >= 500):
		counter3 = counter3 + 1
	if(data[i][0]-data[i][1] >= 250 or data[i][1]-data[i][0] >= 250):
		counter4 = counter4 + 1
	if(data[i][0]-data[i][1] >= 100 or data[i][1]-data[i][0] >= 100):
		counter5 = counter5 + 1

accuracy = 1-counter/len(data)
accuracy2 = 1-counter2/len(data)
accuracy3 = 1-counter3/len(data)
accuracy4 = 1-counter4/len(data)
accuracy5 = 1-counter5/len(data)

print("Accuracy for threshold 1000: ",accuracy)
print("Accuracy for threshold 750: ",accuracy2)
print("Accuracy for threshold 500: ",accuracy3)
print("Accuracy for threshold 250: ",accuracy4)
print("Accuracy for threshold 100: ",accuracy5)

