#CS 471 Assignment 1
import math # for euclidean distance calculation
circle_size = [] # to store the circle sizes

#collect how many circles the user wants to input
num_circles = int(input("Enter the number of circles you want to input: "))

# loop to get different circles sizes and location from the user
for i in range(num_circles):
    x,y, r = map(float, input("Enter x, y, and r separated by spaces: ").split())
    circle_size.append((x, y, r))

#Clusters need to all touch each other in some fashion 
#function to check if two circles touch or overlap
def circles_touch(c1, c2):
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    distance = math.sqrt(dx**2 + dy**2)

    radius_sum = c1[2] + c2[2]
    radius_diff = abs(c1[2] - c2[2])

    return radius_diff <= distance <= radius_sum # Checks if the circles touch or overlap

#Checks if all circles touch each other
def has_clusters(circles):
    connected = [False] * len(circles)
    
    for i in range(len(circles)):
        for j in range (i+1, len(circles)):
            if circles_touch(circles[i], circles[j]):
                connected[i] = True
                connected[j] = True
    return all(connected)  # Checks that all circles touch each other in some fashion

  

#Calls the function 
result = has_clusters(circle_size)
print("True. The circles touch each other, there is a cluster." if result else "False. The circles do not touch each other, there is not cluster.")









