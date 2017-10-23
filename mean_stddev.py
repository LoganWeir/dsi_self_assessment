import math

def mean(value_array):
  return sum(value_array)/len(value_array)


def stddev(value_array):
  var_array = []

  mean = sum(value_array)/len(value_array)

  for item in value_array:

    var_array.append((item - mean)**2)

  variance = sum(var_array)/(len(value_array) - 1)

  print "VARIANCE", variance

  return math.sqrt(variance)



# data = [101, 120, 154, 89, 97, 132, 126, 105, 94, 111, 98, 90, 88, 115, 99, 85, 131, 127, 116]

data = [20, 45, 68, 900, 57, 45, 33, 35, 45, 22]

print len(data)


print stddev(data)


sorted_data = sorted(data)

lower_quart = ((sorted_data[1] + sorted_data[2])/2)

upper_quart = ((sorted_data[7] + sorted_data[8])/2)

print "IQR: ", (upper_quart - lower_quart)

# bald_eagle = [7.4, 7.7, 6.0, 6.7, 8.3, 6.5, 6.9, 7.7, 7.8, 7.3, 6.9, 6.5, 6.3, 
# 4.8, 8.0, 6.8, 5.8, 6.9, 6.3, 6.3, 6.4, 5.1, 6.9, 7.6, 5.6, 6.5, 6.7, 7.8, 6.6, 
# 6.9, 7.0, 6.4, 7.4, 6.0, 7.0, 5.3, 5.8, 6.4, 7.1, 5.5, 7.0, 6.7, 5.8, 6.1, 7.1, 
# 7.9, 7.7, 6.2, 5.3, 6.4, 6.9, 5.9, 7.8, 5.6, 5.0, 5.5, 6.4, 7.1, 8.6, 9.3, 6.8, 
# 7.6, 7.2, 7.1, 5.8, 5.9, 5.1, 6.6, 6.8, 5.7, 6.3, 7.3, 6.3, 7.2, 7.7, 6.0, 7.2, 
# 5.9, 7.2, 7.0, 7.4, 6.5, 7.8, 5.9, 6.3, 6.3, 8.3, 5.9, 6.9, 7.8]

# crowned_eagle = [5.3, 5.6, 5.8, 5.3, 5.6, 4.9, 5.7, 5.4, 5.8, 5.4, 6.0, 5.4, 
# 5.1, 5.4, 5.2, 5.7, 4.8, 5.8, 5.7, 5.1, 5.3, 5.4, 5.7, 6.6, 5.0, 5.4, 5.3, 5.5, 
# 5.2, 5.6, 5.2, 5.9, 5.7, 5.8, 5.5, 5.2, 4.0, 5.8, 5.2, 6.2, 5.4, 4.6, 5.3, 5.8, 
# 6.3, 4.8, 5.6, 5.4, 5.2, 5.4, 5.1, 6.0, 6.1, 5.4, 5.4, 5.3, 5.0, 6.0, 5.0, 5.8, 
# 5.1, 5.3, 4.8, 5.6, 5.7, 6.1, 5.0, 6.4, 5.1, 4.6, 5.3, 6.0, 4.8, 5.4, 4.3, 5.4, 
# 5.1, 4.7, 6.0, 5.5, 5.4, 5.6, 5.2, 5.8, 5.3, 4.9, 5.3, 5.5, 5.7, 4.7, 6.0, 5.6, 
# 4.9, 5.4, 4.3, 5.5, 4.9, 5.3, 5.6, 6.0]


# print len(bald_eagle)
# print len(crowned_eagle)


# # print "Mean: ", mean(data)

# print "Mean, Bald Eagle", mean(bald_eagle)
# print "Standard Deviation, Bald Eagle", stddev(bald_eagle)

# print "Mean, Crowned Eagle", mean(crowned_eagle)
# print "Standard Deviation, Crowned Eagle", stddev(crowned_eagle)




