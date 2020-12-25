#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[4]:


import numpy as np


# 2. Create a null vector of size 10 

# In[ ]:


a = np.zeros(10)
a


# 3. Create a vector with values ranging from 10 to 49

# In[ ]:


vector_10 = np.arange(10, 49)
vector_10


# 4. Find the shape of previous array in question 3

# In[ ]:


vector_10.shape


# 5. Print the type of the previous array in question 3

# In[ ]:


vector_10.dtype


# 6. Print the numpy version and the configuration
# 

# In[ ]:


# Version
np.__version__
# Configuration
np.show_config()


# 7. Print the dimension of the array in question 3
# 

# In[ ]:


vector_10.ndim


# 8. Create a boolean array with all the True values

# In[ ]:


boolean_var = [True]
boolean_arr = np.random.choice(boolean_var, size=5)
boolean_arr


# 9. Create a two dimensional array
# 
# 
# 

# In[ ]:


twod_arr = np.array([[1,2,3],[4,5,6]])
td_arr


# 10. Create a three dimensional array
# 
# 

# In[ ]:


threed_arr = np.array([[[1,2,3],[4,5,6]],[[99,35,16],[7,8,9]]])
threed_arr


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[ ]:


vector_to_rev = np.array([1,3,4,6])
rev_vec = np.flip(vector_to_rev)
rev_vec


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[ ]:


null_vec = np.zeros(10)
null_vec[5] =1
null_vec


# 13. Create a 3x3 identity matrix

# In[ ]:


id_max=np.eye(3)
id_mac


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[ ]:


arr = np.array([1, 2, 3, 4, 5], dtype="float32")
arr.dtype


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[ ]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 

arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
np.multiply(arr1,arr2)


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[ ]:





# 17. Extract all odd numbers from arr with values(0-9)

# In[ ]:


total_array=np.arange(9)
p = 2
extr_arr = (total_array[total_array%2==1])
extr_arr


# 18. Replace all odd numbers to -1 from previous array

# In[ ]:





# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[ ]:


arr_new = np.arange(10)
arr_new[5] =12
arr_new[6] =12
arr_new[7] =12
arr_new[8] =12
arr_new


# 20. Create a 2d array with 1 on the border and 0 inside

# In[ ]:


z_ins = np.array([[1,1,1],[1,0,1],[1,1,1]])
z_ins


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[ ]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1,1] =12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[ ]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0,0] = 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[ ]:


arr12d = np.arange(10)
arr12d
arr123d=arr12d.reshape(2,5)
arr123d[:1]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[ ]:


arr12d = np.arange(10)
arr12d
arr123d=arr12d.reshape(2,5)
arr123d[1:,1:2]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[ ]:


arr12d = np.arange(10)
arr12d
arr123d=arr12d.reshape(2,5)
arr123d[0:2,2:3]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[ ]:


rand_arr = np.random.random((10,10))
min_num = rand_arr.min()
print("Minimum Number is: ", min_num)
max_num = rand_arr.max()
print("Maximum Number is: ", max_num)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[ ]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
comm = np.intersect1d(a,b)
comm


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[ ]:





# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[ ]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data[names != 'Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[ ]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
new_data = names = (names != 'Will') | (names != 'Joe')
data[new_data]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[ ]:


a = np.arange(1,16)
b = a.reshape(5,3)
b


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[ ]:


a = np.arange(1,17)
b = a.reshape(2,8)
c = b.reshape(2,2,4)
c


# 33. Swap axes of the array you created in Question 32

# In[ ]:


np.swapaxes(c,0,1)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[ ]:


a= np.arange(10)
np.sqrt(a)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[ ]:


a = np.random.random(12)
b=np.random.random(12)
max_a = a.max()
max_b = b.max()
array_combination = (max_a, max_b)
array_new = np.vstack(array_combination)
print(array_new)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[ ]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
uniq = np.unique(names)
uniq_sort = sorted(set(names))
uniq_sort


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[ ]:





# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[ ]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
new_array = np.delete(sampleArray, [1,1], 1)
newColumn = np.array([[10,10,10]])
new_array[0,1] =newColumn


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[8]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
dot_pro = np.dot(x*y)
dot_pro


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[6]:


rand_arr = np.random.random(20)
cum_sum = rand_arr.cumsum()
cum_sum

