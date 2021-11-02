#!/usr/bin/env python
# coding: utf-8

# In[1]:


from csv import reader
import os
import csv
from random import randrange

import numpy as np
import math
import operator


# In[2]:


def load_file(filename):
    file = open(filename, "r")
    lines = reader(file)
    data = list(lines)
    vector = ''
    vector_array = []
    for row in range(len(data)):
        vector += data[row][0]
    for i in range(len(vector)):
        vector_array.append(int(vector[i]))
    return vector_array


# In[3]:


def load_class_data(filename, label, path):
    files = os.listdir(path)
    class_data = []
    for file in files:
        if file.startswith(filename):
            class_data.append([load_file(path + '\\' + file), label])
    return class_data


# In[4]:


class_0 = load_class_data('class_0', 0, r'dataset1/training_validation')
class_1 = load_class_data('class_1', 1, r'dataset1/training_validation')
class_2 = load_class_data('class_2', 2, r'dataset1/training_validation')
class_3 = load_class_data('class_3', 3, r'dataset1/training_validation')
class_4 = load_class_data('class_4', 4, r'dataset1/training_validation')
class_5 = load_class_data('class_5', 5, r'dataset1/training_validation')
class_6 = load_class_data('class_6', 6, r'dataset1/training_validation')
class_7 = load_class_data('class_7', 7, r'dataset1/training_validation')
class_8 = load_class_data('class_8', 8, r'dataset1/training_validation')
class_9 = load_class_data('class_9', 9, r'dataset1/training_validation')


# In[5]:


full_training_dataset = class_0 + class_1 + class_2 + class_3 + class_4 + class_5 + class_6 + class_7 + class_8 + class_9


# In[6]:


class_0_test = load_class_data('class_0', 0, r'dataset1/test')
class_1_test = load_class_data('class_1', 1, r'dataset1/test')
class_2_test = load_class_data('class_2', 2, r'dataset1/test')
class_3_test = load_class_data('class_3', 3, r'dataset1/test')
class_4_test = load_class_data('class_4', 4, r'dataset1/test')
class_5_test = load_class_data('class_5', 5, r'dataset1/test')
class_6_test = load_class_data('class_6', 6, r'dataset1/test')
class_7_test = load_class_data('class_7', 7, r'dataset1/test')
class_8_test = load_class_data('class_8', 8, r'dataset1/test')
class_9_test = load_class_data('class_9', 9, r'dataset1/test')


# In[7]:


full_testing_dataset = class_0_test + class_1_test + class_2_test + class_3_test + class_4_test + class_5_test + class_6_test + class_7_test + class_8_test + class_9_test


# In[8]:


def get_euc_distance(instance, training_dataset):
    b = np.array(instance)
    distances = []
    for x in training_dataset:
        a = [np.array(x[0]), x[1]]
        distance = np.sqrt(np.sum((a[0] - b) ** 2))
        distances.append([distance, a[1]])
    return distances


# In[14]:


def get_neighbours(instance, training_dataset, k):
    distances = get_euc_distance(instance, training_dataset)
    distances.sort(key=lambda x: x[0])
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors


# In[15]:


def get_prediction(instance, training_dataset, k):
    neighbours = get_neighbours(instance, training_dataset, k)
    neighbour_addition = 0
    for x in range(len(neighbours)):
        neighbour_addition += neighbours[x][1]
    return neighbour_addition / k


# In[16]:


def get_all_predictions(testing_dataset, training_dataset, k):
    prediction_array = []
    for instance in testing_dataset:
        predictions = get_prediction(instance[0], training_dataset, k)
        prediction_array.append([predictions, instance[1]])
    return prediction_array


# In[17]:


def accuracy_metric(array):
    correct = 0
    for i in range(len(array)):
        if array[i][0] == array[i][1]:
            correct += 1
    return correct / float(len(array)) * 100.0


# In[18]:


def get_all_accuracies(testing_dataset, training_dataset):
    accuracy_array = []
    for i in range(1,12):
        prediction_array = get_all_predictions(testing_dataset, training_dataset, i)
        accuracy = accuracy_metric(prediction_array)
        accuracy_array.append(accuracy)
    return accuracy_array


# In[19]:


def cross_validation_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# In[20]:


split = cross_validation_split(full_training_dataset, 5)


# In[21]:


split1 = [split[0]+split[1]+split[2]+split[3], split[4]]
split2 = [split[0]+split[1]+split[2]+split[4], split[3]]
split3 = [split[0]+split[1]+split[4]+split[3], split[2]]
split4 = [split[0]+split[4]+split[2]+split[3], split[1]]
split5 = [split[4]+split[1]+split[2]+split[3], split[0]]


# In[22]:


split1_accuracy_array = np.array(get_all_accuracies(split1[1], split1[0]))
split2_accuracy_array = np.array(get_all_accuracies(split2[1], split2[0]))
split3_accuracy_array = np.array(get_all_accuracies(split3[1], split3[0]))
split4_accuracy_array = np.array(get_all_accuracies(split4[1], split4[0]))
split5_accuracy_array = np.array(get_all_accuracies(split5[1], split5[0]))


# In[23]:


cv_accuracy_array = ((split1_accuracy_array + split2_accuracy_array + split3_accuracy_array + split4_accuracy_array + split5_accuracy_array)/5).tolist()


# In[25]:


print(cv_accuracy_array)


# In[26]:


accuracy_array = get_all_accuracies(full_testing_dataset, full_training_dataset)


# In[27]:


print(accuracy_array)


# In[ ]:




