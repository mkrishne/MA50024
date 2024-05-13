import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv
import os
import math

fields = []

# Reading csv file for male data
with open("category.csv", "r") as csv_file:
  reader = csv.reader(csv_file, delimiter=',')
  fields = next(reader)
  for row in reader:
        fields.append(row[1])
  fields = np.array(fields)
csv_file.close()

fields = fields[2:]

train_small_csv = []
# Reading csv file for male data
with open("train_small.csv", "r") as csv_file:
  reader = csv.reader(csv_file, delimiter=',')
  train_small_csv = next(reader)
  for row in reader:
        file_name = row[1]
        field_name = row[2]
        train_small_csv.append(list([file_name,field_name]))
  train_small_csv = np.array(train_small_csv, dtype=object)
csv_file.close()
train_small_csv = train_small_csv[3:]

parent_dir = "dir_test"
for var in fields:
  path = os.path.join(parent_dir, var)
  try: 
      os.mkdir(path) 
  except OSError as error: 
      print(error) 