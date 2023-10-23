# visualize loss of train and val
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def merge_sort(arr):
     if len(arr) <= 1:
          return arr

     # Divide the array into two halves
     mid = len(arr) // 2
     left_half = arr[:mid]
     right_half = arr[mid:]

     # Recursively sort the two halves
     left_half = merge_sort(left_half)
     right_half = merge_sort(right_half)

     # Merge the sorted halves
     return merge(left_half, right_half)


def merge(left, right):
     merged = []
     left_idx, right_idx = 0, 0

     # Compare elements from both halves and merge them in sorted order
     while left_idx < len(left) and right_idx < len(right):
          if left[left_idx] <= right[right_idx]:
               merged.append(left[left_idx])
               left_idx += 1
          else:
               merged.append(right[right_idx])
               right_idx += 1

     # Append any remaining elements from the left and right halves
     merged.extend(left[left_idx:])
     merged.extend(right[right_idx:])

     return merged


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
y = [22642.5, 6816.01, 11270.9, 19077.7, 1786.08,
     17815.4, 8249.95, 12431.3, 2374.35, 4395.22,
     743.567, 22864.5, 7173.06, 1828.48, 2738.19]

plt.bar(x, y, label = 'Time per experiment (sec)')
plt.title("Total time spent by experiment")
plt.legend()
plt.show()

"""
plt.plot(epochs, mAP50, label = 'box mAP50')
plt.plot(epochs, mAP95, label = 'segmentation mAP50')
plt.title("Evaluation metrics")
plt.legend()
plt.show()

plt.plot(epochs, seg_loss_train, label = 'train loss')
plt.plot(epochs, seg_loss_val, label = 'val loss')
plt.title("Segmentation loss")
plt.legend()
plt.show()

plt.plot(epochs, box_loss_train, label = 'train loss')
plt.plot(epochs, box_loss_val, label = 'val loss')
plt.title("Box loss")
plt.legend()
plt.show()

plt.plot(epochs, dfl_loss_train, label = 'train loss')
plt.plot(epochs, dfl_loss_val, label = 'val loss')
plt.title("dfl loss")
plt.legend()
plt.show()

plt.plot(epochs, cls_loss_train, label = 'train loss')
plt.plot(epochs, cls_loss_val, label = 'val loss')
plt.title("cls loss")
plt.legend()
plt.show()
#df.columns.values
"""