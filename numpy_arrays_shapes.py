import numpy as np

print("===== 1D ARRAY =====")
arr1 = np.array([10, 20, 30, 40, 50])
print(arr1)
print("Shape:", arr1.shape)

print("\n===== 2D ARRAY =====")
arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(arr2)
print("Shape:", arr2.shape)

print("\n===== 3D ARRAY =====")
arr3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(arr3)
print("Shape:", arr3.shape)

print("\nVerification Completed Successfully!")