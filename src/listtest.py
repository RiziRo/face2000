
import numpy as np

top_list = dict()
m = [1,2,3]
n = [1,2,4]
m = np.array(m)
n = np.array(n)
print(np.sum(np.equal(m,n)))
arr=np.array([[3,1,4],[5,4,3],[3,4,5]])
for i in range(2):
    indices = np.argmax(arr, axis=1)
    arr[np.arange(len(indices)), indices] = -10000
    top_list[i] = indices
print(top_list)