import numpy as np

def power_diff_numpy(input_x,input_y,input_z):
    x_shape = np.shape(input_x)
    y_shape = np.shape(input_y)
    x = np.reshape(input_x,(-1,y_shape[-1]))
    x_new_shape = np.shape(x)
    y = np.reshape(input_y,(-1))
    output = []
    for i in range(x_new_shape[0]):
        tmp1 = tmp2 = x[i]-y
        for j in range(input_z-1):
            tmp1 = tmp1*tmp2
        output.append(tmp1)
    output = np.reshape(output,x_shape)
    return output

