import numpy as np
import theano.tensor as T
from theano import function as Tfunc

def im2col(img, kernel_size):
    depth = img.shape[2]
    # Stack along depth
    duplicated = T.tile(img, (1, 1, kernel_size**2, 1))
    # Roll each repeated layer so each hole through (x,y) has all values from the filter block
    for c in range(kernel_size):
        layer = c*kernel_size*depth
        next_layer = layer + depth*kernel_size
        # Roll all remaining layers along x-direction
        duplicated = T.set_subtensor(duplicated[:,:,layer:next_layer,:], T.roll(duplicated[:,:,layer:next_layer,:], -c, axis=1))
        for r in range(kernel_size):
            layer = c*kernel_size*depth + r*depth
            next_layer = layer + depth
            # Roll single layer along y-direction
            duplicated = T.set_subtensor(duplicated[:,:,layer:next_layer,:], T.roll(duplicated[:,:,layer:next_layer,:], -r, axis=0))

    convolved_dims = (img.shape[0] - kernel_size + 1, img.shape[1] - kernel_size + 1)
    duplicated = duplicated.dimshuffle((3, 1, 0, 2))
    flattened = T.reshape(duplicated[:,:convolved_dims[0],:convolved_dims[1],:],
                          (img.shape[3], convolved_dims[0]*convolved_dims[1], depth*(kernel_size**2)))
    return flattened.dimshuffle((2, 1, 0))

def col2im(img, dims, kernel_size, depth):
    n_images = img.shape[2]
    strides = T.arange(0, img.shape[1])
    trimmed = img[0:depth,strides,:]
    rearranged = trimmed.dimshuffle((2, 1, 0))
    shaped = T.reshape(rearranged, (n_images, dims[1], dims[0], depth))
    return shaped.dimshuffle((2, 1, 3, 0))


if __name__ == '__main__':
    T_img = T.tensor4('img')
    reshaped = im2col(T_img, 3)
    flat_func = Tfunc([T_img], reshaped)
    my_img = np.arange(32).reshape((4,4,2,1))
    np_reshaped = flat_func(my_img)
    print(my_img[:,:,0,0])
    print(my_img[:,:,1,0])
    print(np_reshaped[:,:,0])

    flat_img = T.tensor3('flatimg')
    shaped = col2im(flat_img, (2,2), 3, 2)
    shape_func = Tfunc([flat_img], shaped)
    shaped = shape_func(np_reshaped)
    print(shaped[:,:,0,0])
    print(shaped[:,:,1,0])