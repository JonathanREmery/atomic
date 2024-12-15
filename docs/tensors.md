# Tensors

A tensor is the fundamental data structure in atomic. It represents data ranging from scalars to multi-dimensional arrays. The data is stored in a contiguous block of memory, and the shape of the tensor defines how the data is organized.

You can create a tensor using the `NewTensor` function, which takes a shape and data as arguments. The shape is a slice of integers that specifies the dimensions of the tensor. The data is a slice of floats that represents the values of the tensor.

```go
t := tensor.NewTensor([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
```

This will create a 2x3 tensor with the following data:

```
1 2 3
4 5 6
```

You can create 0 dimensional tensors, which are scalars:

```go
t := tensor.NewTensor([]int{}, []float64{3.14})
```