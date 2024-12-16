# Tensors

A `Tensor` is the fundamental data structure in atomic. It represents data ranging from scalars to multi-dimensional arrays. The data is stored in a contiguous block of memory, and the `shape` of the `Tensor` defines how the data is organized.

You can create a tensor using the `NewTensor` function, which takes `shape`, and `data` as arguments. The `shape` is a slice of integers that specifies the dimensions of the tensor. The `data` is a slice of floats that represents the values of the tensor.

You can create a 0-dimensional `Tensor`, which represents a scalar:

```go
t, err := tensor.NewTensor([]int{}, []float64{3.14})
```

You can have as many dimensions as you want, but they must be positive integers:

```go
t, err := tensor.NewTensor([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
```

This will create a 2x3 `Tensor` with the following data:

```
1 2 3
4 5 6
```

You can add, subtract, multiply, and divide `Tensors` using the `Add`, `Sub`, `Mul`, and `Div` methods.

```go
t1, _ := tensor.NewTensor([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
t2, _ := tensor.NewTensor([]int{2, 3}, []float64{7, 8, 9, 10, 11, 12})

t3, _ := t1.Add(t2)
t3, _ = t3.Sub(t1.Mul(t2))
t4, _ := t3.Div(t2)
```

For more advanced tensor operations, see:
- [Views](views.md) - Learn about efficient tensor reshaping without data copying
- [Broadcasting](broadcasting.md) - Understand how atomic handles operations between tensors of different shapes