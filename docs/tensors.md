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

## Views

`Views` allow for efficient reshaping of `Tensors`. A `View` is a `Tensor` that shares the same data as the original `Tensor`, but with a different `shape` and `stride`.

```go

t1 := tensor.NewTensor([]int{3, 2}, []float64{1, 2, 3, 4, 5, 6})
t2 := t1.View([]int{2, 3})

/*
t1 [
  1 2
  3 4
  5 6
]

t2 [
  1 2 3
  4 5 6
]
*/

```

## Broadcasting

`Broadcasting` is a way to change the shape of a `Tensor` without changing its data. It is used to make `Tensors` compatible with each other. `Broadcasting` is done automatically when a `Tensor` is created and the size of the `data` does not match the size of the `shape`. This is called "auto-broadcasting". `Broadcasting` also occurs automatically when an operation is performed between `Tensors` with different shapes.

There are some rules to make sure that two `Tensors` can be `Broadcasted` together:
1. The `Tensor` with the smaller number of dimensions is broadcasted to the shape of the `Tensor` with the larger number of dimensions.
2. The `shapes` of the `Tensors` are aligned on the right side, then the smaller `shape` is padded with 1s on the left.
3. So long as every dimension either matches or is `broadcasted` to 1, the `Tensors` can be `Broadcasted`.