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

## Broadcasting

Broadcasting is a way to change the shape of a tensor without changing its data. It is used to make tensors compatible with each other.

### Broadcasting Scalars

```go
t1 := tensor.NewTensor([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
/*
Shape:
[2, 3]
Stride:
[3, 1]
Data:
[1, 2, 3, 4, 5, 6]

Representation:
[1, 2, 3]
[4, 5, 6]
*/


t2 := tensor.NewTensor([]int{}, []float64{3})
/*
Shape:
[]
Stride:
[]
Data:
[3]

Representation:
3
*/


t3 := tensor.Broadcast(t2, t1.Shape())
/*
Shape:
[2, 3]
Stride:
[3, 1]
Data:
[3, 3, 3, 3, 3, 3]

Representation:
[3, 3, 3]
[3, 3, 3]
*/
```

### Broadcasting 1D Tensors

```go
t1 := tensor.NewTensor([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
/*
Shape:
[2, 3]
Stride:
[3, 1]
Data:
[1, 2, 3, 4, 5, 6]

Representation:
[1, 2, 3]
[4, 5, 6]
*/

t2 := tensor.NewTensor([]int{3}, []float64{3, 3, 3})
/*
Shape:
[3]
Stride:
[1]
Data:
[1, 2, 3]

Representation:
[1
 2,
 3]
*/

t3 := tensor.Broadcast(t2, t1.Shape())
/*
Shape:
[2, 3]
Stride:
[3, 1]
Data:
[1, 2, 3, 1, 2, 3]

Representation:
[1, 2, 3]
[1, 2, 3]
*/
```

### Broadcasting 2D Tensors

```go
t1 := tensor.NewTensor([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
/*
Shape:
[2, 3]
Stride:
[3, 1]
Data:
[1, 2, 3, 4, 5, 6]

Representation:
[1, 2, 3]
[4, 5, 6]
*/

t2 := tensor.NewTensor([]int{1, 3}, []float64{1, 2, 3})
/*
Shape:
[1, 3]
Stride:
[3, 1]
Data:
[1, 2, 3]

Representation:
[1, 2, 3]
*/

t3 := tensor.Broadcast(t2, t1.Shape())
/*
Shape:
[2, 3]
Stride:
[3, 1]
Data:
[1, 2, 3, 1, 2, 3, 1, 2, 3]

Representation:
[1, 2, 3]
[1, 2, 3]
[1, 2, 3]
*/
```