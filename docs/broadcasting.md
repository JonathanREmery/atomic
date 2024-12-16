# Broadcasting

Broadcasting is a powerful mechanism in atomic that allows operations between `Tensors` of different shapes. It automatically expands smaller tensors to match the shape of larger ones, enabling element-wise operations without explicitly copying data.

## Broadcasting Rules

For two tensors to be compatible for broadcasting, they must follow these rules:

1. **Dimension Alignment**: Shapes are compared from right to left
2. **Size Compatibility**: Each dimension must either:
   - Have the same size, or
   - Have size 1 in one of the tensors, or
   - Be missing from the tensor with fewer dimensions

## Examples

### Basic Broadcasting

```go
// Create a 2x3 tensor
a := tensor.NewTensor([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
/*
[
  1 2 3
  4 5 6
]
*/

// Create a 1x3 tensor (row vector)
b := tensor.NewTensor([]int{1, 3}, []float64{7, 8, 9})
/*
[7 8 9]
*/

// Broadcasting happens automatically in operations
c := a.Add(b)
/*
[
  8  10 12
  11 13 15
]
*/
```

### Shape Compatibility Examples

Compatible shapes for broadcasting:
- `(2,3)` and `(3)` → Second tensor is treated as `(1,3)`
- `(3,1)` and `(3,4)` → First tensor's columns are broadcast
- `(2,1,4)` and `(3,1)` → Broadcast to `(2,3,4)`

Incompatible shapes:
- `(2,3)` and `(2,4)` → Mismatched sizes (3 vs 4)
- `(3,2)` and `(2,3)` → Dimensions don't align

## Automatic Broadcasting

Broadcasting happens automatically in atomic when:

1. Creating a tensor with data that doesn't match the shape
2. Performing operations between tensors of different shapes
3. Using mathematical operations (Add, Mul, etc.) with scalars

```go
// Broadcasting with a scalar
t := tensor.NewTensor([]int{2, 2}, []float64{1, 2, 3, 4})
result := t.Add(2) // Adds 2 to every element
```

## Performance Considerations

Broadcasting is memory-efficient as it doesn't create copies of the data. Instead, it uses virtual striding to perform operations as if the smaller tensor was expanded.
