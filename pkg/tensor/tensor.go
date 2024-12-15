package tensor

import (
	"fmt"
)

// TensorStruct represents a tensor
type TensorStruct struct {
	shape  []int
	stride []int
	data   []float64
}

// Tensor is the interface for a tensor
type Tensor interface {
	Shape() []int
	Stride() []int
	Data() []float64
	String() string
}

// NewTensor creates a new tensor with the given shape and data
func NewTensor(shape []int, data []float64) (*TensorStruct, error) {
	// Check if the shape is valid
	for _, dim := range shape {
		if dim < 0 {
			return nil, fmt.Errorf("invalid shape: %v", shape)
		}
	}

	// Check if data was provided
	if len(data) == 0 {
		return nil, fmt.Errorf("no data provided")
	}

	// Check if the shape is 0-dimensional
	if len(shape) == 0 || (len(shape) == 1 && shape[0] == 0) {
		// Create a 0-dimensional tensor
		return &TensorStruct{
			shape:  shape,
			stride: []int{},
			data:   data,
		}, nil
	}

	// Compute the stride
	stride := ComputeStrides(shape)

	// Create the tensor
	return &TensorStruct{
		shape:  shape,
		stride: stride,
		data:   data,
	}, nil
}

// Shape returns the shape of the tensor
func (t *TensorStruct) Shape() []int {
	return t.shape
}

// Stride returns the stride of the tensor
func (t *TensorStruct) Stride() []int {
	return t.stride
}

// Data returns the data of the tensor
func (t *TensorStruct) Data() []float64 {
	return t.data
}

// String returns a string representation of the tensor
func (t *TensorStruct) String() string {
	return fmt.Sprintf("%v", t.data)
}
