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

	// Initialize the stride
	stride := make([]int, len(shape))

	// Calculate the stride
	stride[len(stride)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		stride[i] = shape[i+1] * stride[i+1]
	}

	// Create the tensor
	return &TensorStruct{
		shape:  shape,
		stride: stride,
		data:   data,
	}, nil
}

// Broadcast broadcasts the tensor to the given shape
func Broadcast(t *TensorStruct, shape []int) (*TensorStruct, error) {
	// Check if the shape is valid
	for _, dim := range shape {
		if dim < 0 {
			return nil, fmt.Errorf("invalid shape: %v", shape)
		}
	}

	// Check if the shape is 0-dimensional
	if len(shape) == 0 || (len(shape) == 1 && shape[0] == 0) {
		return nil, fmt.Errorf("cannot broadcast to 0-dimensional tensor")
	}

	// Check if the shape is 1-dimensional
	if len(shape) == 1 && shape[0] == 1 {
		return nil, fmt.Errorf("cannot broadcast to 1-dimensional tensor")
	}

	// Check if broadcasting would reduce dimensionality
	if len(t.shape) > len(shape) {
		return nil, fmt.Errorf("broadcasting cannot reduce dimensionality")
	}

	// Create a 1 padded shape
	paddedShape := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		iInv := len(shape) - 1 - i

		if iInv < len(t.shape) {
			paddedShape[i] = t.shape[iInv]
		} else {
			paddedShape[i] = 1
		}
	}

	// Check if broadcasting is needed
	broadcastingNeeded := false
	for idx, dim := range shape {
		if paddedShape[idx] != dim {
			broadcastingNeeded = true
		}
	}

	// If broadcasting is not needed, return the tensor
	if !broadcastingNeeded {
		newShape := make([]int, len(shape))
		copy(newShape, shape)

		return NewTensor(newShape, t.data)
	}

	// Check if broadcasting is possible
	for idx, dim := range shape {
		if paddedShape[idx] != dim && paddedShape[idx] != 1 {
			return nil, fmt.Errorf("cannot broadcast to shape %v", shape)
		}
	}

	// Initialize the new shape
	newShape := make([]int, len(shape))
	copy(newShape, shape)

	// Initialize the new data
	newData := make([]float64, len(t.data))
	copy(newData, t.data)

	// Initialize the number of replications
	numReplicates := shape[0]

	// Get the product of the remaining dimensions
	for i := 1; i < len(shape); i++ {
		numReplicates *= shape[i]
	}

	// Divide the number of replications by the number of elements in the tensor
	numReplicates /= len(t.data)

	// Subtract 1 from the number of replications
	numReplicates -= 1

	// Replicate the data
	for i := 0; i < numReplicates; i++ {
		newData = append(newData, newData[:len(t.data)]...)
	}

	// Return the new tensor
	return NewTensor(newShape, newData)
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
