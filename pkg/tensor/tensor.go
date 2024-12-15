package tensor

import (
	"fmt"
	"reflect"
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
	Add(*TensorStruct) (*TensorStruct, error)
	Sub(*TensorStruct) (*TensorStruct, error)
	Mul(*TensorStruct) (*TensorStruct, error)
	Div(*TensorStruct) (*TensorStruct, error)
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

	// Calculate the expected data size
	expectedLength := 1
	for _, dim := range shape {
		expectedLength *= dim
	}

	// Check if we have too much data
	if len(data) > expectedLength {
		return nil, fmt.Errorf("data length %d exceeds shape capacity %d", len(data), expectedLength)
	}

	// Compute the stride
	stride := ComputeStrides(shape)

	// Check if we should broadcast
	if len(data) < expectedLength {
		// Broadcast the data
		return Broadcast(
			&TensorStruct{
				shape:  shape,
				stride: stride,
				data:   data,
			},
			shape,
		)
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
	// Check for nil tensor
	if t == nil {
		return nil, fmt.Errorf("cannot broadcast nil tensor")
	}

	// Validate target shape
	if len(shape) == 0 {
		return nil, fmt.Errorf("target shape cannot be empty")
	}

	// Check if any dimension is negative
	for _, dim := range shape {
		if dim < 0 {
			return nil, fmt.Errorf("invalid dimension in target shape: %d", dim)
		}
	}

	// Calculate the expected data size
	expectedLength := 1
	for _, dim := range shape {
		expectedLength *= dim
	}

	// Check if we have the right amount of data already
	if len(t.data) == expectedLength {
		return t, nil
	}

	// Check if we have too much data
	if len(t.data) > expectedLength {
		return nil, fmt.Errorf("data length %d exceeds shape capacity %d", len(t.data), expectedLength)
	}

	// Initialize the actual shape
	var actualShape []int
	if len(t.data) == 1 {
		// Single value should be treated as a scalar
		actualShape = []int{}
	} else {
		// Multiple values should be treated as a 1D tensor
		actualShape = []int{len(t.data)}
	}

	// Initialize the desired shape
	desiredShape := shape

	// Align the shapes
	alignedShape, err := AlignShapes(actualShape, desiredShape)
	if err != nil {
		return nil, fmt.Errorf("cannot broadcast data of length %d to shape %v: %v", len(t.data), shape, err)
	}

	// For scalars, we need to use the desired shape directly
	if len(actualShape) == 0 {
		// Create broadcasted data
		broadcastedData := make([]float64, expectedLength)
		for i := range broadcastedData {
			broadcastedData[i] = t.data[0]
		}

		// Create and return the tensor
		return &TensorStruct{
			shape:  desiredShape,
			stride: ComputeStrides(desiredShape),
			data:   broadcastedData,
		}, nil
	}

	// Compute the broadcast stride
	broadcastStride := ComputeBroadcastStrides(actualShape, alignedShape, ComputeStrides(actualShape))

	// Create broadcasted data
	broadcastedData := make([]float64, expectedLength)
	for i := range broadcastedData {
		broadcastedData[i] = t.data[i%len(t.data)]
	}

	// Create and return the tensor
	return &TensorStruct{
		shape:  desiredShape,
		stride: broadcastStride,
		data:   broadcastedData,
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

// Add adds another tensor to this tensor
func (t *TensorStruct) Add(other *TensorStruct) (*TensorStruct, error) {
	// Check if shapes are compatible
	if !reflect.DeepEqual(t.shape, other.shape) {
		// Check if shapes are broadcastable
		otherBroadcasted, err := Broadcast(other, t.shape)
		if err == nil {
			// If so, broadcast the other tensor and add
			return t.Add(otherBroadcasted)
		}

		// If not, return an error
		return nil, fmt.Errorf("incompatible shapes: %v and %v", t.shape, other.shape)
	}

	// Create a new tensor with the same shape
	result := make([]float64, len(t.data))

	// Perform element-wise addition
	for i := range t.data {
		result[i] = t.data[i] + other.data[i]
	}

	// Return the new tensor
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// Sub subtracts another tensor from this tensor
func (t *TensorStruct) Sub(other *TensorStruct) (*TensorStruct, error) {
	// Check if shapes are compatible
	if !reflect.DeepEqual(t.shape, other.shape) {
		// Check if shapes are broadcastable
		otherBroadcasted, err := Broadcast(other, t.shape)
		if err == nil {
			// If so, broadcast the other tensor and subtract
			return t.Sub(otherBroadcasted)
		}

		// If not, return an error
		return nil, fmt.Errorf("incompatible shapes: %v and %v", t.shape, other.shape)
	}

	// Create a new tensor with the same shape
	result := make([]float64, len(t.data))

	// Perform element-wise subtraction
	for i := range t.data {
		result[i] = t.data[i] - other.data[i]
	}

	// Return the new tensor
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// Mul multiplies this tensor by another tensor
func (t *TensorStruct) Mul(other *TensorStruct) (*TensorStruct, error) {
	// Check if shapes are compatible
	if !reflect.DeepEqual(t.shape, other.shape) {
		// Check if shapes are broadcastable
		otherBroadcasted, err := Broadcast(other, t.shape)
		if err == nil {
			// If so, broadcast the other tensor and multiply
			return t.Mul(otherBroadcasted)
		}

		// If not, return an error
		return nil, fmt.Errorf("incompatible shapes: %v and %v", t.shape, other.shape)
	}

	// Create a new tensor with the same shape
	result := make([]float64, len(t.data))

	// Perform element-wise multiplication
	for i := range t.data {
		result[i] = t.data[i] * other.data[i]
	}

	// Return the new tensor
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// Div divides this tensor by another tensor
func (t *TensorStruct) Div(other *TensorStruct) (*TensorStruct, error) {
	// Check if shapes are compatible
	if !reflect.DeepEqual(t.shape, other.shape) {
		// Check if shapes are broadcastable
		otherBroadcasted, err := Broadcast(other, t.shape)
		if err == nil {
			// If so, broadcast the other tensor and divide
			return t.Div(otherBroadcasted)
		}

		// If not, return an error
		return nil, fmt.Errorf("incompatible shapes: %v and %v", t.shape, other.shape)
	}

	// Create a new tensor with the same shape
	result := make([]float64, len(t.data))

	// Perform element-wise division
	for i := range t.data {
		if other.data[i] == 0 {
			return nil, fmt.Errorf("division by zero")
		}
		result[i] = t.data[i] / other.data[i]
	}

	// Return the new tensor
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}
