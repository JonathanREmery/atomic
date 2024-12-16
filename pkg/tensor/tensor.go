package tensor

import (
	"fmt"
	"reflect"
	"strings"
)

// TensorStruct represents a tensor
type TensorStruct struct {
	shape  []int
	stride []int
	data   []float64
}

// computeStrides computes the stride of a tensor given its shape
func computeStrides(shape []int) []int {
	// If the shape is empty, return the empty strides
	if len(shape) == 0 {
		return []int{}
	}

	// Initialize the stride
	strides := make([]int, len(shape))

	// Calculate the stride
	strides[len(strides)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	// Return the stride
	return strides
}

// Tensor is the interface for a tensor
type Tensor interface {
	Shape() []int
	Rank() int
	Stride() []int
	Data() []float64

	String() string

	View(shape []int) (*ViewStruct, error)
	Broadcast(shape []int) (*BroadcastStruct, error)

	Add(*TensorStruct) (*TensorStruct, error)
	Sub(*TensorStruct) (*TensorStruct, error)
	Mul(*TensorStruct) (*TensorStruct, error)
	Div(*TensorStruct) (*TensorStruct, error)

	addBroadcast(*BroadcastStruct) (*TensorStruct, error)
	subBroadcast(*BroadcastStruct) (*TensorStruct, error)
	mulBroadcast(*BroadcastStruct) (*TensorStruct, error)
	divBroadcast(*BroadcastStruct) (*TensorStruct, error)
}

// NewScalar creates a new scalar tensor
func NewScalar(data float64) *TensorStruct {
	return &TensorStruct{
		shape:  []int{},
		stride: []int{},
		data:   []float64{data},
	}
}

// NewTensor creates a new tensor with the given shape and data
func NewTensor(shape []int, data []float64) (*TensorStruct, error) {
	// Check if any dimensions are negative
	for _, dim := range shape {
		if dim < 0 {
			return nil, fmt.Errorf("invalid shape: %v", shape)
		}
	}

	// Check if data is empty
	if len(data) == 0 {
		return nil, fmt.Errorf("no data provided")
	}

	// Check if the shape is a scalar
	if len(shape) == 0 || (len(shape) == 1 && shape[0] == 0) {
		return NewScalar(data[0]), nil
	}

	// Calculate the expected data size
	expectedLength := 1
	for _, dim := range shape {
		expectedLength *= dim
	}

	// Check if we don't have enough data
	if len(data) < expectedLength {
		// TODO: Replicate data
		return nil, fmt.Errorf("data length %d is less than shape capacity %d", len(data), expectedLength)
	}

	// Check if we have too much data
	if len(data) > expectedLength {
		return nil, fmt.Errorf("data length %d exceeds shape capacity %d", len(data), expectedLength)
	}

	// Compute the stride
	stride := computeStrides(shape)

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

// Rank returns the rank of the tensor
func (t *TensorStruct) Rank() int {
	return len(t.shape)
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
	switch len(t.shape) {
	case 0:
		return fmt.Sprintf("%.2f", t.data[0])
	case 1:
		return fmt.Sprintf("[%s]", formatSlice(t.data))
	case 2:
		rows := make([]string, t.shape[0])
		for i := 0; i < t.shape[0]; i++ {
			start := i * t.shape[1]
			end := start + t.shape[1]
			rows[i] = formatSlice(t.data[start:end])
		}
		return "[\n " + strings.Join(rows, "\n ") + "\n]"
	default:
		return fmt.Sprintf("Tensor(shape=%v, data=[%s])", t.shape, formatSlice(t.data))
	}
}

// View returns a view of the tensor
func (t *TensorStruct) View(shape []int) (*ViewStruct, error) {
	return NewView(t).Reshape(shape)
}

// Broadcast returns a broadcasted tensor
func (t *TensorStruct) Broadcast(shape []int) (*BroadcastStruct, error) {
	return NewBroadcast(shape, t)
}

// Add adds another tensor to this tensor
func (t *TensorStruct) Add(other *TensorStruct) (*TensorStruct, error) {
	// Check if shapes not the same
	if !reflect.DeepEqual(t.shape, other.shape) {
		// Sort the tensors by rank
		tensors := []*TensorStruct{t, other}
		if other.Rank() < t.Rank() {
			tensors = []*TensorStruct{other, t}
		}

		// Broadcast the smaller tensor to the shape of the larger tensor
		broadcasted, err := NewBroadcast(tensors[1].shape, tensors[0])
		if err != nil {
			// If there is an error, return it
			return nil, err
		}

		// Add the broadcasted tensor to the larger tensor
		return tensors[1].addBroadcast(broadcasted)
	}

	// Initialize the result
	result := make([]float64, len(t.data))

	// Perform element-wise addition
	for i := range t.data {
		result[i] = t.data[i] + other.data[i]
	}

	// Return the new tensor, with the result data
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// Sub subtracts another tensor from this tensor
func (t *TensorStruct) Sub(other *TensorStruct) (*TensorStruct, error) {
	// Check if shapes are not the same
	if !reflect.DeepEqual(t.shape, other.shape) {
		// Sort the tensors by rank
		tensors := []*TensorStruct{t, other}
		if other.Rank() < t.Rank() {
			tensors = []*TensorStruct{other, t}
		}

		// Broadcast the smaller tensor to the shape of the larger tensor
		broadcasted, err := NewBroadcast(tensors[1].shape, tensors[0])
		if err != nil {
			// If there is an error, return it
			return nil, err
		}

		// Subtract the broadcasted tensor from the larger tensor
		return tensors[1].subBroadcast(broadcasted)
	}

	// Initialize the result
	result := make([]float64, len(t.data))

	// Perform element-wise subtraction
	for i := range t.data {
		result[i] = t.data[i] - other.data[i]
	}

	// Return the new tensor, with the result data
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// Mul multiplies this tensor by another tensor
func (t *TensorStruct) Mul(other *TensorStruct) (*TensorStruct, error) {
	// Check if shapes are not the same
	if !reflect.DeepEqual(t.shape, other.shape) {
		// Sort the tensors by rank
		tensors := []*TensorStruct{t, other}
		if other.Rank() < t.Rank() {
			tensors = []*TensorStruct{other, t}
		}

		// Broadcast the smaller tensor to the shape of the larger tensor
		broadcasted, err := NewBroadcast(tensors[1].shape, tensors[0])
		if err != nil {
			// If there is an error, return it
			return nil, err
		}

		// Multiply the broadcasted tensor with the larger tensor
		return tensors[1].mulBroadcast(broadcasted)
	}

	// Initialize the result
	result := make([]float64, len(t.data))

	// Perform element-wise multiplication
	for i := range t.data {
		result[i] = t.data[i] * other.data[i]
	}

	// Return the new tensor, with the result data
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// Div divides this tensor by another tensor
func (t *TensorStruct) Div(other *TensorStruct) (*TensorStruct, error) {
	// Check if shapes are not the same
	if !reflect.DeepEqual(t.shape, other.shape) {
		// Sort the tensors by rank
		tensors := []*TensorStruct{t, other}
		if other.Rank() < t.Rank() {
			tensors = []*TensorStruct{other, t}
		}

		// Broadcast the smaller tensor to the shape of the larger tensor
		broadcasted, err := NewBroadcast(tensors[1].shape, tensors[0])
		if err != nil {
			// If there is an error, return it
			return nil, err
		}

		// Divide the broadcasted tensor with the larger tensor
		return tensors[1].divBroadcast(broadcasted)
	}

	// Initialize the result
	result := make([]float64, len(t.data))

	// Perform element-wise division
	for i := range t.data {
		if other.data[i] == 0 {
			return nil, fmt.Errorf("division by zero")
		}
		result[i] = t.data[i] / other.data[i]
	}

	// Return the new tensor, with the result data
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// addBroadcast adds a broadcast to this tensor
func (t *TensorStruct) addBroadcast(b *BroadcastStruct) (*TensorStruct, error) {
	// Check if shapes not the same
	if !reflect.DeepEqual(t.shape, b.broadcastShape) {
		return nil, fmt.Errorf("shape mismatch")
	}

	// Initialize the result
	result := make([]float64, len(t.data))

	// Perform element-wise addition
	for i := range t.data {
		result[i] = t.data[i] + b.GetFlat(i)
	}

	// Return the new tensor, with the result data
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// subBroadcast subtracts a broadcast from this tensor
func (t *TensorStruct) subBroadcast(b *BroadcastStruct) (*TensorStruct, error) {
	// Check if shapes not the same
	if !reflect.DeepEqual(t.shape, b.broadcastShape) {
		return nil, fmt.Errorf("shape mismatch")
	}

	// Initialize the result
	result := make([]float64, len(t.data))

	// Perform element-wise subtraction
	for i := range t.data {
		result[i] = t.data[i] - b.GetFlat(i)
	}

	// Return the new tensor, with the result data
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// mulBroadcast multiplies a broadcast with this tensor
func (t *TensorStruct) mulBroadcast(b *BroadcastStruct) (*TensorStruct, error) {
	// Check if shapes not the same
	if !reflect.DeepEqual(t.shape, b.broadcastShape) {
		return nil, fmt.Errorf("shape mismatch")
	}

	// Initialize the result
	result := make([]float64, len(t.data))

	// Perform element-wise multiplication
	for i := range t.data {
		result[i] = t.data[i] * b.GetFlat(i)
	}

	// Return the new tensor, with the result data
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}

// divBroadcast divides this tensor by a broadcast
func (t *TensorStruct) divBroadcast(b *BroadcastStruct) (*TensorStruct, error) {
	// Check if shapes not the same
	if !reflect.DeepEqual(t.shape, b.broadcastShape) {
		return nil, fmt.Errorf("shape mismatch")
	}

	// Initialize the result
	result := make([]float64, len(t.data))

	// Perform element-wise division
	for i := range t.data {
		val := b.GetFlat(i)
		if val == 0 {
			return nil, fmt.Errorf("division by zero")
		}
		result[i] = t.data[i] / val
	}

	// Return the new tensor, with the result data
	return &TensorStruct{
		shape:  t.shape,
		stride: t.stride,
		data:   result,
	}, nil
}
