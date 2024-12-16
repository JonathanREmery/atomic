package tensor

import (
	"fmt"
	"strings"
)

// BroadcastStruct represents a broadcast
type BroadcastStruct struct {
	broadcastShape []int
	strides        []int
	tensor         *TensorStruct
}

// Broadcast is the interface for a broadcast
type Broadcast interface {
	Shape() []int
	Stride() []int
	Data() []float64

	String() string

	GetFlat(idx int) float64
	Get(idx []int) (float64, error)

	ToTensor() (*TensorStruct, error)
}

// validBroadcast checks if two tensors can be broadcasted
func validBroadcast(broadcastShape []int, tensor *TensorStruct) error {
	// Check if broadcast shape is empty
	if len(broadcastShape) == 0 {
		return fmt.Errorf("cannot broadcast to a 0-dimensional shape")
	}

	// Check if tensor is nil
	if tensor == nil {
		return fmt.Errorf("cannot broadcast nil tensor")
	}

	// Check if any dimensions are negative
	for _, dim := range broadcastShape {
		if dim < 0 {
			return fmt.Errorf("negative dimension in target shape: %d", dim)
		}
	}

	// Get the tensor's shape
	tensorShape := tensor.Shape()

	// Check each dimension from right to left
	for i := 0; i < len(broadcastShape) || i < len(tensorShape); i++ {
		// Get the dimensions, defaulting to 1 if out of bounds
		targetDim := 1
		if i < len(broadcastShape) {
			targetDim = broadcastShape[len(broadcastShape)-1-i]
		}
		sourceDim := 1
		if i < len(tensorShape) {
			sourceDim = tensorShape[len(tensorShape)-1-i]
		}

		// Check if dimensions are compatible
		if sourceDim != targetDim && sourceDim != 1 && targetDim != 1 {
			return fmt.Errorf("incompatible shapes for broadcasting: %v and %v", tensorShape, broadcastShape)
		}
	}

	// Broadcast is valid
	return nil
}

// alignShapes aligns the shapes of two tensors if possible
func alignShapes(s1 []int, s2 []int) ([]int, error) {
	// Initialize the aligned shape
	alignedShape := make([]int, len(s2))

	// Populate the aligned shape
	for s2Idx := len(s2) - 1; s2Idx >= 0; s2Idx-- {
		s1Idx := len(s2) - 1 - s2Idx

		// Check if s1Idx is out of bounds
		if s1Idx < len(s1) {
			// If not, use the value from s1
			alignedShape[s2Idx] = s1[s1Idx]
		} else {
			// If s1Idx is out of bounds, use 1
			alignedShape[s2Idx] = 1
		}
	}

	// Return the aligned shape
	return alignedShape, nil
}

// computeBroadcastStrides computes the strides of a broadcast
func computeBroadcastStrides(tensorShape []int, alignedShape []int, tensorStrides []int) []int {
	// Initialize the broadcast strides
	broadcastStrides := make([]int, len(alignedShape))

	// Copy the tensor strides to the broadcast strides
	copy(broadcastStrides, tensorStrides)
	// Set the last stride to 1
	broadcastStrides[len(broadcastStrides)-1] = 1

	// Set the broadcasted dimensions to 0
	for i := 0; i < len(alignedShape)-len(tensorShape); i++ {
		broadcastStrides[i] = 0
	}

	// Return the broadcast strides
	return broadcastStrides
}

// NewBroadcast creates a new broadcast struct from a tensor
func NewBroadcast(broadcastShape []int, tensor *TensorStruct) (*BroadcastStruct, error) {
	// Check broadcast validity
	if err := validBroadcast(broadcastShape, tensor); err != nil {
		return nil, err
	}

	// Align the shapes
	alignedShape, err := alignShapes(tensor.Shape(), broadcastShape)
	if err != nil {
		return nil, err
	}

	// Compute strides
	broadcastStrides := computeBroadcastStrides(tensor.Shape(), alignedShape, tensor.Stride())

	// Return the broadcast
	return &BroadcastStruct{
		broadcastShape: broadcastShape,
		tensor:         tensor,
		strides:        broadcastStrides,
	}, nil
}

// Shape returns the shape of the broadcast
func (b *BroadcastStruct) Shape() []int {
	return b.broadcastShape
}

// Stride returns the stride of the broadcast
func (b *BroadcastStruct) Stride() []int {
	return b.strides
}

// Data returns the data of the broadcast
func (b *BroadcastStruct) Data() []float64 {
	return b.tensor.data
}

// String returns a string representation of the broadcast
func (b *BroadcastStruct) String() string {
	switch len(b.broadcastShape) {
	case 0:
		// Scalar case
		return fmt.Sprintf("%.2f", b.tensor.data[0])
	case 1:
		// 1D case
		values := make([]string, b.broadcastShape[0])
		for i := 0; i < b.broadcastShape[0]; i++ {
			// If stride is 0, repeat the first value
			idx := 0
			if b.strides[0] != 0 {
				idx = i * b.strides[0]
			}
			values[i] = fmt.Sprintf("%.2f", b.tensor.data[idx])
		}
		return fmt.Sprintf("[%s]", strings.Join(values, " "))
	case 2:
		// 2D case
		rows := make([]string, b.broadcastShape[0])
		for i := 0; i < b.broadcastShape[0]; i++ {
			// Calculate row offset, if stride is 0, use 0 to repeat the same row
			rowOffset := 0
			if b.strides[0] != 0 {
				rowOffset = i * b.strides[0]
			}

			// Build each row
			row := make([]string, b.broadcastShape[1])
			for j := 0; j < b.broadcastShape[1]; j++ {
				// Calculate column offset, if stride is 0, use 0 to repeat the same value
				colOffset := 0
				if b.strides[1] != 0 {
					colOffset = j * b.strides[1]
				}
				row[j] = fmt.Sprintf("%.2f", b.tensor.data[rowOffset+colOffset])
			}
			rows[i] = "[" + strings.Join(row, " ") + "]"
		}
		return "[\n " + strings.Join(rows, "\n ") + "\n]"
	default:
		// For higher dimensions, show shape and first few values
		values := make([]string, b.broadcastShape[len(b.broadcastShape)-1])
		for i := 0; i < b.broadcastShape[len(b.broadcastShape)-1]; i++ {
			idx := 0
			if b.strides[len(b.strides)-1] != 0 {
				idx = i * b.strides[len(b.strides)-1]
			}
			values[i] = fmt.Sprintf("%.2f", b.tensor.data[idx])
		}
		return fmt.Sprintf("Broadcast(shape=%v, data=[%s])", b.broadcastShape, strings.Join(values, " "))
	}
}

// GetFlat returns the value at the given flat index
func (b *BroadcastStruct) GetFlat(idx int) float64 {
	return b.tensor.data[idx%len(b.tensor.data)]
}

// Get returns the value at the given index
func (b *BroadcastStruct) Get(idx []int) (float64, error) {
	// Check if enough indices are provided
	if len(idx) != len(b.broadcastShape) {
		return 0, fmt.Errorf("index dimension mismatch: expected %d, got %d", len(b.broadcastShape), len(idx))
	}

	// Check if indices are within bounds
	for i, v := range idx {
		if v < 0 || v >= b.broadcastShape[i] {
			return 0, fmt.Errorf("index out of bounds at dimension %d: %d >= %d", i, v, b.broadcastShape[i])
		}
	}

	// Convert broadcast index to tensor index
	tensorIdx := make([]int, len(b.tensor.Shape()))
	for i := range tensorIdx {
		if i < len(idx) {
			// Get corresponding tensor dimension size
			tensorDim := b.tensor.Shape()[i]
			if tensorDim == 1 {
				// Broadcasting case: tensor dim is 1
				tensorIdx[i] = 0
			} else {
				// Normal case: use index directly
				tensorIdx[i] = idx[i]
			}
		} else {
			// If index is out of bounds, set to 0
			tensorIdx[i] = 0
		}
	}

	// Convert tensor index to flat index
	flatIndex := 0
	for i, v := range tensorIdx {
		flatIndex += v * b.tensor.Stride()[i]
	}

	// Return value
	return b.tensor.Data()[flatIndex], nil
}

// ToTensor returns creates a new tensor from the broadcast
func (b *BroadcastStruct) ToTensor() (*TensorStruct, error) {
	// Calculate total size of broadcasted tensor
	size := 1
	for _, dim := range b.broadcastShape {
		size *= dim
	}

	// Create result data
	data := make([]float64, size)

	// Fill data using broadcast rules
	for i := 0; i < size; i++ {
		// Convert flat index to multi-dimensional indices
		indices := make([]int, len(b.broadcastShape))
		remaining := i
		for j := len(b.broadcastShape) - 1; j >= 0; j-- {
			indices[j] = remaining % b.broadcastShape[j]
			remaining = remaining / b.broadcastShape[j]
		}

		// Get value using Get method which handles broadcasting
		val, err := b.Get(indices)
		if err != nil {
			return nil, fmt.Errorf("error getting value at index %v: %v", indices, err)
		}
		data[i] = val
	}

	// Return new tensor
	return NewTensor(b.broadcastShape, data)
}
