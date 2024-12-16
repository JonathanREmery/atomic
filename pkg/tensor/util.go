package tensor

import (
	"fmt"
	"strings"
)

// formatSlice formats a slice of floats as a string
func formatSlice(data []float64) string {
	// Format the slice
	formatted := make([]string, len(data))
	for i, v := range data {
		formatted[i] = fmt.Sprintf("%.2f", v)
	}

	// Join the formatted slice
	return strings.Join(formatted, " ")
}

// ComputeStrides computes the stride of a tensor given its shape
func ComputeStrides(shape []int) []int {
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

// ComputeBroadcastStrides computes the stride of a tensor given its shape, broadcast shape, and strides
func ComputeBroadcastStrides(originalShape []int, alignedShape []int, originalStrides []int) []int {
	// If either shape is empty, return the empty strides
	if len(originalShape) == 0 || len(alignedShape) == 0 {
		return []int{}
	}

	// Initialize the broadcast strides
	broadcastStrides := make([]int, len(alignedShape))

	// Calculate the broadcast strides
	for i := range alignedShape {
		if i < len(originalShape) && alignedShape[i] == originalShape[i] {
			// Dimensions match, so use the original strides
			broadcastStrides[i] = originalStrides[i]
		} else {
			// Dimension is broadcasted, so set the stride to 0
			broadcastStrides[i] = 0
		}
	}

	// Return the broadcast strides
	return broadcastStrides
}

// AlignShapes aligns the shapes of two tensors
func AlignShapes(shape1 []int, shape2 []int) ([]int, error) {
	// Check if shape1 is valid
	for _, dim := range shape1 {
		if dim < 0 {
			return nil, fmt.Errorf("invalid shape: %v", shape1)
		}
	}

	// Check if shape2 is valid
	for _, dim := range shape2 {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid shape: %v", shape2)
		}
	}

	// Check if both shapes are scalars
	shape1IsScalar := len(shape1) == 0 || (len(shape1) == 1 && (shape1[0] == 0 || shape1[0] == 1))
	shape2IsScalar := len(shape2) == 0 || (len(shape2) == 1 && (shape2[0] == 0 || shape2[0] == 1))
	if shape1IsScalar && shape2IsScalar {
		return nil, fmt.Errorf("cannot align two scalar tensors")
	}

	// Check if shapes have the same rank
	if len(shape1) == len(shape2) {
		// Check if the elements are the same
		for i := 0; i < len(shape1); i++ {
			// If they aren't the same, return an error
			if shape1[i] != shape2[i] {
				return nil, fmt.Errorf("cannot align shapes %v and %v", shape1, shape2)
			}
		}

		// If they are the same, return the shape
		return shape1, nil
	}

	// Get the shape to align
	shapeToAlign := shape1
	if len(shape1) > len(shape2) {
		shapeToAlign = shape2
	}

	// Get the shape to align to
	shapeToAlignTo := shape1
	if len(shape1) < len(shape2) {
		shapeToAlignTo = shape2
	}

	// Get the max rank
	maxRank := len(shapeToAlignTo)

	// Initialize the aligned shape
	alignedShape := make([]int, maxRank)

	// Align the shapes
	for i := maxRank - 1; i >= 0; i-- {
		iInv := maxRank - 1 - i
		if iInv < len(shapeToAlign) {
			alignedShape[i] = shapeToAlign[iInv]
		} else {
			alignedShape[i] = 1
		}
	}

	// Check if the aligned shape is valid
	for idx, dim := range alignedShape {
		if dim != shapeToAlignTo[idx] && dim != 1 {
			return nil, fmt.Errorf("cannot align shapes %v and %v", shape1, shape2)
		}
	}

	// Return the aligned shape
	return alignedShape, nil
}

// ValidReshape checks if a reshape is valid
func ValidReshape(currentShape []int, desiredShape []int) bool {
	// Calculate the current size of the data
	currentSize := 1
	for _, dim := range currentShape {
		currentSize *= dim
	}

	// Calculate the desired size of the data
	desiredSize := 1
	for _, dim := range desiredShape {
		desiredSize *= dim
	}

	// Check if the current size is equal to the desired size
	return currentSize == desiredSize
}
