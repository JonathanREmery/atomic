package tensor

import "fmt"

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
