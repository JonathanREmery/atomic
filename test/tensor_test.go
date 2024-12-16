package test

import (
	"reflect"
	"testing"

	"github.com/JonathanREmery/atomic.git/pkg/tensor"
)

// TestNewTensor tests the NewTensor function
func TestNewTensor(t *testing.T) {
	testCases := []struct {
		name           string
		shape          []int
		data           []float64
		expectedShape  []int
		expectedStride []int
		expectedErr    bool
	}{
		{"InvalidShape", []int{-1}, []float64{}, nil, nil, true},
		{"EmptyTensor", []int{}, []float64{}, nil, nil, true},
		{"ScalarTensor", []int{}, []float64{3.14}, []int{}, []int{}, false},
		{"OneDimSingleElement", []int{1}, []float64{3.14}, []int{1}, []int{1}, false},
		{"OneDimMultipleElements", []int{3}, []float64{1, 2, 3}, []int{3}, []int{1}, false},
		{"TwoDimSingleElement", []int{1, 1}, []float64{3.14}, []int{1, 1}, []int{1, 1}, false},
		{"TwoDimMultipleElements", []int{2, 2}, []float64{1, 2, 3, 4}, []int{2, 2}, []int{2, 1}, false},
		{"ThreeDimSingleElement", []int{1, 1, 1}, []float64{3.14}, []int{1, 1, 1}, []int{1, 1, 1}, false},
		{"ThreeDimMultipleElements", []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2}, []int{4, 2, 1}, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tensor, err := tensor.NewTensor(tc.shape, tc.data)

			if tc.expectedErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("Expected no error, got %v", err)
				return
			}

			if !reflect.DeepEqual(tensor.Shape(), tc.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tc.expectedShape, tensor.Shape())
			}

			if !reflect.DeepEqual(tensor.Stride(), tc.expectedStride) {
				t.Errorf("Expected stride %v, got %v", tc.expectedStride, tensor.Stride())
			}
		})
	}
}

// TestNewTensorBroadcast tests the auto-broadcasting feature of NewTensor
func TestNewTensorBroadcast(t *testing.T) {
	testCases := []struct {
		name           string
		shape          []int
		data           []float64
		expectedShape  []int
		expectedStride []int
		expectedData   []float64
		expectedErr    bool
	}{
		{
			name:           "ScalarTo2D",
			shape:          []int{2, 3},
			data:           []float64{5},
			expectedShape:  []int{2, 3},
			expectedStride: []int{3, 1},
			expectedData:   []float64{5, 5, 5, 5, 5, 5},
			expectedErr:    false,
		},
		{
			name:           "ScalarTo3D",
			shape:          []int{2, 2, 2},
			data:           []float64{3},
			expectedShape:  []int{2, 2, 2},
			expectedStride: []int{4, 2, 1},
			expectedData:   []float64{3, 3, 3, 3, 3, 3, 3, 3},
			expectedErr:    false,
		},
		{
			name:           "VectorTo2D_Invalid",
			shape:          []int{2, 3},
			data:           []float64{1, 2},
			expectedShape:  nil,
			expectedStride: nil,
			expectedData:   nil,
			expectedErr:    true,
		},
		{
			name:           "VectorTo2D_Valid",
			shape:          []int{3, 1},
			data:           []float64{1, 2, 3},
			expectedShape:  []int{3, 1},
			expectedStride: []int{1, 1},
			expectedData:   []float64{1, 2, 3},
			expectedErr:    false,
		},
		{
			name:           "EmptyShape",
			shape:          []int{},
			data:           []float64{42},
			expectedShape:  []int{},
			expectedStride: []int{},
			expectedData:   []float64{42},
			expectedErr:    false,
		},
		{
			name:           "ScalarToScalar",
			shape:          []int{1},
			data:           []float64{7},
			expectedShape:  []int{1},
			expectedStride: []int{1},
			expectedData:   []float64{7},
			expectedErr:    false,
		},
		{
			name:           "TooMuchData",
			shape:          []int{2, 2},
			data:           []float64{1, 2, 3, 4, 5},
			expectedShape:  nil,
			expectedStride: nil,
			expectedData:   nil,
			expectedErr:    true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tensor, err := tensor.NewTensor(tc.shape, tc.data)

			if tc.expectedErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("Expected no error, got %v", err)
				return
			}

			if !reflect.DeepEqual(tensor.Shape(), tc.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tc.expectedShape, tensor.Shape())
			}

			if !reflect.DeepEqual(tensor.Stride(), tc.expectedStride) {
				t.Errorf("Expected stride %v, got %v", tc.expectedStride, tensor.Stride())
			}

			if !reflect.DeepEqual(tensor.Data(), tc.expectedData) {
				t.Errorf("Expected data %v, got %v", tc.expectedData, tensor.Data())
			}
		})
	}
}

// TestAlignShapes tests the AlignShapes function
func TestAlignShapes(t *testing.T) {
	testCases := []struct {
		name     string
		shape1   []int
		shape2   []int
		expected []int
		hasError bool
	}{
		{"ScalarAndScalar1", []int{}, []int{}, nil, true},
		{"ScalarAndScalar2", []int{1}, []int{}, nil, true},
		{"ScalarAndScalar3", []int{}, []int{1}, nil, true},
		{"ScalarAndScalar4", []int{1}, []int{1}, nil, true},
		{"ScalarAndVector1", []int{}, []int{2}, []int{1}, false},
		{"ScalarAndVector2", []int{1}, []int{2}, nil, true},
		{"ScalarAndVector3", []int{1}, []int{4}, nil, true},
		{"ScalarAndVector4", []int{1}, []int{2, 4}, []int{1, 1}, false},
		{"VectorAndVector1", []int{2}, []int{2}, []int{2}, false},
		{"VectorAndVector2", []int{2}, []int{4}, nil, true},
		{"VectorAndMatrix1", []int{2}, []int{4, 2}, []int{1, 2}, false},
		{"VectorAndMatrix2", []int{8}, []int{4, 2}, nil, true},
		{"VectorAndMatrix3", []int{8}, []int{4, 8}, []int{1, 8}, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tensor.AlignShapes(tc.shape1, tc.shape2)

			if tc.hasError {
				if err == nil {
					t.Errorf("Expected an error, but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if !reflect.DeepEqual(result, tc.expected) {
					t.Errorf("Expected %v, but got %v", tc.expected, result)
				}
			}
		})
	}
}

// TestComputeStrides tests the ComputeStrides function
func TestComputeStrides(t *testing.T) {
	testCases := []struct {
		name     string
		shape    []int
		expected []int
	}{
		{"EmptyShape", []int{}, []int{}},
		{"ScalarShape", []int{1}, []int{1}},
		{"OneDimShape", []int{2}, []int{1}},
		{"TwoDimShape", []int{2, 2}, []int{2, 1}},
		{"ThreeDimShape", []int{2, 2, 2}, []int{4, 2, 1}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tensor.ComputeStrides(tc.shape)
			if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("Expected %v, but got %v", tc.expected, result)
			}
		})
	}
}

// TestComputeBroadcastStrides tests the ComputeBroadcastStrides function
func TestComputeBroadcastStrides(t *testing.T) {
	testCases := []struct {
		name     string
		shape1   []int
		shape2   []int
		strides1 []int
		expected []int
	}{
		{"EmptyShapes", []int{}, []int{}, []int{}, []int{}},
		{"ScalarShapes", []int{1}, []int{1}, []int{1}, []int{1}},
		{"OneDimShapes", []int{2}, []int{2}, []int{1}, []int{1}},
		{"TwoDimShapes", []int{2, 2}, []int{2, 2}, []int{2, 1}, []int{2, 1}},
		{"ThreeDimShapes", []int{2, 2, 2}, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tensor.ComputeBroadcastStrides(tc.shape1, tc.shape2, tc.strides1)
			if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("Expected %v, but got %v", tc.expected, result)
			}
		})
	}
}

// TestBroadcast tests the Broadcast function
func TestBroadcast(t *testing.T) {
	testCases := []struct {
		name          string
		inputShape    []int
		inputData     []float64
		targetShape   []int
		expectedShape []int
		expectedErr   bool
	}{
		{
			name:          "BroadcastScalarTo1D",
			inputShape:    []int{},
			inputData:     []float64{5.0},
			targetShape:   []int{3},
			expectedShape: []int{3},
			expectedErr:   false,
		},
		{
			name:          "Broadcast1DTo2D",
			inputShape:    []int{3},
			inputData:     []float64{1.0, 2.0, 3.0},
			targetShape:   []int{2, 3},
			expectedShape: []int{2, 3},
			expectedErr:   false,
		},
		{
			name:          "InvalidTargetShapeWithNegativeDimension",
			inputShape:    []int{2},
			inputData:     []float64{1.0, 2.0},
			targetShape:   []int{-1, 2},
			expectedShape: nil,
			expectedErr:   true,
		},
		{
			name:          "EmptyTargetShape",
			inputShape:    []int{2},
			inputData:     []float64{1.0, 2.0},
			targetShape:   []int{},
			expectedShape: nil,
			expectedErr:   true,
		},
		{
			name:          "SameShapeBroadcast",
			inputShape:    []int{2, 2},
			inputData:     []float64{1.0, 2.0, 3.0, 4.0},
			targetShape:   []int{2, 2},
			expectedShape: []int{2, 2},
			expectedErr:   false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create input tensor
			inputTensor, err := tensor.NewTensor(tc.inputShape, tc.inputData)
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}

			// Test broadcast
			result, err := tensor.Broadcast(inputTensor, tc.targetShape)

			// Check error cases
			if tc.expectedErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Check shape
			if !reflect.DeepEqual(result.Shape(), tc.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tc.expectedShape, result.Shape())
			}

			// Check data length matches shape size
			expectedSize := 1
			for _, dim := range tc.expectedShape {
				expectedSize *= dim
			}
			if len(result.Data()) != expectedSize {
				t.Errorf("Expected data length %d, got %d", expectedSize, len(result.Data()))
			}
		})
	}
}

// TestArithmeticOps tests the arithmetic operations (Add, Sub, Mul, Div)
func TestArithmeticOps(t *testing.T) {
	testCases := []struct {
		name          string
		t1Shape       []int
		t1Data        []float64
		t2Shape       []int
		t2Data        []float64
		expectedShape []int
		expectedAdd   []float64
		expectedSub   []float64
		expectedMul   []float64
		expectedDiv   []float64
		expectAddErr  bool
		expectSubErr  bool
		expectMulErr  bool
		expectDivErr  bool
	}{
		{
			name:          "SameShapeScalars",
			t1Shape:       []int{},
			t1Data:        []float64{4.0},
			t2Shape:       []int{},
			t2Data:        []float64{2.0},
			expectedShape: []int{},
			expectedAdd:   []float64{6.0},
			expectedSub:   []float64{2.0},
			expectedMul:   []float64{8.0},
			expectedDiv:   []float64{2.0},
			expectAddErr:  false,
			expectSubErr:  false,
			expectMulErr:  false,
			expectDivErr:  false,
		},
		{
			name:          "SameShape1D",
			t1Shape:       []int{3},
			t1Data:        []float64{1.0, 2.0, 3.0},
			t2Shape:       []int{3},
			t2Data:        []float64{4.0, 5.0, 6.0},
			expectedShape: []int{3},
			expectedAdd:   []float64{5.0, 7.0, 9.0},
			expectedSub:   []float64{-3.0, -3.0, -3.0},
			expectedMul:   []float64{4.0, 10.0, 18.0},
			expectedDiv:   []float64{0.25, 0.4, 0.5},
			expectAddErr:  false,
			expectSubErr:  false,
			expectMulErr:  false,
			expectDivErr:  false,
		},
		{
			name:          "BroadcastScalarTo1D",
			t1Shape:       []int{3},
			t1Data:        []float64{2.0, 4.0, 6.0},
			t2Shape:       []int{},
			t2Data:        []float64{2.0},
			expectedShape: []int{3},
			expectedAdd:   []float64{4.0, 6.0, 8.0},
			expectedSub:   []float64{0.0, 2.0, 4.0},
			expectedMul:   []float64{4.0, 8.0, 12.0},
			expectedDiv:   []float64{1.0, 2.0, 3.0},
			expectAddErr:  false,
			expectSubErr:  false,
			expectMulErr:  false,
			expectDivErr:  false,
		},
		{
			name:          "IncompatibleShapes",
			t1Shape:       []int{2},
			t1Data:        []float64{1.0, 2.0},
			t2Shape:       []int{3},
			t2Data:        []float64{1.0, 2.0, 3.0},
			expectedShape: nil,
			expectedAdd:   nil,
			expectedSub:   nil,
			expectedMul:   nil,
			expectedDiv:   nil,
			expectAddErr:  true,
			expectSubErr:  true,
			expectMulErr:  true,
			expectDivErr:  true,
		},
		{
			name:          "DivisionByZero",
			t1Shape:       []int{2},
			t1Data:        []float64{1.0, 2.0},
			t2Shape:       []int{2},
			t2Data:        []float64{1.0, 0.0},
			expectedShape: []int{2},
			expectedAdd:   []float64{2.0, 2.0},
			expectedSub:   []float64{0.0, 2.0},
			expectedMul:   []float64{1.0, 0.0},
			expectedDiv:   nil,
			expectAddErr:  false,
			expectSubErr:  false,
			expectMulErr:  false,
			expectDivErr:  true,
		},
		{
			name:          "Broadcast2DTo2D",
			t1Shape:       []int{2, 2},
			t1Data:        []float64{1.0, 2.0, 3.0, 4.0},
			t2Shape:       []int{1, 2},
			t2Data:        []float64{2.0, 3.0},
			expectedShape: []int{2, 2},
			expectedAdd:   []float64{3.0, 5.0, 5.0, 7.0},
			expectedSub:   []float64{-1.0, -1.0, 1.0, 1.0},
			expectedMul:   []float64{2.0, 6.0, 6.0, 12.0},
			expectedDiv:   []float64{0.5, 2.0 / 3.0, 1.5, 4.0 / 3.0},
			expectAddErr:  false,
			expectSubErr:  false,
			expectMulErr:  false,
			expectDivErr:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create tensors
			t1, err := tensor.NewTensor(tc.t1Shape, tc.t1Data)
			if err != nil {
				t.Fatalf("Failed to create first tensor: %v", err)
			}

			t2, err := tensor.NewTensor(tc.t2Shape, tc.t2Data)
			if err != nil {
				t.Fatalf("Failed to create second tensor: %v", err)
			}

			// Test Add
			resultAdd, err := t1.Add(t2)
			if tc.expectAddErr {
				if err == nil {
					t.Error("Expected Add error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected Add error: %v", err)
				} else {
					if !reflect.DeepEqual(resultAdd.Shape(), tc.expectedShape) {
						t.Errorf("Add: Expected shape %v, got %v", tc.expectedShape, resultAdd.Shape())
					}
					if !reflect.DeepEqual(resultAdd.Data(), tc.expectedAdd) {
						t.Errorf("Add: Expected data %v, got %v", tc.expectedAdd, resultAdd.Data())
					}
				}
			}

			// Test Sub
			resultSub, err := t1.Sub(t2)
			if tc.expectSubErr {
				if err == nil {
					t.Error("Expected Sub error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected Sub error: %v", err)
				} else {
					if !reflect.DeepEqual(resultSub.Shape(), tc.expectedShape) {
						t.Errorf("Sub: Expected shape %v, got %v", tc.expectedShape, resultSub.Shape())
					}
					if !reflect.DeepEqual(resultSub.Data(), tc.expectedSub) {
						t.Errorf("Sub: Expected data %v, got %v", tc.expectedSub, resultSub.Data())
					}
				}
			}

			// Test Mul
			resultMul, err := t1.Mul(t2)
			if tc.expectMulErr {
				if err == nil {
					t.Error("Expected Mul error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected Mul error: %v", err)
				} else {
					if !reflect.DeepEqual(resultMul.Shape(), tc.expectedShape) {
						t.Errorf("Mul: Expected shape %v, got %v", tc.expectedShape, resultMul.Shape())
					}
					if !reflect.DeepEqual(resultMul.Data(), tc.expectedMul) {
						t.Errorf("Mul: Expected data %v, got %v", tc.expectedMul, resultMul.Data())
					}
				}
			}

			// Test Div
			resultDiv, err := t1.Div(t2)
			if tc.expectDivErr {
				if err == nil {
					t.Error("Expected Div error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected Div error: %v", err)
				} else {
					if !reflect.DeepEqual(resultDiv.Shape(), tc.expectedShape) {
						t.Errorf("Div: Expected shape %v, got %v", tc.expectedShape, resultDiv.Shape())
					}
					if !almostEqual(resultDiv.Data(), tc.expectedDiv) {
						t.Errorf("Div: Expected data %v, got %v", tc.expectedDiv, resultDiv.Data())
					}
				}
			}
		})
	}
}

// TestView tests the View functionality
func TestView(t *testing.T) {
	// Create a test tensor
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, err := tensor.NewTensor([]int{2, 3}, data)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	// Test View method
	view, err := tensor.View([]int{2, 3})
	if err != nil {
		t.Fatalf("Failed to create view: %v", err)
	}
	if !reflect.DeepEqual(view.Shape(), []int{2, 3}) {
		t.Errorf("Expected shape %v, got %v", []int{2, 3}, view.Shape())
	}
	if !reflect.DeepEqual(view.Stride(), []int{3, 1}) {
		t.Errorf("Expected stride %v, got %v", []int{3, 1}, view.Stride())
	}
	if !reflect.DeepEqual(view.Data(), data) {
		t.Errorf("Expected data %v, got %v", data, view.Data())
	}

	// Test creating another view from the view
	newView, err := view.View([]int{3, 2})
	if err != nil {
		t.Fatalf("Failed to create new view: %v", err)
	}
	if !reflect.DeepEqual(newView.Shape(), []int{3, 2}) {
		t.Errorf("Expected shape %v, got %v", []int{3, 2}, newView.Shape())
	}
	if !reflect.DeepEqual(newView.Data(), data) {
		t.Errorf("Expected data %v, got %v", data, newView.Data())
	}

	// Test Reshape method
	reshapedView, err := view.Reshape([]int{6})
	if err != nil {
		t.Fatalf("Failed to reshape view: %v", err)
	}
	if !reflect.DeepEqual(reshapedView.Shape(), []int{6}) {
		t.Errorf("Expected shape %v, got %v", []int{6}, reshapedView.Shape())
	}
	if !reflect.DeepEqual(reshapedView.Data(), data) {
		t.Errorf("Expected data %v, got %v", data, reshapedView.Data())
	}

	// Test error cases
	testCases := []struct {
		name      string
		shape     []int
		expectErr bool
	}{
		{"InvalidShape", []int{7}, true},      // Total elements don't match
		{"NegativeShape", []int{-1, 6}, true}, // Negative dimension
		{"EmptyShape", []int{}, true},         // Empty shape
		{"ValidShape", []int{2, 3}, false},    // Original shape
		{"ValidReshape", []int{1, 6}, false},  // Valid reshape
	}

	// Run the test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := view.Reshape(tc.shape)
			if (err != nil) != tc.expectErr {
				t.Errorf("Test case %s: expected error %v, got %v", tc.name, tc.expectErr, err != nil)
			}
		})
	}

	// Test string representation
	expectedStr := "[\n [1.00 2.00 3.00]\n [4.00 5.00 6.00]\n]"
	if view.String() != expectedStr {
		t.Errorf("Expected string representation %q, got %q", expectedStr, view.String())
	}
}

// almostEqual compares two float64 slices with a small epsilon for floating point comparison
func almostEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	const epsilon = 1e-10
	for i := range a {
		if diff := a[i] - b[i]; diff > epsilon || diff < -epsilon {
			return false
		}
	}
	return true
}
