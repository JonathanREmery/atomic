package tensor

import (
	"reflect"
	"testing"
)

// checkEqual checks if the expected and got values are equal
func checkEqual(t *testing.T, name string, expected, got interface{}) {
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("Expected %s %v, got %v", name, expected, got)
	}
}

// almostEqual compares two float64 slices with a small epsilon for floating point comparison
func almostEqual(a, b []float64) bool {
	// Check if the slices have different lengths
	if len(a) != len(b) {
		return false
	}

	// Check if the slices are almost equal
	const epsilon = 1e-6  // More lenient epsilon for division results
	for i := range a {
		if diff := a[i] - b[i]; diff > epsilon || diff < -epsilon {
			// The slices are not almost equal
			return false
		}
	}

	// The slices are almost equal
	return true
}

// mustNewTensor creates a new tensor or fails the test
func mustNewTensor(t *testing.T, shape []int, data []float64) *TensorStruct {
	tensor, err := NewTensor(shape, data)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	return tensor
}

// TestNewTensor tests the NewTensor function
func TestNewTensor(t *testing.T) {
	testCases := []struct {
		name           string
		shape          []int
		data           []float64
		expectedShape  []int
		expectedStride []int
		expectedData   []float64
		expectedRank   int
		expectedErr    bool
	}{
		{"InvalidShape", []int{-1}, []float64{}, nil, nil, nil, 0, true},
		{"EmptyTensor", []int{}, []float64{}, nil, nil, nil, 0, true},
		{"ScalarTensor", []int{}, []float64{3.14}, []int{}, []int{}, []float64{3.14}, 0, false},
		{"OneDimSingleElement", []int{1}, []float64{3.14}, []int{1}, []int{1}, []float64{3.14}, 1, false},
		{"OneDimMultipleElements", []int{3}, []float64{1, 2, 3}, []int{3}, []int{1}, []float64{1, 2, 3}, 1, false},
		{"TwoDimSingleElement", []int{1, 1}, []float64{3.14}, []int{1, 1}, []int{1, 1}, []float64{3.14}, 2, false},
		{"TwoDimMultipleElements", []int{2, 2}, []float64{1, 2, 3, 4}, []int{2, 2}, []int{2, 1}, []float64{1, 2, 3, 4}, 2, false},
		{"ThreeDimSingleElement", []int{1, 1, 1}, []float64{3.14}, []int{1, 1, 1}, []int{1, 1, 1}, []float64{3.14}, 3, false},
		{"ThreeDimMultipleElements", []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, 3, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tensor, err := NewTensor(tc.shape, tc.data)

			if tc.expectedErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("Expected no error, got %v", err)
			}

			checkEqual(t, "Shape", tc.expectedShape, tensor.Shape())
			checkEqual(t, "Stride", tc.expectedStride, tensor.Stride())
			checkEqual(t, "Data", tc.expectedData, tensor.Data())
			checkEqual(t, "Rank", tc.expectedRank, tensor.Rank())
		})
	}
}

// TestArithmeticOps tests the arithmetic operations (Add, Sub, Mul, Div)
func TestArithmeticOps(t *testing.T) {
	testCases := []struct {
		name          string
		t1, t2        *TensorStruct
		expectedShape []int
		expectedAdd   []float64
		expectedSub   []float64
		expectedMul   []float64
		expectedDiv   []float64
		expectErr     bool
	}{
		{
			name:          "SameShapeScalars",
			t1:            mustNewTensor(t, []int{}, []float64{4.0}),
			t2:            mustNewTensor(t, []int{}, []float64{2.0}),
			expectedShape: []int{},
			expectedAdd:   []float64{6.0},
			expectedSub:   []float64{2.0},
			expectedMul:   []float64{8.0},
			expectedDiv:   []float64{2.0},
		},
		{
			name:          "SameShape1D",
			t1:            mustNewTensor(t, []int{3}, []float64{1.0, 2.0, 3.0}),
			t2:            mustNewTensor(t, []int{3}, []float64{4.0, 5.0, 6.0}),
			expectedShape: []int{3},
			expectedAdd:   []float64{5.0, 7.0, 9.0},
			expectedSub:   []float64{-3.0, -3.0, -3.0},
			expectedMul:   []float64{4.0, 10.0, 18.0},
			expectedDiv:   []float64{0.25, 0.4, 0.5},
		},
		{
			name:          "SameShape2D",
			t1:            mustNewTensor(t, []int{2, 2}, []float64{1.0, 2.0, 3.0, 4.0}),
			t2:            mustNewTensor(t, []int{2, 2}, []float64{5.0, 6.0, 7.0, 8.0}),
			expectedShape: []int{2, 2},
			expectedAdd:   []float64{6.0, 8.0, 10.0, 12.0},
			expectedSub:   []float64{-4.0, -4.0, -4.0, -4.0},
			expectedMul:   []float64{5.0, 12.0, 21.0, 32.0},
			expectedDiv:   []float64{0.2, 0.333333, 0.428571, 0.5},
		},
		{
			name:          "DifferentShape",
			t1:            mustNewTensor(t, []int{2}, []float64{1.0, 2.0}),
			t2:            mustNewTensor(t, []int{3}, []float64{3.0, 4.0, 5.0}),
			expectedShape: nil,
			expectErr:     true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Test Add
			add, err := tc.t1.Add(tc.t2)
			if (err != nil) != tc.expectErr {
				t.Errorf("Add: expected error %v, got %v", tc.expectErr, err != nil)
			}
			if err == nil {
				checkEqual(t, "Add Shape", tc.expectedShape, add.Shape())
				if !almostEqual(tc.expectedAdd, add.Data()) {
					t.Errorf("Add: expected %v, got %v", tc.expectedAdd, add.Data())
				}
			}

			// Test Sub
			sub, err := tc.t1.Sub(tc.t2)
			if (err != nil) != tc.expectErr {
				t.Errorf("Sub: expected error %v, got %v", tc.expectErr, err != nil)
			}
			if err == nil {
				checkEqual(t, "Sub Shape", tc.expectedShape, sub.Shape())
				if !almostEqual(tc.expectedSub, sub.Data()) {
					t.Errorf("Sub: expected %v, got %v", tc.expectedSub, sub.Data())
				}
			}

			// Test Mul
			mul, err := tc.t1.Mul(tc.t2)
			if (err != nil) != tc.expectErr {
				t.Errorf("Mul: expected error %v, got %v", tc.expectErr, err != nil)
			}
			if err == nil {
				checkEqual(t, "Mul Shape", tc.expectedShape, mul.Shape())
				if !almostEqual(tc.expectedMul, mul.Data()) {
					t.Errorf("Mul: expected %v, got %v", tc.expectedMul, mul.Data())
				}
			}

			// Test Div
			div, err := tc.t1.Div(tc.t2)
			if (err != nil) != tc.expectErr {
				t.Errorf("Div: expected error %v, got %v", tc.expectErr, err != nil)
			}
			if err == nil {
				checkEqual(t, "Div Shape", tc.expectedShape, div.Shape())
				if !almostEqual(tc.expectedDiv, div.Data()) {
					t.Errorf("Div: expected %v, got %v", tc.expectedDiv, div.Data())
				}
			}
		})
	}
}
