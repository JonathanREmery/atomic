package test

import (
	"reflect"
	"testing"

	"github.com/JonathanREmery/atomic.git/pkg/tensor"
)

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
