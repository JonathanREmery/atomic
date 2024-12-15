package test

import (
	"reflect"
	"testing"

	"github.com/JonathanREmery/atomic.git/pkg/tensor"
)

// TestNewTensor tests the NewTensor function
func TestNewTensor(t *testing.T) {
	// Try to create a tensor with an invalid shape
	_, err := tensor.NewTensor([]int{-1}, []float64{})
	if err == nil {
		t.Errorf("Expected error, got nil")
	}

	// Try to create an empty tensor
	_, err = tensor.NewTensor([]int{}, []float64{})
	if err == nil {
		t.Errorf("Expected error, got nil")
	}

	// Create a 0 dimensional tensor
	t2, err := tensor.NewTensor([]int{}, []float64{3.14})
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Check that the shape is correct
	if !reflect.DeepEqual(t2.Shape(), []int{}) {
		t.Errorf("Expected shape to be %v, got %v", []int{}, t2.Shape())
	}

	// Check that the stride is correct
	if !reflect.DeepEqual(t2.Stride(), []int{}) {
		t.Errorf("Expected stride to be %v, got %v", []int{}, t2.Stride())
	}

	// Create a 1 dimensional tensor, with a single element
	t3, err := tensor.NewTensor([]int{1}, []float64{3.14})
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Check that the shape is correct
	if !reflect.DeepEqual(t3.Shape(), []int{1}) {
		t.Errorf("Expected shape to be %v, got %v", []int{3}, t3.Shape())
	}

	// Check that the stride is correct
	if !reflect.DeepEqual(t3.Stride(), []int{1}) {
		t.Errorf("Expected stride to be %v, got %v", []int{1}, t3.Stride())
	}

	// Create a 1 dimensional tensor, with multiple elements
	t4, err := tensor.NewTensor([]int{3}, []float64{1, 2, 3})
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Check that the shape is correct
	if !reflect.DeepEqual(t4.Shape(), []int{3}) {
		t.Errorf("Expected shape to be %v, got %v", []int{3}, t4.Shape())
	}

	// Check that the stride is correct
	if !reflect.DeepEqual(t4.Stride(), []int{1}) {
		t.Errorf("Expected stride to be %v, got %v", []int{1}, t4.Stride())
	}

	// Create a 2 dimensional tensor, with a single element
	t5, err := tensor.NewTensor([]int{1, 1}, []float64{3.14})
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Check that the shape is correct
	if !reflect.DeepEqual(t5.Shape(), []int{1, 1}) {
		t.Errorf("Expected shape to be %v, got %v", []int{1, 1}, t5.Shape())
	}

	// Check that the stride is correct
	if !reflect.DeepEqual(t5.Stride(), []int{1, 1}) {
		t.Errorf("Expected stride to be %v, got %v", []int{1, 1}, t5.Stride())
	}

	// Create a 2 dimensional tensor, with multiple elements
	t6, err := tensor.NewTensor([]int{2, 2}, []float64{1, 2, 3, 4})
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Check that the shape is correct
	if !reflect.DeepEqual(t6.Shape(), []int{2, 2}) {
		t.Errorf("Expected shape to be %v, got %v", []int{2, 2}, t6.Shape())
	}

	// Check that the stride is correct
	if !reflect.DeepEqual(t6.Stride(), []int{2, 1}) {
		t.Errorf("Expected stride to be %v, got %v", []int{2, 1}, t6.Stride())
	}

	// Create a 3 dimensional tensor, with a single element
	t7, err := tensor.NewTensor([]int{1, 1, 1}, []float64{3.14})
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Check that the shape is correct
	if !reflect.DeepEqual(t7.Shape(), []int{1, 1, 1}) {
		t.Errorf("Expected shape to be %v, got %v", []int{1, 1, 1}, t7.Shape())
	}

	// Check that the stride is correct
	if !reflect.DeepEqual(t7.Stride(), []int{1, 1, 1}) {
		t.Errorf("Expected stride to be %v, got %v", []int{1, 1, 1}, t7.Stride())
	}

	// Create a 3 dimensional tensor, with multiple elements
	t8, err := tensor.NewTensor([]int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Check that the shape is correct
	if !reflect.DeepEqual(t8.Shape(), []int{2, 2, 2}) {
		t.Errorf("Expected shape to be %v, got %v", []int{2, 2, 2}, t8.Shape())
	}

	// Check that the stride is correct
	if !reflect.DeepEqual(t8.Stride(), []int{4, 2, 1}) {
		t.Errorf("Expected stride to be %v, got %v", []int{4, 2, 1}, t8.Stride())
	}
}
