package tensor

import (
	"testing"
)

// TestView tests the View functionality
func TestView(t *testing.T) {
	// Create a test tensor
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, err := NewTensor([]int{2, 3}, data)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	// Test View method
	t.Run("View", func(t *testing.T) {
		view, err := tensor.View([]int{2, 3})
		if err != nil {
			t.Fatalf("Failed to create view: %v", err)
		}
		checkEqual(t, "Shape", []int{2, 3}, view.Shape())
		checkEqual(t, "Stride", []int{3, 1}, view.Stride())
		checkEqual(t, "Data", data, view.Data())
	})

	// Test creating another view from the view
	t.Run("ViewFromView", func(t *testing.T) {
		view, _ := tensor.View([]int{2, 3})
		newView, err := view.View([]int{3, 2})
		if err != nil {
			t.Fatalf("Failed to create new view: %v", err)
		}
		checkEqual(t, "Shape", []int{3, 2}, newView.Shape())
		checkEqual(t, "Data", data, newView.Data())
	})

	// Test Reshape method
	t.Run("Reshape", func(t *testing.T) {
		view, _ := tensor.View([]int{2, 3})
		reshapedView, err := view.Reshape([]int{6})
		if err != nil {
			t.Fatalf("Failed to reshape view: %v", err)
		}
		checkEqual(t, "Shape", []int{6}, reshapedView.Shape())
		checkEqual(t, "Data", data, reshapedView.Data())
	})

	// Test error cases
	testCases := []struct {
		name      string
		shape     []int
		expectErr bool
	}{
		{"InvalidShape", []int{7}, true},
		{"NegativeShape", []int{-1, 6}, true},
		{"EmptyShape", []int{}, true},
		{"ValidShape", []int{2, 3}, false},
		{"ValidReshape", []int{1, 6}, false},
	}

	t.Run("ErrorCases", func(t *testing.T) {
		view, _ := tensor.View([]int{2, 3})
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				_, err := view.Reshape(tc.shape)
				if (err != nil) != tc.expectErr {
					t.Errorf("Expected error %v, got %v", tc.expectErr, err != nil)
				}
			})
		}
	})

	// Test string representation
	t.Run("StringRepresentation", func(t *testing.T) {
		view, _ := tensor.View([]int{2, 3})
		expectedStr := "[\n [1.00 2.00 3.00]\n [4.00 5.00 6.00]\n]"
		if view.String() != expectedStr {
			t.Errorf("Expected %q, got %q", expectedStr, view.String())
		}
	})
}

// Additional view-specific test cases
func TestViewEdgeCases(t *testing.T) {
	// Test viewing a scalar
	t.Run("ViewScalar", func(t *testing.T) {
		scalar := mustNewTensor(t, []int{}, []float64{42})
		view, err := scalar.View([]int{1})
		if err != nil {
			t.Fatalf("Failed to create view from scalar: %v", err)
		}
		checkEqual(t, "Shape", []int{1}, view.Shape())
		checkEqual(t, "Data", []float64{42}, view.Data())
	})

	// Test viewing with same shape
	t.Run("ViewSameShape", func(t *testing.T) {
		tensor := mustNewTensor(t, []int{2, 2}, []float64{1, 2, 3, 4})
		view, err := tensor.View([]int{2, 2})
		if err != nil {
			t.Fatalf("Failed to create view with same shape: %v", err)
		}
		checkEqual(t, "Shape", []int{2, 2}, view.Shape())
		checkEqual(t, "Data", []float64{1, 2, 3, 4}, view.Data())
	})

	// Test invalid view shapes
	t.Run("InvalidViews", func(t *testing.T) {
		tensor := mustNewTensor(t, []int{2, 2}, []float64{1, 2, 3, 4})
		
		testCases := []struct {
			name  string
			shape []int
		}{
			{"TooManyElements", []int{2, 2, 2}},
			{"TooFewElements", []int{1}},
			{"ZeroDimension", []int{0, 4}},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				_, err := tensor.View(tc.shape)
				if err == nil {
					t.Error("Expected error for invalid view shape, got nil")
				}
			})
		}
	})
}
