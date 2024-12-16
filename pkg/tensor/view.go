package tensor

import (
	"fmt"
	"strings"
)

// ViewStruct represents a view of a tensor with its own shape and stride
type ViewStruct struct {
	shape  []int
	stride []int
	tensor *TensorStruct
}

// View interface extends the Tensor interface
type View interface {
	Shape() []int
	Stride() []int
	Data() []float64

	String() string

	View(shape []int) (*ViewStruct, error)
	Reshape(shape []int) (*ViewStruct, error)
}

// NewView creates a new ViewStruct from a TensorStruct
func NewView(tensor *TensorStruct) *ViewStruct {
	return &ViewStruct{
		shape:  tensor.shape,
		stride: tensor.stride,
		tensor: tensor,
	}
}

// Shape returns the shape of the view
func (v *ViewStruct) Shape() []int {
	return v.shape
}

// Stride returns the stride of the view
func (v *ViewStruct) Stride() []int {
	return v.stride
}

// Data returns the underlying data of the tensor
func (v *ViewStruct) Data() []float64 {
	return v.tensor.Data()
}

// String returns a string representation of the view
func (v *ViewStruct) String() string {
	switch len(v.shape) {
	case 0:
		return fmt.Sprintf("%.2f", v.tensor.data[0])
	case 1:
		return fmt.Sprintf("[%s]", formatSlice(v.tensor.data))
	case 2:
		rows := make([]string, v.shape[0])
		for i := 0; i < v.shape[0]; i++ {
			row := make([]string, v.shape[1])
			for j := 0; j < v.shape[1]; j++ {
				row[j] = fmt.Sprintf("%.2f", v.tensor.data[i*v.stride[0]+j*v.stride[1]])
			}
			rows[i] = "[" + strings.Join(row, " ") + "]"
		}
		return "[\n " + strings.Join(rows, "\n ") + "\n]"
	default:
		return fmt.Sprintf("Tensor(shape=%v, data=[%s])", v.shape, formatSlice(v.tensor.data))
	}
}

// View returns a view of the tensor with the given shape
func (v *ViewStruct) View(shape []int) (*ViewStruct, error) {
	// Create a new view with the given shape
	return NewView(v.tensor).Reshape(shape)
}

// Reshape reshapes the view to the given shape
func (v *ViewStruct) Reshape(shape []int) (*ViewStruct, error) {
	// Check if the reshape is valid
	if !ValidReshape(v.shape, shape) {
		return nil, fmt.Errorf("invalid reshape")
	}

	// Create a new view with the new shape
	return &ViewStruct{
		shape:  shape,
		stride: ComputeStrides(shape),
		tensor: v.tensor,
	}, nil
}
