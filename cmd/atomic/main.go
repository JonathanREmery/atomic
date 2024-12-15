package main

import (
	"fmt"

	"github.com/JonathanREmery/atomic.git/pkg/tensor"
)

func main() {
	t1, err := tensor.NewTensor([]int{3, 3}, []float64{1, 2, 3})
	if err != nil {
		panic(err)
	}

	fmt.Printf("Shape: %v\n", t1.Shape())
	fmt.Printf("Stride: %v\n", t1.Stride())
	fmt.Printf("Data: %v\n", t1.Data())
}
