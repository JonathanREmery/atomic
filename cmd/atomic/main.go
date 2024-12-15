package main

import (
	"fmt"

	"github.com/JonathanREmery/atomic.git/pkg/tensor"
)

func main() {
	shape1 := []int{2}
	shape2 := []int{4, 2}

	fmt.Printf("shape1: %v\n", shape1)
	fmt.Printf("shape2: %v\n", shape2)

	alignedShape, err := tensor.AlignShapes(shape1, shape2)
	if err != nil {
		panic(err)
	}

	fmt.Printf("alignedShape: %v\n", alignedShape)
}
