package main

import (
	"fmt"

	"github.com/JonathanREmery/atomic.git/pkg/tensor"
)

func main() {
	t1, _ := tensor.NewTensor([]int{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	fmt.Printf("t1: %v\n", t1)

	/*
		[
		  1 2
		  3 4
		  5 6
		]
	*/

	t2, _ := t1.View([]int{2, 3})
	fmt.Printf("t2: %v\n", t2)

	/*
		[
		  1 2 3
		  4 5 6
		]
	*/
}
