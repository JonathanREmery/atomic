package main

import (
	"fmt"

	"github.com/JonathanREmery/atomic.git/pkg/tensor"
)

func main() {
	t1, _ := tensor.NewTensor([]int{1, 1}, []float64{3.14})

	fmt.Printf("t1: %v\n", t1)

	t2, _ := tensor.Broadcast(t1, []int{2, 2})

	fmt.Printf("t3: %v\n", t2)
}
