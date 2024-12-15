package main

import (
	"fmt"

	"github.com/JonathanREmery/atomic.git/pkg/tensor"
)

func main() {
	t1, _ := tensor.NewTensor([]int{3, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	t2, _ := tensor.NewTensor([]int{}, []float64{10})

	fmt.Printf("t1: %v\n", t1)
	fmt.Printf("t2: %v\n", t2)

	t3, err := t1.Add(t2)
	if err != nil {
		panic(err)
	}

	fmt.Printf("t3: %v\n", t3)

	t4, err := t1.Sub(t2)
	if err != nil {
		panic(err)
	}

	fmt.Printf("t4: %v\n", t4)

	t5, err := t1.Mul(t2)
	if err != nil {
		panic(err)
	}

	fmt.Printf("t5: %v\n", t5)

	t6, err := t1.Div(t2)
	if err != nil {
		panic(err)
	}

	fmt.Printf("t6: %v\n", t6)

	t7, _ := tensor.NewTensor([]int{3, 3}, []float64{1, 2, 3})

	fmt.Printf("t7: %v\n", t7)
}
