package main

import (
	"fmt"

	"github.com/JonathanREmery/atomic.git/pkg/tensor"
)

func main() {
	x, err := tensor.NewTensor([]int{3}, []float64{1, 2, 3})
	if err != nil {
		panic(err)
	}

	W, err := tensor.NewTensor([]int{2, 3}, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	if err != nil {
		panic(err)
	}

	b := tensor.NewScalar(1)

	fmt.Printf("x: %v\n", x)
	fmt.Printf("x.shape: %v\n", x.Shape())
	fmt.Printf("x.stride: %v\n\n", x.Stride())

	fmt.Printf("W: %v\n", W)
	fmt.Printf("W.shape: %v\n", W.Shape())
	fmt.Printf("W.stride: %v\n\n", W.Stride())

	fmt.Printf("b: %v\n", b)
	fmt.Printf("b.shape: %v\n", b.Shape())
	fmt.Printf("b.stride: %v\n\n", b.Stride())

	Wx, err := W.Mul(x)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Wx: %v\n", Wx)
	fmt.Printf("Wx.shape: %v\n", Wx.Shape())
	fmt.Printf("Wx.stride: %v\n\n", Wx.Stride())

	WxPLUSb, err := Wx.Add(b)
	if err != nil {
		panic(err)
	}

	fmt.Printf("WxPLUSb: %v\n", WxPLUSb)
	fmt.Printf("WxPLUSb.shape: %v\n", WxPLUSb.Shape())
	fmt.Printf("WxPLUSb.stride: %v\n\n", WxPLUSb.Stride())
}
