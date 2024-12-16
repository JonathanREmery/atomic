package tensor

import (
	"fmt"
	"strings"
)

// formatSlice formats a slice of floats as a string
func formatSlice(data []float64) string {
	// Format the slice
	formatted := make([]string, len(data))
	for i, v := range data {
		formatted[i] = fmt.Sprintf("%.2f", v)
	}

	// Join the formatted slice
	return strings.Join(formatted, " ")
}
