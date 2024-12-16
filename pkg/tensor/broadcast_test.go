package tensor

import (
	"reflect"
	"testing"
)

func TestValidBroadcast(t *testing.T) {
	tests := []struct {
		name           string
		broadcastShape []int
		tensorShape    []int
		data          []float64
		wantErr       bool
	}{
		{
			name:           "valid broadcast - same shape",
			broadcastShape: []int{2, 3},
			tensorShape:    []int{2, 3},
			data:          []float64{1, 2, 3, 4, 5, 6},
			wantErr:       false,
		},
		{
			name:           "valid broadcast - scalar to shape",
			broadcastShape: []int{2, 3},
			tensorShape:    []int{1, 1},
			data:          []float64{1},
			wantErr:       false,
		},
		{
			name:           "valid broadcast - row to matrix",
			broadcastShape: []int{3, 4},
			tensorShape:    []int{1, 4},
			data:          []float64{1, 2, 3, 4},
			wantErr:       false,
		},
		{
			name:           "invalid broadcast - incompatible shapes",
			broadcastShape: []int{2, 3},
			tensorShape:    []int{2, 4},
			data:          []float64{1, 2, 3, 4, 5, 6, 7, 8},
			wantErr:       true,
		},
		{
			name:           "invalid broadcast - empty shape",
			broadcastShape: []int{},
			tensorShape:    []int{2, 3},
			data:          []float64{1, 2, 3, 4, 5, 6},
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensor(tt.tensorShape, tt.data)
			if err != nil {
				t.Fatalf("Failed to create tensor: %v", err)
			}
			err = validBroadcast(tt.broadcastShape, tensor)
			if (err != nil) != tt.wantErr {
				t.Errorf("validBroadcast() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestNewBroadcast(t *testing.T) {
	tests := []struct {
		name           string
		broadcastShape []int
		tensorShape    []int
		data          []float64
		wantShape     []int
		wantErr       bool
	}{
		{
			name:           "broadcast scalar to 2x2",
			broadcastShape: []int{2, 2},
			tensorShape:    []int{1, 1},
			data:          []float64{5},
			wantShape:     []int{2, 2},
			wantErr:       false,
		},
		{
			name:           "broadcast row to matrix",
			broadcastShape: []int{3, 2},
			tensorShape:    []int{1, 2},
			data:          []float64{1, 2},
			wantShape:     []int{3, 2},
			wantErr:       false,
		},
		{
			name:           "broadcast column to matrix",
			broadcastShape: []int{2, 3},
			tensorShape:    []int{2, 1},
			data:          []float64{1, 2},
			wantShape:     []int{2, 3},
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensor(tt.tensorShape, tt.data)
			if err != nil {
				t.Fatalf("Failed to create tensor: %v", err)
			}
			broadcast, err := NewBroadcast(tt.broadcastShape, tensor)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewBroadcast() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err == nil {
				if !reflect.DeepEqual(broadcast.Shape(), tt.wantShape) {
					t.Errorf("NewBroadcast() shape = %v, want %v", broadcast.Shape(), tt.wantShape)
				}
			}
		})
	}
}

func TestBroadcastGet(t *testing.T) {
	tests := []struct {
		name           string
		broadcastShape []int
		tensorShape    []int
		data          []float64
		indices       []int
		want          float64
		wantErr       bool
	}{
		{
			name:           "get from scalar broadcast",
			broadcastShape: []int{2, 2},
			tensorShape:    []int{1, 1},
			data:          []float64{5},
			indices:       []int{1, 1},
			want:          5,
			wantErr:       false,
		},
		{
			name:           "get from row broadcast",
			broadcastShape: []int{3, 2},
			tensorShape:    []int{1, 2},
			data:          []float64{1, 2},
			indices:       []int{1, 1},
			want:          2,
			wantErr:       false,
		},
		{
			name:           "invalid indices",
			broadcastShape: []int{2, 2},
			tensorShape:    []int{1, 1},
			data:          []float64{5},
			indices:       []int{2, 2},
			want:          0,
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensor(tt.tensorShape, tt.data)
			if err != nil {
				t.Fatalf("Failed to create tensor: %v", err)
			}
			broadcast, err := NewBroadcast(tt.broadcastShape, tensor)
			if err != nil {
				t.Fatalf("Failed to create broadcast: %v", err)
			}

			got, err := broadcast.Get(tt.indices)
			if (err != nil) != tt.wantErr {
				t.Errorf("Broadcast.Get() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err == nil && got != tt.want {
				t.Errorf("Broadcast.Get() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBroadcastToTensor(t *testing.T) {
	tests := []struct {
		name           string
		broadcastShape []int
		tensorShape    []int
		data          []float64
		wantData      []float64
		wantErr       bool
	}{
		{
			name:           "scalar to 2x2",
			broadcastShape: []int{2, 2},
			tensorShape:    []int{1, 1},
			data:          []float64{5},
			wantData:      []float64{5, 5, 5, 5},
			wantErr:       false,
		},
		{
			name:           "row to 2x2",
			broadcastShape: []int{2, 2},
			tensorShape:    []int{1, 2},
			data:          []float64{1, 2},
			wantData:      []float64{1, 2, 1, 2},
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensor(tt.tensorShape, tt.data)
			if err != nil {
				t.Fatalf("Failed to create tensor: %v", err)
			}
			broadcast, err := NewBroadcast(tt.broadcastShape, tensor)
			if err != nil {
				t.Fatalf("Failed to create broadcast: %v", err)
			}

			result, err := broadcast.ToTensor()
			if (err != nil) != tt.wantErr {
				t.Errorf("Broadcast.ToTensor() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err == nil {
				if !reflect.DeepEqual(result.Data(), tt.wantData) {
					t.Errorf("Broadcast.ToTensor() = %v, want %v", result.Data(), tt.wantData)
				}
			}
		})
	}
}
