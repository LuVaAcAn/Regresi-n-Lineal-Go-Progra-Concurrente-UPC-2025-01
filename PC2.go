package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type Point struct {
	X, Y float64
}

// Versión Secuencial
func linearRegression(points []Point) (float64, float64) {
	n := float64(len(points))
	var sumX, sumY, sumXY, sumXX float64

	for _, p := range points {
		sumX += p.X
		sumY += p.Y
		sumXY += p.X * p.Y
		sumXX += p.X * p.X
	}

	beta1 := (n*sumXY - sumX*sumY) / (n*sumXX - sumX*sumX)
	beta0 := (sumY / n) - beta1*(sumX/n)

	return beta0, beta1
}

// Versión Concurrente
func concurrentLinearRegression(points []Point, goroutines int) (float64, float64) {
	var wg sync.WaitGroup
	var mu sync.Mutex
	n := float64(len(points))
	var sumX, sumY, sumXY, sumXX float64

	chunkSize := len(points) / goroutines

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		start := i * chunkSize
		end := start + chunkSize
		if i == goroutines-1 {
			end = len(points)
		}

		go func(pts []Point) {
			defer wg.Done()
			var localSumX, localSumY, localSumXY, localSumXX float64

			for _, p := range pts {
				localSumX += p.X
				localSumY += p.Y
				localSumXY += p.X * p.Y
				localSumXX += p.X * p.X
			}

			mu.Lock()
			sumX += localSumX
			sumY += localSumY
			sumXY += localSumXY
			sumXX += localSumXX
			mu.Unlock()
		}(points[start:end])
	}

	wg.Wait()

	beta1 := (n*sumXY - sumX*sumY) / (n*sumXX - sumX*sumX)
	beta0 := (sumY / n) - beta1*(sumX/n)

	return beta0, beta1
}

// Generador de Datos Sintéticos
func generateData(n int) []Point {
	rand.Seed(time.Now().UnixNano())
	points := make([]Point, n)
	for i := 0; i < n; i++ {
		x := float64(i)
		noise := rand.NormFloat64() * 10
		y := 2*x + 3 + noise
		points[i] = Point{X: x, Y: y}
	}
	return points
}

func main() {
	dataSize := 1000000
	numGoroutines := 4

	data := generateData(dataSize)

	//Versión Secuencial
	startSeq := time.Now()
	beta0Seq, beta1Seq := linearRegression(data)
	elapsedSeq := time.Since(startSeq)

	//Versión Concurrente
	startConc := time.Now()
	beta0Conc, beta1Conc := concurrentLinearRegression(data, numGoroutines)
	elapsedConc := time.Since(startConc)

	fmt.Println("=== Regresión Lineal en Go ===")
	fmt.Printf("Dataset: %d puntos\n", dataSize)
	fmt.Printf("Goroutines: %d\n", numGoroutines)
	fmt.Println("\n--- Secuencial ---")
	fmt.Printf("β0 (Intercepto): %.4f\n", beta0Seq)
	fmt.Printf("β1 (Pendiente): %.4f\n", beta1Seq)
	fmt.Printf("Tiempo: %v\n", elapsedSeq)
	fmt.Println("\n--- Concurrente ---")
	fmt.Printf("β0 (Intercepto): %.4f\n", beta0Conc)
	fmt.Printf("β1 (Pendiente): %.4f\n", beta1Conc)
	fmt.Printf("Tiempo: %v\n", elapsedConc)
	fmt.Println("\n--- Rendimiento ---")
	speedup := float64(elapsedSeq) / float64(elapsedConc)
	fmt.Printf("Speedup: %.2fx\n", speedup)
}
