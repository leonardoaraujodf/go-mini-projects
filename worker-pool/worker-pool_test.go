package main

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// --- helpers ---

func newStartedPool(t *testing.T, numWorkers, maxJobs int) *WorkerPool {
	t.Helper()
	wp := NewWorkerPool(context.Background(), numWorkers, maxJobs)
	wp.Start()
	t.Cleanup(func() {
		// Ignore errors: the pool might have already been shut down by the test.
		_ = wp.Shutdown()
	})
	return wp
}

// --- API tests ---

func TestSubmitBeforeStart(t *testing.T) {
	wp := NewWorkerPool(context.Background(), 2, 10)
	err := wp.Submit(context.Background(), func() error { return nil })
	if err == nil {
		t.Fatal("expected error when submitting to a pool that has not been started")
	}
}

func TestShutdownBeforeStart(t *testing.T) {
	wp := NewWorkerPool(context.Background(), 2, 10)
	err := wp.Shutdown()
	if err == nil {
		t.Fatal("expected error when shutting down a pool that has not been started")
	}
}

func TestStartIdempotent(t *testing.T) {
	wp := NewWorkerPool(context.Background(), 2, 10)
	// Calling Start multiple times must not panic or spawn extra workers.
	wp.Start()
	wp.Start()
	wp.Start()

	if err := wp.Shutdown(); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected shutdown error: %v", err)
	}
}

func TestSubmitAndExecute(t *testing.T) {
	wp := newStartedPool(t, 2, 10)

	var executed atomic.Bool
	done := make(chan struct{})
	err := wp.Submit(context.Background(), func() error {
		executed.Store(true)
		close(done)
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected submit error: %v", err)
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("job was not executed within timeout")
	}

	if !executed.Load() {
		t.Fatal("job was not executed")
	}
}

func TestSubmitCapacityExceeded(t *testing.T) {
	// 1 worker, queue capacity of 1.
	// Submit a blocking job to occupy the worker, fill the buffer, then expect
	// the next submit to fail immediately.
	wp := NewWorkerPool(context.Background(), 1, 1)
	wp.Start()
	defer func() { _ = wp.Shutdown() }()

	workerBusy := make(chan struct{})
	unblockWorker := make(chan struct{})

	// This job occupies the single worker.
	if err := wp.Submit(context.Background(), func() error {
		close(workerBusy)
		<-unblockWorker
		return nil
	}); err != nil {
		t.Fatalf("first submit failed: %v", err)
	}

	// Wait until the worker picks up the job so the queue is definitely empty.
	<-workerBusy

	// Fill the single-slot buffer.
	if err := wp.Submit(context.Background(), func() error { return nil }); err != nil {
		t.Fatalf("second submit (filling buffer) failed: %v", err)
	}

	// This submit must fail because the worker is busy and the buffer is full.
	err := wp.Submit(context.Background(), func() error { return nil })
	if err == nil {
		t.Fatal("expected capacity error, got nil")
	}

	close(unblockWorker)
}

func TestSubmitContextCanceled(t *testing.T) {
	wp := NewWorkerPool(context.Background(), 1, 0) // unbuffered channel
	wp.Start()
	defer func() { _ = wp.Shutdown() }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	err := wp.Submit(ctx, func() error { return nil })
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", err)
	}
}

func TestJobErrorIsHandled(t *testing.T) {
	// A job that returns an error must not crash the pool or stop other jobs.
	wp := newStartedPool(t, 2, 10)

	if err := wp.Submit(context.Background(), func() error {
		return errors.New("deliberate job error")
	}); err != nil {
		t.Fatalf("submit failed: %v", err)
	}

	// Subsequent job must still execute.
	done := make(chan struct{})
	if err := wp.Submit(context.Background(), func() error {
		close(done)
		return nil
	}); err != nil {
		t.Fatalf("submit failed: %v", err)
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("pool stopped processing after a job error")
	}
}

func TestPanicRecovery(t *testing.T) {
	// A job that panics must not crash the pool; subsequent jobs must run.
	wp := newStartedPool(t, 2, 10)

	if err := wp.Submit(context.Background(), func() error {
		panic("deliberate panic")
	}); err != nil {
		t.Fatalf("submit failed: %v", err)
	}

	done := make(chan struct{})
	if err := wp.Submit(context.Background(), func() error {
		close(done)
		return nil
	}); err != nil {
		t.Fatalf("submit after panic failed: %v", err)
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("pool stopped processing after a panicking job")
	}
}

func TestSubmitAfterShutdown(t *testing.T) {
	wp := newStartedPool(t, 2, 10)

	if err := wp.Shutdown(); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected shutdown error: %v", err)
	}

	err := wp.Submit(context.Background(), func() error { return nil })
	if err == nil {
		t.Fatal("expected error when submitting to a shut-down pool")
	}
}

func TestShutdownWaitsForInFlightJobs(t *testing.T) {
	// Shutdown must not return until all currently-executing jobs finish.
	// (Buffered-but-not-yet-started jobs are not guaranteed to run after Shutdown.)
	const numWorkers = 2

	wp := NewWorkerPool(context.Background(), numWorkers, numWorkers)
	wp.Start()

	started := make(chan struct{}, numWorkers)
	unblock := make(chan struct{})
	var counter atomic.Int32

	// Submit exactly numWorkers jobs so every worker is busy.
	for range numWorkers {
		if err := wp.Submit(context.Background(), func() error {
			started <- struct{}{}
			<-unblock
			counter.Add(1)
			return nil
		}); err != nil {
			t.Fatalf("submit failed: %v", err)
		}
	}

	// Wait until all workers are inside their job.
	for range numWorkers {
		select {
		case <-started:
		case <-time.After(2 * time.Second):
			t.Fatal("workers did not start within timeout")
		}
	}

	// Begin shutdown in the background; it must block until jobs complete.
	shutdownDone := make(chan error, 1)
	go func() { shutdownDone <- wp.Shutdown() }()

	// Shutdown should still be blocked at this point.
	select {
	case <-shutdownDone:
		t.Fatal("Shutdown returned before in-flight jobs were unblocked")
	case <-time.After(50 * time.Millisecond):
		// expected: Shutdown is waiting
	}

	// Unblock the jobs; Shutdown must now complete.
	close(unblock)

	select {
	case err := <-shutdownDone:
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("unexpected shutdown error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Shutdown did not complete after jobs finished")
	}

	if got := counter.Load(); got != int32(numWorkers) {
		t.Fatalf("expected %d jobs to complete, got %d", numWorkers, got)
	}
}

// --- Concurrency tests ---

func TestAllJobsExecuted(t *testing.T) {
	const numWorkers = 4
	const numJobs = 100

	wp := newStartedPool(t, numWorkers, numJobs)

	var counter atomic.Int32
	var wg sync.WaitGroup
	wg.Add(numJobs)

	for range numJobs {
		if err := wp.Submit(context.Background(), func() error {
			counter.Add(1)
			wg.Done()
			return nil
		}); err != nil {
			t.Fatalf("submit failed: %v", err)
		}
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("not all jobs completed within timeout")
	}

	if got := counter.Load(); got != numJobs {
		t.Fatalf("expected %d executions, got %d", numJobs, got)
	}
}

func TestJobsRunConcurrently(t *testing.T) {
	// Verify that numWorkers jobs can run at the same time.
	// Each job signals it has started, then waits for permission to finish.
	// If workers are truly concurrent, all numWorkers jobs start before any finishes.
	const numWorkers = 4

	wp := newStartedPool(t, numWorkers, numWorkers)

	started := make(chan struct{}, numWorkers)
	unblock := make(chan struct{})

	for range numWorkers {
		if err := wp.Submit(context.Background(), func() error {
			started <- struct{}{}
			<-unblock
			return nil
		}); err != nil {
			t.Fatalf("submit failed: %v", err)
		}
	}

	// Wait for all workers to signal they have started.
	timeout := time.After(2 * time.Second)
	for range numWorkers {
		select {
		case <-started:
		case <-timeout:
			t.Fatal("jobs did not start concurrently within timeout")
		}
	}

	// All numWorkers jobs are running simultaneously — unblock them.
	close(unblock)
}

func TestConcurrentSubmit(t *testing.T) {
	// Multiple goroutines submit jobs concurrently; no races, no deadlocks.
	const numWorkers = 8
	const numSubmitters = 16
	const jobsPerSubmitter = 50
	const total = numSubmitters * jobsPerSubmitter

	wp := newStartedPool(t, numWorkers, total)

	var counter atomic.Int32
	var jobWg sync.WaitGroup
	jobWg.Add(total)

	var submitterWg sync.WaitGroup
	submitterWg.Add(numSubmitters)

	for range numSubmitters {
		go func() {
			defer submitterWg.Done()
			for range jobsPerSubmitter {
				if err := wp.Submit(context.Background(), func() error {
					counter.Add(1)
					jobWg.Done()
					return nil
				}); err != nil {
					// Should not happen: queue is large enough for all jobs.
					jobWg.Done()
					t.Errorf("unexpected submit error: %v", err)
				}
			}
		}()
	}

	submitterWg.Wait()

	// Wait for every job to execute before asserting.
	done := make(chan struct{})
	go func() { jobWg.Wait(); close(done) }()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("not all jobs completed within timeout")
	}

	if got := counter.Load(); got != total {
		t.Fatalf("expected %d jobs executed, got %d", total, got)
	}
}

func TestShutdownIdempotent(t *testing.T) {
	// Calling Shutdown concurrently from multiple goroutines must not panic.
	wp := newStartedPool(t, 2, 10)

	var wg sync.WaitGroup
	const goroutines = 8
	wg.Add(goroutines)

	for range goroutines {
		go func() {
			defer wg.Done()
			_ = wp.Shutdown()
		}()
	}

	wg.Wait()
}

func TestParentContextCancellation(t *testing.T) {
	// Cancelling the parent context should stop the workers via egCtx.
	ctx, cancel := context.WithCancel(context.Background())
	wp := NewWorkerPool(ctx, 2, 10)
	wp.Start()

	cancel()

	// After parent context cancellation, eg.Wait() returns the first non-nil
	// worker error (context.Canceled). Shutdown propagates that error.
	err := wp.Shutdown()
	if err == nil {
		t.Fatal("expected an error after parent context cancellation")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", err)
	}
}
