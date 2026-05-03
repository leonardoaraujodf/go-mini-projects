package main

import (
	"context"
	"fmt"
	"runtime/debug"
	"sync"

	"golang.org/x/sync/errgroup"
)

type Job func() error

type WorkerPool struct {
	jobsChan     chan Job
	eg           *errgroup.Group
	egCtx        context.Context
	cancel       context.CancelFunc
	numWorkers   int
	mu           sync.RWMutex
	started      bool
	startOnce    sync.Once
	shutdownOnce sync.Once
}

func NewWorkerPool(ctx context.Context, numWorkers int, maxJobs int) *WorkerPool {
	ctx, cancel := context.WithCancel(ctx)
	eg, egCtx := errgroup.WithContext(ctx)
	return &WorkerPool{
		jobsChan:   make(chan Job, maxJobs),
		eg:         eg,
		egCtx:      egCtx,
		cancel:     cancel,
		numWorkers: numWorkers,
	}
}

func (wp *WorkerPool) Start() {
	wp.startOnce.Do(func() {
		wp.mu.Lock()
		defer wp.mu.Unlock()
		if wp.started {
			return
		}
		wp.started = true
		for i := 0; i < wp.numWorkers; i++ {
			wp.eg.Go(func() error {
				return worker(wp.egCtx, wp.jobsChan)
			})
		}
	})

}

func worker(ctx context.Context, jobsChan <-chan Job) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case job, ok := <-jobsChan:
			if !ok {
				// If channel has been closed, no more jobs to process
				// Worker can finish
				return nil
			}
			err := func() (err error) {
				defer func() {
					if r := recover(); r != nil {
						err = fmt.Errorf("job panicked: %v\n%s", r, debug.Stack())
					}
				}()
				return job()
			}()
			if err != nil {
				return err
			}
		}
	}
}

func (wp *WorkerPool) Submit(ctx context.Context, job Job) error {
	wp.mu.RLock()
	defer wp.mu.RUnlock()
	if !wp.started {
		return fmt.Errorf("worker pool is not running")
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-wp.egCtx.Done():
		return wp.egCtx.Err()
	case wp.jobsChan <- job:
	default:
		return fmt.Errorf("not enough capacity")
	}
	return nil
}

func (wp *WorkerPool) Shutdown() error {
	wp.mu.Lock()
	if !wp.started {
		wp.mu.Unlock()
		return fmt.Errorf("worker pool is not running")
	}
	wp.mu.Unlock()

	wp.shutdownOnce.Do(func() {
		wp.mu.Lock()
		wp.started = false
		close(wp.jobsChan)
		wp.mu.Unlock()

		if wp.cancel != nil {
			wp.cancel()
		}
	})

	return wp.eg.Wait()
}
