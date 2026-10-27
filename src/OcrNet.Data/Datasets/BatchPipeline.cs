using System.Collections.Concurrent;
using OcrNet.Core.Tensors;
using OcrNet.Core.Training;

namespace OcrNet.Data.Datasets;

/// <summary>
/// Produces batches on background threads while the trainer consumes them, so data augmentation
/// overlaps with the math. Batches are delivered strictly in index order and each batch's content
/// depends only on its index, which keeps training deterministic regardless of thread timing.
/// </summary>
public static class BatchPipeline
{
    /// <param name="batchCount">Number of batches in the epoch.</param>
    /// <param name="fill">Fills batch <c>index</c>; must set the batch size via <see cref="Batch.Resize"/>.</param>
    public static IEnumerable<Batch> Run(
        int batchCount, int batchSize, TensorShape shape, int producers,
        Action<int, Batch> fill, CancellationToken cancellationToken)
    {
        producers = Math.Clamp(producers, 1, Math.Max(1, batchCount));
        int capacity = producers * 2;
        var pool = new ConcurrentBag<Batch>();
        for (int i = 0; i < capacity; i++)
            pool.Add(new Batch(batchSize, shape));

        var ready = new Batch?[batchCount];
        var gate = new object();
        using var free = new SemaphoreSlim(capacity, capacity);
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        Exception? failure = null;
        int next = -1;

        var tasks = Enumerable.Range(0, producers).Select(_ => Task.Factory.StartNew(() =>
        {
            try
            {
                while (true)
                {
                    free.Wait(stop.Token);
                    int index = Interlocked.Increment(ref next);
                    if (index >= batchCount)
                    {
                        free.Release();
                        return;
                    }
                    if (!pool.TryTake(out var batch))
                        batch = new Batch(batchSize, shape);
                    fill(index, batch);
                    lock (gate)
                    {
                        ready[index] = batch;
                        Monitor.PulseAll(gate);
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception ex)
            {
                lock (gate)
                {
                    failure = ex;
                    Monitor.PulseAll(gate);
                }
            }
        }, stop.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default)).ToArray();

        try
        {
            for (int index = 0; index < batchCount; index++)
            {
                Batch batch;
                lock (gate)
                {
                    while (ready[index] is null)
                    {
                        if (failure is not null)
                            throw new InvalidOperationException("Batch generation failed: " + failure.Message, failure);
                        cancellationToken.ThrowIfCancellationRequested();
                        Monitor.Wait(gate, 50);
                    }
                    batch = ready[index]!;
                    ready[index] = null;
                }

                yield return batch;

                pool.Add(batch);
                free.Release();
            }
        }
        finally
        {
            stop.Cancel();
            try
            {
                Task.WaitAll(tasks);
            }
            catch (AggregateException)
            {
                // Producers only end with cancellation here; errors were surfaced above.
            }
        }
    }
}
