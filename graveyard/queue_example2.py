import torch.multiprocessing as mp
import time


def input_iterator(n):
  for i in range(n):
    yield i


def f(i):
  for j in range(100000):
    pass
  return i*i


def request_task(i, in_queue, nproc, *out_queues):
  this_proc_id = int(mp.current_process().name.split("-")[-1]) % nproc
  item = (this_proc_id, "Hello from {}".format(this_proc_id))

  time.sleep(0.005)

  in_queue.put_nowait(item)

  out = out_queues[this_proc_id]
  item_out = out.get(True, 1.0)

  time.sleep(0.005)

  if item_out[0] != item[0]:
    assert False

  return True


def request_task_many(worker_id, in_queue, out_queue, tasks):
  print("Requesting {} tasks".format(tasks))
  for _ in range(tasks):
    item = (worker_id, "Hello from {}".format(worker_id))
    time.sleep(0.005)

    in_queue.put_nowait(item)
    item_out = out_queue.get(True, 1.0)

    time.sleep(0.005)

    if item_out[0] != item[0]:
      assert False


def inference_worker(in_queue, nproc, *out_queues):
  try:
    while True:
      if not in_queue.empty():
        item = in_queue.get_nowait()
        request_proc_id = int(item[0])

        # Simulate doing inference with a model.
        time.sleep(0.001)
        out = out_queues[request_proc_id]
        out.put_nowait(item)

  except KeyboardInterrupt:
    print("Keyboard interrupt, worker exiting")


if __name__ == "__main__":
  total_jobs = 10000
  num_proc = 20
  jobs_per_proc = total_jobs // num_proc
  # pool = mp.Pool(processes=num_proc)
  
  manager = mp.Manager()
  in_queue = manager.Queue()

  out_queues = []
  for i in range(num_proc):
    out_queues.append(manager.Queue())

  worker = mp.Process(target=inference_worker, args=(in_queue, num_proc, *out_queues))
  worker.start()

  t0 = time.time()

  processes = []
  for i in range(num_proc):
    p = mp.Process(target=request_task_many, args=(i, in_queue, out_queues[i], jobs_per_proc))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()

  # results = [pool.apply_async(request_task, (i, in_queue, out_queue, lock)) for i in input_iterator(10000)]
  # results = pool.starmap_async(request_task, [(i, in_queue, num_proc, *out_queues) for i in input_iterator(10000)], chunksize=100)
  # results.get()
  # print(results)
  elapsed = time.time() - t0
  print("Took {} sec".format(elapsed))

  worker.terminate()
