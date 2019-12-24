import torch.multiprocessing as mp
import time


def input_iterator(n):
  for i in range(n):
    yield i


def f(i):
  for j in range(100000):
    pass
  return i*i


def request_task(i, in_queue, out_queue, lock):
  this_proc_id = mp.current_process().name
  item = (this_proc_id, "Hello from {}".format(this_proc_id))

  time.sleep(0.002)

  lock.acquire()
  in_queue.put_nowait(item)
  out = out_queue.get(True, 1.0)
  lock.release()

  if out[0] != item[0]:
    assert False

  return True


def inference_worker(in_queue, out_queue):
  try:
    while True:
      if not in_queue.empty():
        item = in_queue.get_nowait()
        time.sleep(0.001)
        out_queue.put_nowait(item)
  except KeyboardInterrupt:
    print("Keyboard interrupt, worker exiting")


if __name__ == "__main__":
  num_proc = 20
  pool = mp.Pool(processes=num_proc)
  
  manager = mp.Manager()
  in_queue = manager.Queue()
  out_queue = manager.Queue()
  lock = manager.Lock()

  # workers = []
  # for _ in range(num_workers):
  worker = mp.Process(target=inference_worker, args=(in_queue, out_queue))
  worker.start()
    # workers.append(worker)

  t0 = time.time()
  # results = [pool.apply_async(request_task, (i, in_queue, out_queue, lock)) for i in input_iterator(10000)]
  results = pool.starmap_async(request_task, [(i, in_queue, out_queue, lock) for i in input_iterator(100000)])
  results.get()

  # for _ in pool.imap_unordered()
  # results.get()
  print(results)
  elapsed = time.time() - t0
  print("Took {} sec".format(elapsed))

  # for w in workers:
  worker.terminate()
