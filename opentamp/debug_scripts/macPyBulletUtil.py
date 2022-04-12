import pybullet as p
import time

def stepSimulation(t_limit=None, delay=0.):
    start_t = time.time()

    envid = p.connect(p.SHARED_MEMORY, 0)
    time.sleep(0.1)
    print('Connected to physics server', envid)
    while (t_limit is None or time.time() - start_t < t_limit):
        p.stepSimulation()
        if delay > 0: time.sleep(delay)

if __name__ == "__main__":
    stepSimulation()
