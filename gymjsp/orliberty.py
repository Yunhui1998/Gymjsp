import os
import numpy as np
import gymjsp


def load_instance(instance: str = 'ft6'):
    """
    Loads the specified JSSP instance and gets the matrix used to describe that instance.

    Args:
        instance: Instance name. Values for abz5-9, ft06, ft10, ft20, la01-la40, orb01-orb10, swv01-swv20, yn1-yn4.

    Returns:
        * N: Number of jobs.
        
        * M: Number of machines
        
        * time_mat: The processing time matrix. Shape is (N, M).
        
        * machine_mat: The machine processing matrix. Shape is (N, M).
    Example:
        >>> N, M, time_mat, machine_mat = load_instance('abz5')
    """
    path = os.path.join(os.path.dirname(gymjsp.__file__), 'jobshop', 'jobshop.txt')
    
    assert os.path.exists(path)
    assert os.path.isfile(path)

    # time为一个实例的时间表，machine为一个实例的机器表
    # times为该文件包含的所有实例，machines同理
    time = []
    times = []
    machine = []
    machines = []
    with open(path, "r") as file:
        lines = file.readlines()
        start = -1

        for i, line in enumerate(lines):
            if line.find("instance " + instance) != -1:
                start = i + 4
                break
        line = lines[start]
        line = line.strip()
        line = list(filter(None, line.split(' ')))
        assert len(line) == 2
        N = int(line[0])
        M = int(line[1])
        start += 1

        for i in range(N):
            machine = []
            time = []
            line = lines[start + i]
            line = line.strip()
            line = list(filter(None, line.split(' ')))
            line = [int(x) for x in line]

            for j in range(M):
                machine.append(line[2 * j])
                time.append(line[2 * j + 1])

            times.append(time)
            machines.append(machine)

    times = np.array(times)
    machines = np.array(machines)
    return N, M, times, machines


def load_random(N, M):
    """
    Load several randomly generated JSSP instances according to certain rules, 
    and obtain the relevant information describing these instances.

    Args:
        * N: number of jobs for the instance to be generated. Optional values: {15, 20, 30, 50, 100}.
        
        * M: Number of machines to generate instances. Optional values: {15, 20}.

    Returns:
        * I: Number of instances.

        * N: Number of jobs.
        
        * M: Number of machines
        
        * time_mat: The processing time matrix. Shape is (I, N, M).
        
        * machine_mat: The machine processing matrix. Shape is (I, N, M).
    Example:
        >>> I, N, M, time_mat, machine_mat = load_random(30, 15)
    """

    path = os.path.join(os.path.dirname(gymjsp.__file__), 'jobshop','tai{}_{}.txt'.format(N, M))
    # print(path)
    assert os.path.exists(path)
    assert os.path.isfile(path)

    state = "start"

    time = []
    times = []
    machine = []
    machines = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) <= 0:
                continue
            if line[0].isalpha():
                state = __next_state(state)
                continue
            nums = list(filter(None, line.split(' ')))

            if "row" == state:
                if machine:
                    machines.append(machine)
                    machine = []
                    times.append(time)
                    time = []
            elif "times" == state:
                time.append(
                    [int(num) for num in nums]
                )
            elif "machines" == state:
                machine.append(
                    [int(num) for num in nums]
                )
            else:
                raise RuntimeError('State error.')

        machines.append(machine)
        times.append(time)

    I = len(times)
    N = len(times[0])
    M = len(times[0][0])
    times = np.array(times)
    machines = np.array(machines)
    return I, N, M, times, machines


def __next_state(state):
    if "start" == state:
        return "row"
    elif "row" == state:
        return 'times'
    elif "times" == state:
        return 'machines'
    elif "machines" == state:
        return "row"
    else:
        return "error"
