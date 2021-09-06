import queue


def fifo(env):
    """First-In, First-Out."""
    m = env.num_machines
    machine_input = [queue.Queue() for _ in range(m)]
    his = set()

    reward, count = 0, 0
    t = env.global_time
    while env.global_time-t < env.schedule_cycle:
         
        state = env.get_machine_doable_action()
        done = env.is_done()

        if done:
            break
        
        for machine_id in range(m):
            for ops_id in state[machine_id]:
                if ops_id not in his:
                    his.add(ops_id)
                    machine_input[machine_id].put(ops_id)

        available_machines_id = env.get_available_machines_id()
        if not available_machines_id:
            env.process_one_second()
        else:
            for machine_id, machine_q in enumerate(machine_input):
                if machine_id in available_machines_id:
                    if not machine_q.empty():
                        ops_id = machine_q.get()
                        job_id, step_id = env.job_manager.sur_index_dict[ops_id]
                        operation = env.job_manager[job_id][step_id]
                        machine_id = operation.machine_id
                        machine = env.machine_manager[machine_id]
                        action = operation
                        machine.transit(env.global_time, action)
        reward += env.cal_reward()
        count += 1
    return reward / count


def lifo(env):
    """Last in, first out."""
    m = env.num_machines
    machine_input = [queue.LifoQueue() for _ in range(m)]
    his = set()

    reward, count = 0, 0
    t = env.global_time
    while env.global_time-t < env.schedule_cycle:
         
        state = env.get_machine_doable_action()
        done = env.is_done()

        if done:
            break

        for machine_id in range(m):
            for ops_id in state[machine_id]:
                if ops_id not in his:
                    his.add(ops_id)
                    machine_input[machine_id].put(ops_id)

        available_machines_id = env.get_available_machines_id()
        if not available_machines_id:
            env.process_one_second()
        else:
            for machine_id, machine_q in enumerate(machine_input):
                if machine_id in available_machines_id:
                    if not machine_q.empty():
                        ops_id = machine_q.get()
                        job_id, step_id = env.job_manager.sur_index_dict[ops_id]
                        operation = env.job_manager[job_id][step_id]
                        machine_id = operation.machine_id
                        machine = env.machine_manager[machine_id]
                        action = operation
                        machine.transit(env.global_time, action)
        reward += env.cal_reward()
        count += 1
    return reward / count


def lpt(env):
    """Longest processing time."""
    m = env.num_machines
    machine_input = [queue.PriorityQueue() for _ in range(m)]
    his = set()

    reward, count = 0, 0
    t = env.global_time
    while env.global_time-t < env.schedule_cycle:
         
        state = env.get_machine_doable_action()
        done = env.is_done()

        if done:
            break

        for machine_id in range(m):
            for ops_id in state[machine_id]:
                if ops_id not in his:
                    his.add(ops_id)
                    pro = env.get_process_time(ops_id)
                    machine_input[machine_id].put((-pro, ops_id))

        available_machines_id = env.get_available_machines_id()
        if not available_machines_id:
            env.process_one_second()
        else:
            for machine_id, machine_q in enumerate(machine_input):
                if machine_id in available_machines_id:
                    if not machine_q.empty():
                        _, ops_id = machine_q.get()
                        job_id, step_id = env.job_manager.sur_index_dict[ops_id]
                        operation = env.job_manager[job_id][step_id]
                        machine_id = operation.machine_id
                        machine = env.machine_manager[machine_id]
                        action = operation
                        machine.transit(env.global_time, action)
        reward += env.cal_reward()
        count += 1
    return reward / count


def spt(env):
    """Shortest processing time."""
    m = env.num_machines
    machine_input = [queue.PriorityQueue() for _ in range(m)]
    his = set()

    reward, count = 0, 0
    t = env.global_time
    while env.global_time-t < env.schedule_cycle:
         
        state = env.get_machine_doable_action()
        done = env.is_done()

        if done:
            break

        for machine_id in range(m):
            for ops_id in state[machine_id]:
                if ops_id not in his:
                    his.add(ops_id)
                    pro = env.get_process_time(ops_id)
                    machine_input[machine_id].put((pro, ops_id))

        available_machines_id = env.get_available_machines_id()
        if not available_machines_id:
            env.process_one_second()
        else:
            for machine_id, machine_q in enumerate(machine_input):
                if machine_id in available_machines_id:
                    if not machine_q.empty():
                        _, ops_id = machine_q.get()
                        job_id, step_id = env.job_manager.sur_index_dict[ops_id]
                        operation = env.job_manager[job_id][step_id]
                        machine_id = operation.machine_id
                        machine = env.machine_manager[machine_id]
                        action = operation
                        machine.transit(env.global_time, action)
        reward += env.cal_reward()
        count += 1
    return reward / count


def ltpt(env):
    """Longest Total Processing Time."""
    m = env.num_machines
    machine_input = [queue.PriorityQueue() for _ in range(m)]
    his = set()

    reward, count = 0, 0
    t = env.global_time
    while env.global_time-t < env.schedule_cycle:
         
        state = env.get_machine_doable_action()
        done = env.is_done()

        if done:
            break

        for machine_id in range(m):
            for ops_id in state[machine_id]:
                if ops_id not in his:
                    his.add(ops_id)
                    pro = env.get_job_process_time(ops_id)
                    machine_input[machine_id].put((-pro, ops_id))

        available_machines_id = env.get_available_machines_id()
        if not available_machines_id:
            env.process_one_second()
        else:
            for machine_id, machine_q in enumerate(machine_input):
                if machine_id in available_machines_id:
                    if not machine_q.empty():
                        _, ops_id = machine_q.get()
                        job_id, step_id = env.job_manager.sur_index_dict[ops_id]
                        operation = env.job_manager[job_id][step_id]
                        machine_id = operation.machine_id
                        machine = env.machine_manager[machine_id]
                        action = operation
                        machine.transit(env.global_time, action)
        reward += env.cal_reward()
        count += 1
    return reward / count


def stpt(env):
    """Shortest Total Processing Time."""
    m = env.num_machines
    machine_input = [queue.PriorityQueue() for _ in range(m)]
    his = set()

    done = False
    reward, count = 0, 0
    t = env.global_time
    while env.global_time-t < env.schedule_cycle:
         
        state = env.get_machine_doable_action()
        done = env.is_done()

        if done:
            break

        for machine_id in range(m):
            for ops_id in state[machine_id]:
                if ops_id not in his:
                    his.add(ops_id)
                    pro = env.get_job_process_time(ops_id)
                    machine_input[machine_id].put((pro, ops_id))

        available_machines_id = env.get_available_machines_id()
        if not available_machines_id:
            env.process_one_second()
        else:
            for machine_id, machine_q in enumerate(machine_input):
                if machine_id in available_machines_id:
                    if not machine_q.empty():
                        _, ops_id = machine_q.get()
                        job_id, step_id = env.job_manager.sur_index_dict[ops_id]
                        operation = env.job_manager[job_id][step_id]
                        machine_id = operation.machine_id
                        machine = env.machine_manager[machine_id]
                        action = operation
                        machine.transit(env.global_time, action)
        reward += env.cal_reward()
        count += 1
    return reward / count


def mor(env):
    """Most Operation Remaining."""
    m = env.num_machines
    machine_input = [queue.PriorityQueue() for _ in range(m)]
    his = set()

    t = env.global_time
    reward, count = 0, 0
    while env.global_time-t < env.schedule_cycle:
         
        state = env.get_machine_doable_action()
        done = env.is_done()

        if done:
            break

        for machine_id in range(m):
            for ops_id in state[machine_id]:
                if ops_id not in his:
                    his.add(ops_id)
                    pro = env.get_num_remain_ops(ops_id)
                    machine_input[machine_id].put((-pro, ops_id))
        
        available_machines_id = env.get_available_machines_id()
        if not available_machines_id:
            env.process_one_second()
        else:
            for machine_id, machine_q in enumerate(machine_input):
                if machine_id in available_machines_id:
                    if not machine_q.empty():
                        _, ops_id = machine_q.get()
                        job_id, step_id = env.job_manager.sur_index_dict[ops_id]
                        operation = env.job_manager[job_id][step_id]
                        machine_id = operation.machine_id
                        machine = env.machine_manager[machine_id]
                        action = operation
                        machine.transit(env.global_time, action)
        reward += env.cal_reward()
        count += 1
    return reward / count


def lor(env):
    """Least Operation Remaining"""
    m = env.num_machines
    machine_input = [queue.PriorityQueue() for _ in range(m)]
    his = set()

    t = env.global_time
    reward, count = 0, 0
    while env.global_time-t < env.schedule_cycle:
        state = env.get_machine_doable_action()
        done = env.is_done()

        if done:
            break

        for machine_id in range(m):
            for ops_id in state[machine_id]:
                if ops_id not in his:
                    his.add(ops_id)
                    pro = env.get_num_remain_ops(ops_id)
                    machine_input[machine_id].put((pro, ops_id))

        available_machines_id = env.get_available_machines_id()
        if not available_machines_id:
            env.process_one_second()
        else:
            for machine_id, machine_q in enumerate(machine_input):
                if machine_id in available_machines_id:
                    if not machine_q.empty():
                        _, ops_id = machine_q.get()
                        job_id, step_id = env.job_manager.sur_index_dict[ops_id]
                        operation = env.job_manager[job_id][step_id]
                        machine_id = operation.machine_id
                        machine = env.machine_manager[machine_id]
                        action = operation
                        machine.transit(env.global_time, action)
        reward += env.cal_reward()
        count += 1
    return reward / count

