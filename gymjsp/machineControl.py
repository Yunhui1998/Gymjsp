import random
from collections import OrderedDict
import numpy as np
from .configs import (PROCESSING_NODE,
                      DONE_NODE,
                      DELAYED_NODE)


class MachineSet:
    """
    Manage all machines. You can directly use ID to access corresponding Machine. 

    Args:
        * machine_matrix(np.ndarray): Machine processing matrix from OR-Liberty.
        
        * job_manager(JobManager): Corresponding JobManager.
        
        * delay(bool): True indicates that machine can a machine can enter a wait state.
        
        * verbose(bool): True indicates manager will print the info while start/end one operation.

    Attributes:
        machines(OrderedDict): Map the machine ID(int) to the corresponding Machine. 
    """

    def __init__(self,
                 machine_matrix,
                 job_manager,
                 delay=True,
                 verbose=False):
        machine_matrix = machine_matrix.astype(int)

        # Parse machine indices
        machine_index = list(set(machine_matrix.flatten().tolist()))

        # Global machines dict
        self.machines = OrderedDict()
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            possible_ops = []
            for job_id, step_id in zip(job_ids, step_ids):
                possible_ops.append(job_manager[job_id][step_id])
            m_id += 1  # To make machine index starts from 1
            self.machines[m_id] = Machine(m_id, possible_ops, delay, verbose)

    def do_processing(self, t: int):
        """
        Drive all machines to t.

        Args:
            t: Current simulation time.
        """
        for _, machine in self.machines.items():
            machine.do_processing(t)

    def load_op(self, machine_id, op, t):
        """
        Specify the machine to process the operation.

        Args:
            machine_id(int): Id of the machine to perform the processing.
            op(Operation): The operation to be processed.
            t: Current simulation time.
        """
        self.machines[machine_id].load_op(op, t)

    def __getitem__(self, index):
        return self.machines[index]

    def get_available_machines(self, shuffle_machine: bool = True):
        """
        Get all available machines.

        Args:
            shuffle_machine: True indicates that the list is scrambled.
        Returns:
            Contains a list of all available machines.
        """
        m_list = []
        for _, m in self.machines.items():
            if m.available():
                m_list.append(m)

        if shuffle_machine:
            m_list = random.sample(m_list, len(m_list))

        return m_list

    def get_idle_machines(self):
        """
        Get all idle machines.

        Returns:
            Contains a list of all idle machines.
        """
        m_list = []
        for _, m in self.machines.items():
            if m.current_op is None and not m.work_done():
                m_list.append(m)
        return m_list

    def cal_total_cost(self):
        """
        Calculate the length of queues for all machines.

        Returns:
            int: The length of queues for all machines.
        """
        c = 0
        for _, m in self.machines.items():
            c += len(m.doable_ops_no_delay)
        return c

    def update_cost_function(self, cost):
        """
        update all cost functions of machines.
        
        Args:
            cost: new cost.
        """
        for _, m in self.machines.items():
            m.cost += cost

    def get_machines(self):
        """
        Get all machines.

        Returns:
            list: An out-of-order list of all machines.
        """
        m_list = [m for _, m in self.machines.items()]
        return random.sample(m_list, len(m_list))

    def all_delayed(self):
        """
        Determine whether all machines are waiting.

        Returns:
            bool: True indicates that all machines are in the waiting state.
        """
        return np.product([m.delayed_op is not None for _, m in self.machines.items()])

    def fab_stuck(self):
        """
        Determine whether all machines are waiting and unavailable.

        Returns:
            bool: True indicates that all machines are unavailable and all machines are in the wait state.
        """
        # All machines are not available and All machines are delayed.
        all_machines_not_available_cond = not self.get_available_machines()
        all_machines_delayed_cond = self.all_delayed()
        return all_machines_not_available_cond and all_machines_delayed_cond


class Machine:
    """
    The simulation machine.

    Attributes:
        * machine_id(int): Machine ID.
        
        * possible_ops(list): All Operation need to process on this machine.
        
        * remain_ops(list): All unfinished Operation.
        
        * current_op(Operation): The Operation currently being processed
        
        * delayed_op(Operation): The Operation currently waiting.
        
        * prev_op(Operation): The last completed Operation.
        
        * remaining_time(int): The remaining processing time of the current operation.
        
        * done_ops(list): A list of all completed operations.
        
        * num_done_ops(int): Number of operations completed.
        
        * cost: Total current costs.
        
        * delay(bool): True indicates that the machine can enter the wait state when it is idle.
        
        * verbose(bool): True indicates the machine will print the info while start/end one operation.
        
        * total_work_time: Total processing time.
        
        * finish_time: Time to complete all processes.
    """

    def __init__(self, machine_id: int, possible_ops: list, delay: bool, verbose: bool):
        """
        Args:
            machine_id: Machine ID.
            possible_ops: All Operation need to process on this machine.
            delay: True indicates that the machine can enter the wait state when it is idle.
            verbose: True indicates the machine will print the info while start/end one operation.
        """
        self.machine_id = machine_id
        self.possible_ops = possible_ops
        self.remain_ops = possible_ops
        self.current_op = None
        self.delayed_op = None
        self.prev_op = None
        self.remaining_time = 0
        self.done_ops = []
        self.num_done_ops = 0
        self.cost = 0
        self.delay = delay
        self.verbose = verbose
        self.finish_time = 0
        self.total_work_time = 0

    def __str__(self):
        return "Machine {}".format(self.machine_id)

    def available(self):
        """
        Returns:
            True indicates the machine can load operation.
        """
        future_work_exist_cond = bool(self.doable_ops())
        currently_not_processing_cond = self.current_op is None
        not_wait_for_delayed_cond = not self.wait_for_delayed()
        ret = future_work_exist_cond and currently_not_processing_cond and not_wait_for_delayed_cond
        return ret

    def wait_for_delayed(self):
        """
        Returns:
            True indicates the machine is waiting.
        """
        wait_for_delayed_cond = self.delayed_op is not None
        ret = wait_for_delayed_cond
        if wait_for_delayed_cond:
            delayed_op_ready_cond = self.delayed_op.prev_op.node_status == DONE_NODE
            ret = ret and not delayed_op_ready_cond
        return ret

    def doable_ops(self):
        """
        Returns:
            A list of Operations that can be processed on the machine.
        """
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == DONE_NODE
                prev_process = op.prev_op.node_status == PROCESSING_NODE
                first_op = not bool(self.done_ops)
                if self.delay:
                    # each machine's first processing operation should not be a reserved operation
                    if first_op:
                        cond = prev_done
                    else:
                        cond = (prev_done or prev_process)
                else:
                    cond = prev_done

                if cond:
                    doable_ops.append(op)
                else:
                    pass

        return doable_ops

    @property
    def doable_ops_id(self):
        doable_ops_id = []
        doable_ops = self.doable_ops()
        for op in doable_ops:
            doable_ops_id.append(op.id)

        return doable_ops_id

    @property
    def doable_ops_no_delay(self):
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == DONE_NODE
                if prev_done:
                    doable_ops.append(op)
        return doable_ops

    def work_done(self):
        """
        Returns:
            bool: True indicates that all operations on the machine are completed.
        """
        return not self.remain_ops

    def load_op(self, t, op):
        """
        At time T, the operation 'op' is loaded into the machine for processing.

        Args:
            t(int): Current simulation time.
            op(Operation): The Operation to be processed.
        """
        # Procedures for double-checkings
        # If machine waits for the delayed job is done:
        if self.wait_for_delayed():
            raise RuntimeError("Machine {} waits for the delayed job {} but load {}".format(self.machine_id,
                                                                                            print(self.delayed_op),
                                                                                            print(op)))

        # ignore input when the machine is not available
        if not self.available():
            raise RuntimeError("Machine {} is not available".format(self.machine_id))

        # ignore when input op's previous op is not done yet:
        if not op.processible():
            raise RuntimeError("Operation {} is not processible yet".format(print(op)))

        if op not in self.possible_ops:
            raise RuntimeError("Machine {} can't perform job {} step {}".format(self.machine_id,
                                                                                op.job_id,
                                                                                op.step_id))

        # Essential condition for checking whether input is delayed
        # if delayed then, flush dealed_op attr
        if op == self.delayed_op:
            if self.verbose:
                print("[DELAYED OP LOADED] / MACHINE {} / {} / at {}".format(self.machine_id, op, t))
            self.delayed_op = None

        else:
            if self.verbose:
                print("[LOAD] / Machine {} / {} on at {}".format(self.machine_id, op, t))

        # Update operation's attributes
        op.node_status = PROCESSING_NODE
        op.remaining_time = op.processing_time
        op.start_time = t

        # Update machine's attributes
        self.current_op = op
        self.remaining_time = op.processing_time
        self.remain_ops.remove(self.current_op)

    def unload(self, t):
        """
        If the loaded operation has completed, it is unloaded.

        Args:
            t: Current simulation time.
        """
        if self.verbose:
            print("[UNLOAD] / Machine {} / Op {} / t = {}".format(self.machine_id, self.current_op, t))
        self.total_work_time += self.current_op.processing_time
        self.current_op.node_status = DONE_NODE
        self.current_op.end_time = t
        self.done_ops.append(self.current_op)
        self.num_done_ops += 1
        self.prev_op = self.current_op
        self.current_op = None
        self.remaining_time = 0
        if self.work_done():
            self.finish_time = t

    def do_processing(self, t):
        """
        The machine runs up to t.

        Args:
            t: Current simulation time.
        """
        if self.remaining_time > 0:  # When machine do some operation
            if self.current_op is not None:
                self.current_op.remaining_time -= 1
                if self.current_op.remaining_time <= 0:
                    if self.current_op.remaining_time < 0:
                        raise RuntimeWarning("Negative remaining time observed")
                    if self.verbose:
                        print("[OP DONE] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.current_op, t))
                    self.unload(t)
            # to compute idle_time reward, we need to count delayed_time
            elif self.delayed_op is not None:
                self.delayed_op.delayed_time += 1
                self.delayed_op.remaining_time -= 1

            doable_ops = self.doable_ops()
            if doable_ops:
                for op in doable_ops:
                    op.waiting_time += 1
            else:
                pass

            self.remaining_time -= 1

    def transit(self, t, a):
        """
        At the current simulation time t, load operation a.
        If the operation is not doable (the previous operation has not been completed), 
        the machine goes into a waiting state for the operation.

        Args:
            t(int): Current simulation time.
            a(Operation): The operation to be processed.
        """
        if self.available():  # Machine is ready to process.
            if a.processible():  # selected action is ready to be loaded right now.
                self.load_op(t, a)
                if a.processing_time <= 0:
                    self.unload(t)
            else:  # When input operation turns out to be 'delayed'
                a.node_status = DELAYED_NODE
                self.delayed_op = a
                self.delayed_op.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.current_op = None  # MACHINE is now waiting for delayed ops
                if self.verbose:
                    print(
                        "[DELAYED OP CHOSEN] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.delayed_op,
                                                                                      t))
        else:
            raise RuntimeError("Access to not available machine")
