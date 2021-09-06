import os
from typing import Any

import gym
import numpy as np
import networkx as nx
from gym import spaces
from scipy.sparse import csr_matrix
from .operationControl import JobSet
from .machineControl import MachineSet
from .configs import (DISJUNCTIVE_TYPE)
from .orliberty import load_random, load_instance
from .heuristicRule import (fifo, lifo, lpt, spt, ltpt, stpt, mor, lor)


def get_nodelist(g: nx.OrderedDiGraph):
    """Get all node in the graph in list."""
    return [i for i in range(nx.number_of_nodes(g))]


class BasicJsspEnv(gym.Env):
    """Basic JSSP environment."""

    def __init__(self,
                 name: str = None,
                 num_jobs: int = None,
                 num_machines: int = None,
                 detach_done: bool = False,
                 embedding_dim: int = 16,
                 delay: bool = False,
                 verbose: bool = False,
                 reward_type: str = 'idle_time'):
        """
        Args:
            name: The name of JSSP instance.(abz5-9, ft06, ft10, ft20, la01-la40, orb01-orb10, swv01-swv20, yn1-yn4).
            
            num_jobs: Number of jobs for the selected instance.
            
            num_machines: Number of machines for the selected instance.
            
            detach_done: True indicates observation will contain no information about completed operations.
            
            embedding_dim: Dimension of embedding.
            
            delay: True indicates environment allows you to choose a operations
            that needs to be processed on a busy machine.
            
            verbose: True indicates environment will print the info while start/end one operation.
            It's useful for debugging.
            
            reward_type: The way rewards are calculated.('idle_time', 'utilization')

        Examples:
            >>> env = BasicJsspEnv('ft06')

            or

            >>> env = BasicJsspEnv(num_jobs=6, num_machines=6)
        """
        super(BasicJsspEnv, self).__init__()

        if name is None:
            assert num_jobs is not None
            assert num_machines is not None
            self.name = '{} machine {} job'.format(num_machines, num_jobs)
            i, n, m, processing_time_matrix, machine_matrix = load_random(num_jobs, num_machines)
            processing_time_matrix = processing_time_matrix[np.random.randint(0, i - 1)]
        else:
            self.name = name
            n, m, processing_time_matrix, machine_matrix = load_instance(name)

        self.machine_matrix = machine_matrix
        self.processing_time_matrix = processing_time_matrix

        self._machine_set = list(set(self.machine_matrix.flatten().tolist()))
        self.num_machines = len(self._machine_set)
        self.detach_done = detach_done
        self.embedding_dim = embedding_dim
        self.num_jobs = self.processing_time_matrix.shape[0]
        self.num_steps = self.processing_time_matrix.shape[1]
        self.use_surrogate_index = True
        self.delay = delay
        self.verbose = verbose
        self.reward_type = reward_type
        self.job_manager = JobSet(self.machine_matrix,
                                  self.processing_time_matrix,
                                  embedding_dim=self.embedding_dim)
        self.machine_manager = MachineSet(self.machine_matrix,
                                          self.job_manager,
                                          self.delay,
                                          self.verbose)
        self.global_time = 0  # -1 matters a lot
        self.num_unsafety = 0
        self.num_wait = 0
        self.num_do_action = 0
        self.num_ = 0
        self.wait_action = self.num_jobs * self.num_steps
        self._max_pro_time = np.max(processing_time_matrix)

        self.action_space = spaces.Discrete(self.num_jobs * self.num_machines + 1)
        self.observation_space = BasicState(self.num_jobs, self.num_steps, self.num_machines)

    def step(self, action: int):
        """
        Process a action contained in action space.

        Args:
            action: Contained in the action space.

        Returns:
            Returns four values. These are:
            * observation(object): An environment-specific object representing your observation of the environment.
            * reward(float): Amount of reward achieved by the previous action.
            * done(boolean): Whether it’s time to reset the environment again. True indicates the episode has terminated.
            * info(dict): Diagnostic information useful for debugging. It can sometimes be useful for learning.
        """
        doable_ops = self.get_doable_ops_in_list()
        self.transit(action)
        g, done = self.observe()
        reward = self.cal_reward(action, doable_ops=doable_ops)

        observation = self.g2s(g)

        info = {
            'cost': self.num_unsafety,
            'wait': self.num_wait,
            'total action': self.num_do_action,
            'makespan': self.global_time
        }
        return observation, reward, done, info

    def reset(self, random_rate: float = 0., shuffle: bool = False, ret: bool = True):
        """
        Reset the environment.

        Args:
            random_rate: Each operation has a random_rate probability to randomly increase or decrease the process time.
            
            shuffle: True indicates each job has a random_rate probability to shuffle.
            
            ret: True indicates this function will returns the initial observation.

        Returns:
            None or the initial observation.
        """
        machine_matrix = self.machine_matrix.copy()
        processing_time_matrix = self.processing_time_matrix.copy()
        # shuffle
        if random_rate:
            if shuffle:
                for job_id in range(self.num_jobs):
                    if np.random.random() > random_rate:
                        continue
                    idx_s = np.random.choice(self.num_steps,
                                             size=max(2, int(self.num_steps * random_rate)),
                                             replace=False)
                    idx_t = idx_s.copy()
                    np.random.shuffle(idx_t)
                    machine_matrix[job_id][idx_s] = machine_matrix[job_id][idx_t]
                    processing_time_matrix[job_id][idx_s] = processing_time_matrix[job_id][idx_t]

            for job_id in range(self.num_jobs):
                for step_id in range(self.num_steps):
                    if np.random.random() < random_rate:
                        bias = np.random.normal(loc=0, scale=0.1)
                        bias = min(max(-1, bias), 1)
                        processing_time_matrix[job_id][step_id] *= 1 + bias
                        processing_time_matrix[job_id][step_id] = int(processing_time_matrix[job_id][step_id])

        # remake env
        self.job_manager = JobSet(machine_matrix,
                                  processing_time_matrix,
                                  embedding_dim=self.embedding_dim)
        self.machine_manager = MachineSet(machine_matrix,
                                          self.job_manager,
                                          self.delay,
                                          self.verbose)
        self.global_time = 0  # -1 matters a lot
        self.num_unsafety = 0
        self.num_do_action = 0
        self.num_wait = 0

        if ret:
            g, _ = self.observe()
            state = self.g2s(g)
            return state  # reward, done, info can't be included

    def seed(self, seed: Any = None) -> Any:
        """Sets the seed for this environment's random number generator(s)."""
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        super().seed(seed)

    def process_one_second(self):
        """process all machines one second."""
        self.global_time += 1
        self.machine_manager.do_processing(self.global_time)

    def transit(self, action):
        """
        Process a action contained in action space. But differ from step, there is no return value.

        Args:
            action: Contained in the action space.
        """
        if self.is_done():
            return

        self.num_do_action += 1
        if action is None:
            # Perform random action
            machine = np.random.choice(self.machine_manager.get_available_machines())
            op_id = np.random.choice(machine.doable_ops_id)
            job_id, step_id = self.job_manager.sur_index_dict[op_id]
            operation = self.job_manager[job_id][step_id]
            action = operation
            machine.transit(self.global_time, action)
        else:
            if isinstance(action, np.ndarray):
                # print(action, action.shape)
                action = int(action)
            if self.use_surrogate_index:
                if action in self.job_manager.sur_index_dict.keys():
                    if action not in self.get_doable_ops_in_list():
                        self.process_one_second()
                        self.num_unsafety += 1
                        return
                    job_id, step_id = self.job_manager.sur_index_dict[action]
                elif action == self.wait_action:
                    self.process_one_second()
                    self.num_wait += 1
                    return
                else:
                    raise RuntimeError("Input action is not valid")
            else:
                job_id, step_id = action

            operation = self.job_manager[job_id][step_id]
            machine_id = operation.machine_id
            machine = self.machine_manager[machine_id]
            action = operation
            machine.transit(self.global_time, action)

    def is_done(self):
        """Whether it’s time to reset the environment again. True indicates the episode has terminated."""
        jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
        if np.prod(jobs_done) == 1:
            done = True
        else:
            done = False
        return done

    def cal_reward(self, action: int = None, reward_type: str = None, doable_ops: list = None):
        """
        Calculate the reward for performing the action.

        Args:
            action: The selected action.
            
            reward_type: Type of reward.

            doable_ops: All doable operations in list.
        """
        if self.is_done():
            return 1

        if reward_type is None:
            reward_type = self.reward_type

        if reward_type == 'utilization':
            t_cost = self.machine_manager.cal_total_cost()
            reward = -t_cost
            reward = (reward / self.num_jobs) * 2 + 1

        elif reward_type == 'idle_time':
            reward = -float(len(self.machine_manager.get_idle_machines())) / float(self.num_machines)

        else:
            raise KeyError(
                "Don't know what is {} reward. Please use 'utilization', 'idle_time', etc.".format(reward_type))

        if not doable_ops:
            doable_ops = self.get_doable_ops_in_list()
        if action == self.wait_action:
            if doable_ops:
                reward -= 0.3
                reward = max(reward, -1)
        elif doable_ops and action is not None and action not in doable_ops:
            reward -= 0.5
            reward = max(reward, -1)

        return reward

    def flush_trivial_ops(self, gamma=1.0):
        """
        Get to the next point where there's a doable operation to do.

        Args:
            gamma: Discount factor.

        Returns:
            Returns three values. These are:

            * m_list(list): Machines that have more than one doable operation.
            
            * reward(float): Amount of reward achieved by the previous action.
            
            * done(boolean): Whether it’s time to reset the environment again. True indicates the episode has terminated.
        """
        done = False
        cum_reward = 0
        while True:
            m_list = []
            do_op_dict = self.get_doable_ops_in_dict()
            all_machine_work = False if bool(do_op_dict) else True

            if all_machine_work:  # all machines are on processing. keep process!
                self.process_one_second()
            else:  # some of machine has possibly trivial action. the others not.
                # load trivial ops to the machines
                num_ops_counter = 1
                for m_id, op_ids in do_op_dict.items():
                    num_ops = len(op_ids)
                    if num_ops == 1:
                        self.transit(op_ids[0])  # load trivial action
                        _, _ = self.observe()
                        r = self.cal_reward()
                        cum_reward = r + gamma * cum_reward
                    else:
                        m_list.append(m_id)
                        num_ops_counter *= num_ops

                # not-all trivial break the loop
                if num_ops_counter != 1:
                    break

            # if it is done
            jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
            done = True if np.prod(jobs_done) == 1 else False

            if done:
                break
        return m_list, cum_reward, done

    def get_available_machines(self, shuffle_machine: bool = True):
        """
        Get all available machines.

        Args:
            shuffle_machine: True indicates that the returned list order is scrambled.

        Returns:
            A list contains all available 'Machines'.
        """
        return self.machine_manager.get_available_machines(shuffle_machine)

    def get_doable_ops_in_dict(self, machine_id: int = None, shuffle_machine: bool = True):
        """
        Get all doable operations.

        Args:
            machine_id: None indicates get all doable operations. Else it will return specified operations.
            
            shuffle_machine:  True indicates that the returned value order is scrambled.

        Returns:
            A dict whose key is machine_id(int, start from 1), value is [op1_id, op2_id, ...].
        """
        if machine_id is None:
            doable_dict = {}
            if self.get_available_machines():
                for m in self.get_available_machines(shuffle_machine):
                    _id = m.machine_id
                    _ops = m.doable_ops_id
                    doable_dict[_id] = _ops
            ret = doable_dict
        else:
            available_machines = [m.machine_id for m in self.get_available_machines()]
            if machine_id in available_machines:
                ret = self.machine_manager[machine_id].doable_ops_id
            else:
                raise RuntimeWarning("Access to the not available machine {}. Return is None".format(machine_id))
        return ret

    def get_doable_ops_in_list(self, machine_id: int = None, shuffle_machine: bool = True):
        """
        Get all doable operations.

        Args:
            machine_id: None indicates get all doable operations. Else it will return specified operations.
            
            shuffle_machine: True indicates that the returned value order is scrambled.

        Returns:
            A list contains all doable operations.
        """
        doable_dict = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        do_ops = []
        for _, v in doable_dict.items():
            do_ops += v
        return do_ops

    def get_doable_ops(self, machine_id: int = None, return_list: bool = False, shuffle_machine: bool = True):
        """
        Get all doable operations.

        Args:
            machine_id: None indicates get all doable operations. Else it will return specified operations.
            
            return_list: True indicates that return list. Else return dict.
            
            shuffle_machine: True indicates that the returned value order is scrambled.

        Returns:
            A dict or list contains all doable operations.
        """
        if return_list:
            ret = self.get_doable_ops_in_list(machine_id, shuffle_machine)
        else:
            ret = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        return ret

    def observe(self, return_doable: bool = True):
        """
        Get observation of the environment.

        Returns:
            Returns two values. There are:

            * observation(OrderedDiGraph): one graph which describes the disjunctive graph and contained other info.
            * done(boolean): Whether it’s time to reset the environment again. True indicates the episode has terminated.
        """
        done = self.is_done()

        g = self.job_manager.observe(detach_done=self.detach_done)

        if return_doable:
            if self.use_surrogate_index:
                do_ops_list = self.get_doable_ops(return_list=True)
                for n in g.nodes:
                    if n in do_ops_list:
                        g.nodes[n]['doable'] = True
                    else:
                        g.nodes[n]['doable'] = False
                    job_id, op_id = self.job_manager.sur_index_dict[n]
                    m_id = self.job_manager[job_id][op_id].machine_id
                    g.nodes[n]['machine'] = m_id

        return g, done

    def eval_performance(self, performance: str = 'unsatety ratio'):
        """
        Evaluate the performance of this episode.

        Args:
            performance: the performance you want to evaluate.
            (unsafety ratio, unsafety number, makespan, utilization)

        Returns:
            Performance of this episode.
        """
        if performance == 'unsafety ratio':
            ret = self.num_unsafety / self.num_do_action
        elif performance == 'unsafety number':
            ret = self.num_unsafety
        elif performance == 'makespan':
            ret = self.global_time
        elif performance == 'utilization':
            ret = {}
            for machine in self.machine_manager:
                ret['Machine {}'.format(machine.machine_id)] = machine.total_work_time / machine.finish_time
        else:
            raise KeyError(
                "Don't know what is {} performance. Please use 'makespan', 'utilization', etc.".format(performance))

        return ret

    def get_doable_action(self):
        """
        Get doable action in list.
        """
        doable_ops = self.get_doable_ops_in_list()
        doable_ops.append(self.wait_action)
        return doable_ops

    def draw_gantt_chart(self, path, benchmark_name, max_x=None):
        """
        Draw the Gantt chart after the episode has terminated.

        Args:
            path: The path that saves the chart. Ends with 'html'.
            
            benchmark_name: The name of instance.
            
            max_x: X maximum(None indicates makespan + 50 )
        """
        if max_x is None:
            max_x = self.global_time + 50
        # Draw a gantt chart
        self.job_manager.draw_gantt_chart(path, benchmark_name, max_x)

    def g2s(self, g: nx.OrderedDiGraph):
        """
        Get the state from one disjunctive graph.

        Args:
            g: Disjunctive graph.

        Returns:
            The state that consists of vector.
        """
        state = np.empty(self.observation_space.shape, dtype=np.float32)
        nodelist = get_nodelist(g)
        for n in nodelist:
            state[n * 10 + 0] = g.nodes[n]['id'][0] / self.num_jobs  # job id
            state[n * 10 + 1] = g.nodes[n]['id'][1] / self.num_steps  # step id
            state[n * 10 + 2] = g.nodes[n]['type']
            state[n * 10 + 3] = g.nodes[n]['complete_ratio']
            state[n * 10 + 4] = g.nodes[n]['processing_time'] / self._max_pro_time
            state[n * 10 + 5] = g.nodes[n]['remaining_ops'] / self.num_steps
            state[n * 10 + 6] = min(1., g.nodes[n]['waiting_time'] / self._max_pro_time)
            state[n * 10 + 7] = g.nodes[n]['remain_time'] / self._max_pro_time
            state[n * 10 + 8] = g.nodes[n]['doable']
            state[n * 10 + 9] = g.nodes[n]['machine'] / self.num_machines
        return state

    def render(self, mode='human'):
        """You should render after all operations finished."""
        self.draw_gantt_chart('tmp.html', self.name)

    def close(self):
        """'Close' will delete the file generated by 'render'."""
        if os.path.exists('tmp.html'):
            os.remove('tmp.html')


class GraphJsspEnv(BasicJsspEnv):
    def __init__(self,
                 name: str = None,
                 num_jobs: int = None,
                 num_machines: int = None,
                 embedding_dim: int = 16,
                 delay: bool = False,
                 verbose: bool = False,
                 reward_type: str = 'idle_time'):
        """
        Args:
            name: The name of JSSP instance.(abz5-9, ft06, ft10, ft20, la01-la40, orb01-orb10, swv01-swv20, yn1-yn4)
            
            num_jobs: Number of jobs for the selected instance.
            
            num_machines: Number of machines for the selected instance.
            
            embedding_dim: Dimension of embedding.
            
            delay: True indicates environment allows you to choose a operations that needs to be processed on a busy machine.
            
            verbose: True indicates environment will print the info while start/end one operation. It's useful for debugging.
            
            reward_type: The way rewards are calculated.('idle_time', 'utilization')

        Examples:
            >>> env = GraphJsspEnv('ft06')

            or

            >>> env = GraphJsspEnv(num_jobs=6, num_machines=6)
        """
        self.feature_dim = 10
        super(GraphJsspEnv, self).__init__(
            num_machines=num_machines,
            num_jobs=num_jobs,
            detach_done=False,
            name=name,
            embedding_dim=embedding_dim,
            delay=delay,
            verbose=verbose,
            reward_type=reward_type,
        )
        self.observation_space = GraphState(self.num_jobs, self.num_machines, self.num_machines)

    def get_adjacent_matrix(self, g: nx.OrderedDiGraph) -> csr_matrix:
        """
        Get the compressed adjacency matrix from the graph.
        You can use 'todense()' or 'toarray()' then get the complete matrix.

        Args:
            g: Disjunctive graph.

        Returns:
            Compressed adjacency matrix.

        Examples:
            >>> env = GraphJsspEnv('ft06')
            >>> a = env.get_adjacent_matrix(g)
            >>> a = a.todense()
        """
        if not hasattr(self, 'A'):
            # shape of data and indices is n*(m-1)*2 + n*m*(n-1), shape of indptr is n*m + 1
            # but data can queeze to 1
            nodelist = [i for i in range(nx.number_of_nodes(g))]
            A = nx.adjacency_matrix(g, nodelist)
            self.A = A
        return self.A

    def get_disjunctive_matrix(self, g: nx.OrderedDiGraph) -> csr_matrix:
        """
        Get the compressed disjunctive edge matrix from the graph.
        You can use 'todense()' or 'toarray()' then get the complete matrix.

        Args:
            g: Disjunctive graph.

        Returns:
            Compressed disjunctive edge matrix.

        Examples:
            >>> env = GraphJsspEnv('ft06')
            >>> d = env.get_disjunctive_matrix(g)
            >>> d = d.todense()
        """
        if not hasattr(self, 'D'):
            # all len of data, row, col is n*(n-1)*m, but data can squeeze to 1
            # length = self.num_jobs*(self.num_jobs-1)*self.num_machine
            num_nodes = nx.number_of_nodes(g)

            nodelist = [i for i in range(num_nodes)]
            data = []
            row = []
            col = []
            for n in nodelist:
                neighbors = g.neighbors(n)
                for m in neighbors:
                    if g.edges[n, m]['type'] == DISJUNCTIVE_TYPE:
                        data.append(1)
                        row.append(n)
                        col.append(m)
            n, m = self.num_jobs, self.num_machines
            assert len(row) == n * m * (n - 1)
            self.D = csr_matrix((data, (row, col)),
                                shape=(n * m, n * m))

        return self.D

    def step(self, action):
        """
        Process a action contained in action space.

        Args:
            action: Contained in the action space.

        Returns:
            Returns four values. These are:
            
            * observation(object): An environment-specific object representing your observation of the environment.
            
            * reward(float): Amount of reward achieved by the previous action.
            
            * done(boolean): Whether it’s time to reset the environment again. True indicates the episode has terminated.
            
            * info(dict): Diagnostic information useful for debugging. It can sometimes be useful for learning.
        """
        doable_ops = self.get_doable_ops_in_list()
        self.transit(action)
        g, done = self.observe()
        reward = self.cal_reward(action, doable_ops=doable_ops)

        feature = self.__get_feature_from_g(g)
        observation = {
            'feature': feature,
            'A': self.get_adjacent_matrix(g),
            'D': self.get_disjunctive_matrix(g)
        }

        info = {
            'cost': self.num_unsafety,
            'wait': self.num_wait,
            'total action': self.num_do_action,
            'makespan': self.global_time
        }
        return observation, reward, done, info

    def reset(self, random_rate: float = 0., shuffle: bool = True, ret: bool = True):
        """
        Reset the environment.

        Args:
            random_rate: Each operation has a random_rate probability to randomly increase or decrease the process time.
            
            shuffle: True indicates each job has a random_rate probability to shuffle.

            ret: True indicates this function will returns the initial observation.
        Returns:
            The initial observation or None.
        """
        super(GraphJsspEnv, self).reset(ret=False, random_rate=random_rate, shuffle=shuffle)

        g, _ = self.observe()
        feature = self.__get_feature_from_g(g)
        if hasattr(self, 'A'):
            delattr(self, 'A')
            delattr(self, 'D')
        A = self.get_adjacent_matrix(g)
        D = self.get_disjunctive_matrix(g)

        observation = {
            'feature': feature,
            'A': A,
            'D': D
        }
        if ret:
            return observation

    def __get_feature_from_g(self, g: nx.OrderedDiGraph) -> np.ndarray:
        """
        Get the feature from one disjunctive graph.

        Args:
            g: Disjunctive graph.
        
        Return:
            The feature that consists of matrix.
        """
        feature = np.zeros((10, self.num_jobs * self.num_steps), dtype=np.float64)

        for n in range(nx.number_of_nodes(g)):
            feature[0, n] = g.nodes[n]['id'][0] / self.num_jobs  # job id
            feature[1, n] = g.nodes[n]['id'][1] / self.num_steps  # step id
            feature[2, n] = g.nodes[n]['type']
            feature[3, n] = g.nodes[n]['complete_ratio']
            feature[4, n] = g.nodes[n]['processing_time'] / self._max_pro_time
            feature[5, n] = g.nodes[n]['remaining_ops'] / self.num_steps
            feature[6, n] = min(1., g.nodes[n]['waiting_time'] / self._max_pro_time)
            feature[7, n] = g.nodes[n]['remain_time'] / self._max_pro_time
            feature[8, n] = g.nodes[n]['doable']
            feature[9, n] = g.nodes[n]['machine'] / self.num_machines
        return feature


class HeuristicJsspEnv(BasicJsspEnv):
    def __init__(self,
                 name: str = None,
                 num_jobs: int = None,
                 num_machines: int = None,
                 embedding_dim: int = 16,
                 delay: bool = False,
                 verbose: bool = False,
                 reward_type: str = 'idle_time',
                 schedule_cycle: int = 1):
        """
        Args:
            name: The name of JSSP instance.(abz5-9, ft06, ft10, ft20, la01-la40, orb01-orb10, swv01-swv20, yn1-yn4).
            
            num_jobs: Number of jobs for the selected instance.
            
            num_machines: Number of machines for the selected instance.
            
            embedding_dim: Dimension of embedding.
            
            delay: True indicates environment allows you to choose a operations that needs to be processed on a busy machine.
            
            verbose: True indicates environment will print the info while start/end one operation. It's useful for debugging.
            
            reward_type: The way rewards are calculated.('idle_time', 'utilization').
            
            schedule_cycle: when you choice an action(heuristic method), this method will process next 'schedule_cycle' second.

        Examples:
            >>> env = HeuristicJsspEnv('ft06')

            or

            >>> env = HeuristicJsspEnv(num_jobs=6, num_machines=6)
        """
        super(HeuristicJsspEnv, self).__init__(
            num_machines=num_machines,
            num_jobs=num_jobs,
            detach_done=False,
            name=name,
            embedding_dim=embedding_dim,
            delay=delay,
            verbose=verbose,
            reward_type=reward_type,
        )
        self.schedule_cycle = schedule_cycle
        self.action_space = spaces.Discrete(8)

    def transit(self, action: int):
        """
        Process a action contained in action space and return reward. But differ from step, this is part of 'step()'.

        Args:
            action: Contained in the action space.

        Returns:
            Amount of reward achieved by this action.
        """
        if self.is_done():
            return 0
        self.num_do_action += 1

        if action == 0:
            r = fifo(self)
        elif action == 1:
            r = lifo(self)
        elif action == 2:
            r = lpt(self)
        elif action == 3:
            r = spt(self)
        elif action == 4:
            r = ltpt(self)
        elif action == 5:
            r = stpt(self)
        elif action == 6:
            r = mor(self)
        elif action == 7:
            r = lor(self)
        else:
            raise RuntimeError('action {} out of action space.'.format(action))
        return r

    def get_machine_doable_action(self):
        """Get all doable actions in all machines."""
        s = [[] for _ in range(self.num_machines)]
        available_machines = self.get_available_machines()
        for machine in available_machines:
            s[machine.machine_id - 1] = machine.doable_ops_id

        return s

    def step(self, action):
        """
        Process a action contained in action space.

        Args:
            action: Contained in the action space.

        Returns:
            Returns four values. These are:
            
            * observation(object): An environment-specific object representing your observation of the environment.
            
            * reward(float): Amount of reward achieved by the previous action.
            
            * done(boolean): Whether it’s time to reset the environment again. True indicates the episode has terminated.
            
            * info(dict): Diagnostic information useful for debugging. It can sometimes be useful for learning.
        """
        reward = self.transit(action)
        g, done = self.observe()

        observation = self.g2s(g)

        info = {
            'cost': self.num_unsafety,
            'wait': self.num_wait,
            'total action': self.num_do_action,
            'makespan': self.global_time
        }

        return observation, reward, done, info

    def get_process_time(self, ops_id: int):
        """Get the processing time of the operation."""
        if self.use_surrogate_index:
            if ops_id in self.job_manager.sur_index_dict.keys():
                job_id, step_id = self.job_manager.sur_index_dict[ops_id]
            else:
                raise RuntimeError("Input action is not valid")
        else:
            job_id, step_id = ops_id

        operation = self.job_manager[job_id][step_id]
        return operation.processing_time

    def get_job_process_time(self, ops_id: int):
        """Get the processing time of the job that contains the operation."""
        if self.use_surrogate_index:
            if ops_id in self.job_manager.sur_index_dict.keys():
                job_id, step_id = self.job_manager.sur_index_dict[ops_id]
            else:
                raise RuntimeError("Input action is not valid")
        else:
            job_id, step_id = ops_id

        job = self.job_manager[job_id]
        return job.processing_time

    def get_num_remain_ops(self, ops_id):
        """Get the number of remain operatios of the job that contains the operation."""
        if self.use_surrogate_index:
            if ops_id in self.job_manager.sur_index_dict.keys():
                job_id, step_id = self.job_manager.sur_index_dict[ops_id]
            else:
                raise RuntimeError("Input action is not valid")
        else:
            job_id, step_id = ops_id

        job = self.job_manager[job_id]
        return job.remaining_ops

    def get_available_machines_id(self, shuffle_machine: bool = True):
        """Get all available machines' id"""
        available_machines = self.get_available_machines(shuffle_machine)
        available_machines_id = []

        for machine in available_machines:
            available_machines_id.append(machine.machine_id - 1)

        return available_machines_id


class HeuristicGraphJsspEnv(HeuristicJsspEnv):
    def __init__(self,
                 name: str = None,
                 num_jobs: int = None,
                 num_machines: int = None,
                 embedding_dim: int = 16,
                 delay: bool = False,
                 verbose: bool = False,
                 reward_type: str = 'idle_time',
                 schedule_cycle: int = 1):
        """
        Args:
            name: The name of JSSP instance.(abz5-9, ft06, ft10, ft20, la01-la40, orb01-orb10, swv01-swv20, yn1-yn4).
            
            num_jobs: Number of jobs for the selected instance.
            
            num_machines: Number of machines for the selected instance.
            
            embedding_dim: Dimension of embedding.
            
            delay: True indicates environment allows you to choose a operations that needs to be processed on a busy machine.
            
            verbose: True indicates environment will print the info while start/end one operation. It's useful for debugging.
            
            reward_type: The way rewards are calculated.('idle_time', 'utilization').
            
            schedule_cycle: when you choice an action(heuristic method), this method will process next 'schedule_cycle' second.

        Examples:
            >>> env = HeuristicGraphJsspEnv('ft06')

            or

            >>> env = HeuristicGraphJsspEnv(num_jobs=6, num_machines=6)
        """
        self.feature_dim = 10
        super(HeuristicGraphJsspEnv, self).__init__(
            num_machines=num_machines,
            num_jobs=num_jobs,
            name=name,
            embedding_dim=embedding_dim,
            delay=delay,
            verbose=verbose,
            reward_type=reward_type,
            schedule_cycle=schedule_cycle,
        )
        self.observation_space = GraphState(self.num_jobs, self.num_machines, self.num_machines)

    def reset(self, random_rate: float = 0., shuffle: bool = True, ret: bool = True):
        """
        Reset the environment.

        Args:
            random_rate: Each operation has a random_rate probability to randomly increase or decrease the process time.
            
            shuffle: True indicates each job has a random_rate probability to shuffle.

            ret: True indicates this function will returns the initial observation.
        Returns:
            The initial observation or None.
        """
        super(HeuristicGraphJsspEnv, self).reset(ret=False, random_rate=random_rate, shuffle=shuffle)

        g, _ = self.observe()
        feature = self.__get_feature_from_g(g)
        if hasattr(self, 'A'):
            delattr(self, 'A')
            delattr(self, 'D')
        A = self.get_adjacent_matrix(g)
        D = self.get_disjunctive_matrix(g)

        observation = {
            'feature': feature,
            'A': A,
            'D': D
        }
        if ret:
            return observation

    def get_adjacent_matrix(self, g: nx.OrderedDiGraph) -> csr_matrix:
        """
        Get the compressed adjacency matrix from the graph.
        You can use 'todense()' or 'toarray()' then get the complete matrix.

        Args:
            g: Disjunctive graph.

        Returns:
            Compressed adjacency matrix.

        Examples:
            >>> env = HeuristicGraphJsspEnv('ft06')
            >>> A = env.get_adjacent_matrix(g)
            >>> A = A.todense()
        """
        if not hasattr(self, 'A'):
            # shape of data and indices is n*(m-1)*2 + n*m*(n-1), shape of indptr is n*m + 1
            # but data can queeze to 1
            nodelist = [i for i in range(nx.number_of_nodes(g))]
            A = nx.adjacency_matrix(g, nodelist)
            self.A = A

        return self.A

    def get_disjunctive_matrix(self, g: nx.OrderedDiGraph) -> csr_matrix:
        """
        Get the compressed disjunctive edge matrix from the graph.
        You can use 'todense()' or 'toarray()' then get the complete matrix.

        Args:
            g: Disjunctive graph.

        Returns:
            Compressed disjunctive edge matrix.

        Examples:
            >>> env = HeuristicGraphJsspEnv('ft06')
            >>> D = env.get_disjunctive_matrix(g)
            >>> D = D.todense()
        """
        if not hasattr(self, 'D'):
            # all len of data, row, col is n*(n-1)*m, but data can queeze to 1
            # length = self.num_jobs*(self.num_jobs-1)*self.num_machine
            num_nodes = nx.number_of_nodes(g)

            nodelist = [i for i in range(num_nodes)]
            data = []
            row = []
            col = []
            for n in nodelist:
                neighbors = g.neighbors(n)
                for m in neighbors:
                    if g.edges[n, m]['type'] == DISJUNCTIVE_TYPE:
                        data.append(1)
                        row.append(n)
                        col.append(m)
            n, m = self.num_jobs, self.num_machines
            assert len(row) == n * m * (n - 1)
            self.D = csr_matrix((data, (row, col)),
                                shape=(n * m, n * m))

        return self.D

    def step(self, action):
        """
        Process a action contained in action space.

        Args:
            action: Contained in the action space.

        Returns:
            Returns four values. These are:
            * observation(object): An environment-specific object representing your observation of the environment.
            
            * reward(float): Amount of reward achieved by the previous action.
            
            * done(boolean): Whether it’s time to reset the environment again. True indicates the episode has terminated.
            
            * info(dict): Diagnostic information useful for debugging. It can sometimes be useful for learning.
        """
        reward = self.transit(action)
        g, done = self.observe()

        feature = self.__get_feature_from_g(g)
        observation = {
            'feature': feature,
            'A': self.get_adjacent_matrix(g),
            'D': self.get_disjunctive_matrix(g)
        }

        info = {
            'cost': self.num_unsafety,
            'wait': self.num_wait,
            'total action': self.num_do_action,
            'makespan': self.global_time
        }

        return observation, reward, done, info

    def __get_feature_from_g(self, g: nx.OrderedDiGraph) -> np.ndarray:
        """
        Get the feature from one disjunctive graph.

        Args:
            g: Disjunctive graph.

        Return:
            The feature that consists of matrix. Shape (number of feature, total steps)
        """
        feature = np.zeros((10, self.num_jobs * self.num_steps), dtype=np.float64)

        for n in range(nx.number_of_nodes(g)):
            feature[0, n] = g.nodes[n]['id'][0] / self.num_jobs  # job id
            feature[1, n] = g.nodes[n]['id'][1] / self.num_steps  # step id
            feature[2, n] = g.nodes[n]['type']
            feature[3, n] = g.nodes[n]['complete_ratio']
            feature[4, n] = g.nodes[n]['processing_time'] / self._max_pro_time
            feature[5, n] = g.nodes[n]['remaining_ops'] / self.num_steps
            feature[6, n] = min(1., g.nodes[n]['waiting_time'] / self._max_pro_time)
            feature[7, n] = g.nodes[n]['remain_time'] / self._max_pro_time
            feature[8, n] = g.nodes[n]['doable']
            feature[9, n] = g.nodes[n]['machine'] / self.num_machines
        return feature


class HeuristicAttentionJsspEnv(HeuristicJsspEnv):
    def __init__(self,
                 name: str = None,
                 num_jobs: int = None,
                 num_machines: int = None,
                 embedding_dim: int = 16,
                 delay: bool = False,
                 verbose: bool = False,
                 reward_type: str = 'idle_time',
                 schedule_cycle: int = 1):
        """
        Args:
            name: The name of JSSP instance.(abz5-9, ft06, ft10, ft20, la01-la40, orb01-orb10, swv01-swv20, yn1-yn4).
            
            num_jobs: Number of jobs for the selected instance.
            
            num_machines: Number of machines for the selected instance.
            
            embedding_dim: Dimension of embedding.
            
            delay: True indicates environment allows you to choose a operations that needs to be processed on a busy machine.
            
            verbose: True indicates environment will print the info while start/end one operation. It's useful for debugging.
            
            reward_type: The way rewards are calculated.('idle_time', 'utilization').
            
            schedule_cycle: when you choice an action(heuristic method), this method will process next 'schedule_cycle' second.

        Examples:
            >>> env = HeuristicAttentionJsspEnv('ft06')

            or

            >>> env = HeuristicAttentionJsspEnv(num_jobs=6, num_machines=6)
        """
        self.feature_dim = 10
        super(HeuristicAttentionJsspEnv, self).__init__(
            num_machines=num_machines,
            num_jobs=num_jobs,
            name=name,
            embedding_dim=embedding_dim,
            delay=delay,
            verbose=verbose,
            reward_type=reward_type,
            schedule_cycle=schedule_cycle,
        )
        self.observation_space = AttentionState(self.num_jobs, self.num_steps, self.num_machines)

    def reset(self, random_rate: float = 0., shuffle: bool = True, ret: bool = True) -> np.ndarray:
        """
        Reset the environment.

        Args:
            random_rate: Each operation has a random_rate probability to randomly increase or decrease the process time.
            
            shuffle: True indicates each job has a random_rate probability to shuffle.

            ret: True indicates this function will returns the initial observation.
        Returns:
            The initial observation or None.
        """
        super(HeuristicJsspEnv, self).reset(ret=False, random_rate=random_rate, shuffle=shuffle)

        g, _ = self.observe()
        observation = self.__get_feature_from_g(g)

        if ret:
            return observation

    def step(self, action):
        """
        Process a action contained in action space.

        Args:
            action: Contained in the action space.

        Returns:
            Returns four values. These are:
            
            * observation(object): An environment-specific object representing your observation of the environment.
            
            * reward(float): Amount of reward achieved by the previous action.
            
            * done(boolean): Whether it’s time to reset the environment again. True indicates the episode has terminated.
            
            * info(dict): Diagnostic information useful for debugging. It can sometimes be useful for learning.
        """
        reward = self.transit(action)
        g, done = self.observe()

        state = self.__get_feature_from_g(g)

        info = {
            'cost': self.num_unsafety,
            'wait': self.num_wait,
            'total action': self.num_do_action,
            'makespan': self.global_time
        }

        return state, reward, done, info

    def __get_feature_from_g(self, g: nx.OrderedDiGraph) -> np.ndarray:
        """
        Get the feature from one disjunctive graph.

        Args:
            g: Disjunctive graph.

        Return:
            The feature that consists of matrix. Shape (total steps, number of feature)
        """
        feature = np.zeros((self.num_jobs * self.num_steps, self.feature_dim), dtype=np.float64)

        for n in range(nx.number_of_nodes(g)):
            feature[n, 0] = g.nodes[n]['id'][0] / self.num_jobs  # job id
            feature[n, 1] = g.nodes[n]['id'][1] / self.num_steps  # step id
            feature[n, 2] = g.nodes[n]['type']
            feature[n, 3] = g.nodes[n]['complete_ratio']
            feature[n, 4] = g.nodes[n]['processing_time'] / self._max_pro_time
            feature[n, 5] = g.nodes[n]['remaining_ops'] / self.num_steps
            feature[n, 6] = min(1., g.nodes[n]['waiting_time'] / self._max_pro_time)
            feature[n, 7] = g.nodes[n]['remain_time'] / self._max_pro_time
            feature[n, 8] = g.nodes[n]['doable']
            feature[n, 9] = g.nodes[n]['machine'] / self.num_machines
        return feature


class BasicState(gym.Space):
    """
    Observation space built on JSSP instances provided by OR-Liberty.
    The features of each node are expanded into 1D vectors and concatenated together as the basic state. 
    Each node contains the following characteristics:
    
    * 'job_id'(int): Describe which job this operation belongs to.
    
    * 'step_id'(int): Describe the sequence of the operation.
    
    * 'type'(int): -1 indicates unfinished, 0 indicates in process, and 1 indicates completed.

    * 'complete_ratio'(float): Represents the completion rate of the whole workpiece when the operation is completed.

    * 'processing_time'(int): Represents the processing time required to complete the operation.
    
    * 'remaining_ops'(int): Number of subsequent operations.

    * 'waiting_time'(int): The waiting time is calculated from the beginning when the process can be processed.
    
    * 'remain_time'(int): The remaining processing time, when not in processing, is 0.
    
    * 'doable'(boolean): True indicates that the operation is currently doable.
    
    * 'machine'(int): Describe the machine on which the operation needs to be processed. 0 if the process cannot be done; The machine id starts at 1.
    """

    def __init__(self, num_jobs, num_steps, num_machines) -> None:
        self.num_jobs = num_jobs
        self.num_steps = num_steps
        self.num_ops = num_jobs * num_steps
        self.num_machines = num_machines
        self.num_feature = 10
        super().__init__(shape=(self.num_ops * self.num_feature,), dtype=np.float32)

    def sample(self):
        """Sample from this space."""
        sample = np.ndarray(self.shape, dtype=self.dtype)
        for op_id in range(self.num_ops):
            sample[op_id + 0] = self.np_random.random()  # job_id
            sample[op_id + 1] = self.np_random.random()  # step_id
            sample[op_id + 2] = self.np_random.randint(-1, 2, dtype=self.dtype)  # type
            sample[op_id + 3] = self.np_random.random()  # complete_ratio
            sample[op_id + 4] = self.np_random.random()  # processing_time
            sample[op_id + 5] = self.np_random.random()  # remaining_ops
            sample[op_id + 6] = self.np_random.random()  # waiting_time
            sample[op_id + 7] = self.np_random.random()  # remain_time
            sample[op_id + 8] = self.np_random.randint(2, dtype=self.dtype)  # doable
            sample[op_id + 9] = self.np_random.random()  # machine

        return sample

    def contains(self, x) -> bool:
        """Determine whether x is included in this space."""
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check

        return x.shape == self.shape and np.all(x <= 1) and np.all(x >= -1)

    def __repr__(self):
        return "Jssp State({}, {}, {})".format(self.num_jobs, self.num_steps, self.num_machines)

    def __eq__(self, other):
        return isinstance(other, BasicState) and self.num_jobs == other.num_jobs and \
               self.num_machines == other.num_machines and self.num_steps == other.num_steps


class GraphState(gym.spaces.Dict):
    """
    Observation space built on JSSP instances provided by OR-Liberty.
    The features of each node form a matrix(features dim, number of nodes) as the graph feature.
    The state will alse contains the adjacency matrix and disjunctive edge matrix.
    Each node contains the following characteristics:
    
    * 'job_id'(int): Describe which job this operation belongs to.
    
    * 'step_id'(int): Describe the sequence of the operation.
    
    * 'type'(int): -1 indicates unfinished, 0 indicates in process, and 1 indicates completed.

    * 'complete_ratio'(float): Represents the completion rate of the whole workpiece when the operation is completed.

    * 'processing_time'(int): Represents the processing time required to complete the operation.
    
    * 'remaining_ops'(int): Number of subsequent operations.

    * 'waiting_time'(int): The waiting time is calculated from the beginning when the process can be processed.
    
    * 'remain_time'(int): The remaining processing time, when not in processing, is 0.
    
    * 'doable'(boolean): True indicates that the operation is currently doable.
    
    * 'machine'(int): Describe the machine on which the operation needs to be processed. 0 if the process cannot be done; The machine id starts at 1.
    """

    def __init__(self, num_jobs, num_steps, num_machines) -> None:
        self.num_jobs = num_jobs
        self.num_steps = num_steps
        self.num_ops = num_jobs * num_steps
        self.num_machines = num_machines
        self.num_feature = 10
        super().__init__(None)

    def sample(self):
        """Sample from this space."""
        num_nodes = self.num_jobs * self.num_steps
        nodelist = [i for i in range(num_nodes)]
        feature = np.ndarray(
            (self.num_feature, num_nodes),
            dtype=np.float64
        )
        data = []
        row = []
        col = []
        for n in nodelist:
            for m in np.random.randint(1, num_nodes / 2):
                data.append(1)
                row.append(n)
                col.append(m)
        n = self.num_jobs
        m = self.num_steps
        A = csr_matrix((data, (row, col)),
                       shape=(n * m, n * m))
        D = csr_matrix((data, (row, col)),
                       shape=(n * m, n * m))
        sample = {
            'feature': feature,
            'A': A,
            'D': D,
        }
        return sample

    def contains(self, x):
        """Determine whether x is included in this space."""
        if not isinstance(x, dict):
            return False
        num_nodes = self.num_jobs * self.num_steps
        has_f = 'feature' in x.keys()
        has_A = 'A' in x.keys()
        has_D = 'D' in x.keys()
        if not (has_f and has_A and has_D):
            return False

        shape_feature = x['feature'].shape != (self.num_feature, num_nodes)
        shape_A = x['A'].todense().shape != (num_nodes, num_nodes)
        shape_D = x['D'].todense().shape != (num_nodes, num_nodes)
        if not (shape_feature and shape_A and shape_D):
            return False
        return True

    def __repr__(self):
        return "Jssp Graph State({}, {}, {})".format(self.num_jobs, self.num_steps, self.num_machines)

    def __eq__(self, other):
        return isinstance(other, GraphState) and self.num_jobs == other.num_jobs and \
               self.num_machines == other.num_machines and self.num_steps == other.num_steps


class AttentionState(spaces.Box):
    """
    Observation space built on JSSP instances provided by OR-Liberty.
    The features of each node form a matrix(number of nodes, features dim) as the graph feature.
    Each node contains the following characteristics:
    
    * 'job_id'(int): Describe which job this operation belongs to.
    
    * 'step_id'(int): Describe the sequence of the operation.
    
    * 'type'(int): -1 indicates unfinished, 0 indicates in process, and 1 indicates completed.

    * 'complete_ratio'(float): Represents the completion rate of the whole workpiece when the operation is completed.

    * 'processing_time'(int): Represents the processing time required to complete the operation.
    
    * 'remaining_ops'(int): Number of subsequent operations.

    * 'waiting_time'(int): The waiting time is calculated from the beginning when the process can be processed.
    
    * 'remain_time'(int): The remaining processing time, when not in processing, is 0.
    
    * 'doable'(boolean): True indicates that the operation is currently doable.
    
    * 'machine'(int): Describe the machine on which the operation needs to be processed. 0 if the process cannot be done; The machine id starts at 1.
    """

    def __init__(self, num_jobs, num_steps, num_machines) -> None:
        self.num_jobs = num_jobs
        self.num_steps = num_steps
        self.num_ops = num_jobs * num_steps
        self.num_machines = num_machines
        self.num_feature = 10
        super().__init__(low=-1.0, high=1.0, shape=(self.num_ops, self.num_feature,), dtype=np.float32)

    def sample(self):
        """Sample from this space."""
        sample = np.ndarray(self.shape, dtype=self.dtype)
        for op_id in range(self.num_ops):
            sample[op_id, 0] = self.np_random.random()  # job_id
            sample[op_id, 1] = self.np_random.random()  # step_id
            sample[op_id, 2] = self.np_random.randint(-1, 2, dtype=self.dtype)  # type
            sample[op_id, 3] = self.np_random.random()  # complete_ratio
            sample[op_id, 4] = self.np_random.random()  # processing_time
            sample[op_id, 5] = self.np_random.random()  # remaining_ops
            sample[op_id, 6] = self.np_random.random()  # waiting_time
            sample[op_id, 7] = self.np_random.random()  # remain_time
            sample[op_id, 8] = self.np_random.randint(2, dtype=self.dtype)  # doable
            sample[op_id, 9] = self.np_random.random()  # machine

        return sample

    def contains(self, x):
        """Determine whether x is included in this space."""
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check

        return x.shape == self.shape and np.all(x <= 1) and np.all(x >= -1)

    def __repr__(self):
        return "Jssp State({}, {}, {})".format(self.num_jobs, self.num_steps, self.num_machines)

    def __eq__(self, other):
        return isinstance(other, AttentionState) and self.num_jobs == other.num_jobs and \
               self.num_machines == other.num_machines and self.num_steps == other.num_steps
