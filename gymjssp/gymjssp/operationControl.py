import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from collections import OrderedDict

from plotly.offline import plot

from .configs import (NOT_START_NODE,
                      PROCESSING_NODE,
                      DONE_NODE,
                      DELAYED_NODE,
                      DUMMY_NODE,
                      CONJUNCTIVE_TYPE,
                      DISJUNCTIVE_TYPE,
                      FORWARD,
                      BACKWARD)


def get_edge_color_map(g, edge_type_color_dict=None):
    if edge_type_color_dict is None:
        edge_type_color_dict = OrderedDict()
        edge_type_color_dict[CONJUNCTIVE_TYPE] = 'k'
        edge_type_color_dict[DISJUNCTIVE_TYPE] = '#F08080'

    colors = []
    for e in g.edges:
        edge_type = g.edges[e]['type']
        colors.append(edge_type_color_dict[edge_type])
    return colors


def calc_positions(g, half_width=None, half_height=None):
    if half_width is None:
        half_width = 30
    if half_height is None:
        half_height = 10

    min_idx = min(g.nodes)
    max_idx = max(g.nodes)

    num_horizontals = max_idx[1] - min_idx[1] + 1
    num_verticals = max_idx[0] - min_idx[1] + 1

    def xidx2coord(x):
        return np.linspace(-half_width, half_width, num_horizontals)[x]

    def yidx2coord(y):
        return np.linspace(-half_height, half_height, num_verticals)[y]

    pos_dict = OrderedDict()
    for n in g.nodes:
        pos_dict[n] = np.array((xidx2coord(n[1]), yidx2coord(n[0])))
    return pos_dict


def get_node_color_map(g, node_type_color_dict=None):
    if node_type_color_dict is None:
        node_type_color_dict = OrderedDict()
        node_type_color_dict[NOT_START_NODE] = '#F0E68C'
        node_type_color_dict[PROCESSING_NODE] = '#ADFF2F'
        node_type_color_dict[DELAYED_NODE] = '#829DC9'
        node_type_color_dict[DONE_NODE] = '#E9E9E9'
        node_type_color_dict[DUMMY_NODE] = '#FFFFFF'

    colors = []
    for n in g.nodes:
        node_type = g.nodes[n]['type']
        colors.append(node_type_color_dict[node_type])
    return colors


class JobSet:
    """
    Manage all machines. You can directly use ID to access corresponding Job. 

    Attributes:
        jobs(OrderedDict): Map the job ID(int) to the corresponding Job. 
        
        sur_index_dict(dict): The operations contained in the Job are numbered sequentially 
        and the mapping of the numbers to the Op is constructed.
    """

    def __init__(self,
                 machine_matrix: np.ndarray,
                 processing_time_matrix: np.ndarray,
                 embedding_dim: int = 16):
        """
        Args:
            machine_matrix: Machine processing matrix from OR-Liberty.
            
            processing_time_matrix: Processing time matrix from OR-Liberty.
            
            embedding_dim: Embedding dimension.
        """
        machine_matrix = machine_matrix.astype(int)
        processing_time_matrix = processing_time_matrix.astype(float)

        self.jobs = OrderedDict()

        # Constructing conjunctive edges
        for job_i, (m, pr_t) in enumerate(zip(machine_matrix, processing_time_matrix)):
            m = m + 1  # To make machine index starts from 1
            self.jobs[job_i] = Job(job_i, m, pr_t, embedding_dim)

        # Constructing disjunctive edges
        machine_index = list(set(machine_matrix.flatten().tolist()))
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            for job_id1, step_id1 in zip(job_ids, step_ids):
                op1 = self.jobs[job_id1][step_id1]
                ops = []
                for job_id2, step_id2 in zip(job_ids, step_ids):
                    if (job_id1 == job_id2) and (step_id1 == step_id2):
                        continue  # skip itself
                    else:
                        ops.append(self.jobs[job_id2][step_id2])
                op1.disjunctive_ops = ops

        self.use_surrogate_index = True

        if self.use_surrogate_index:
            # Constructing surrogate indices:
            num_ops = 0
            self.sur_index_dict = dict()
            for job_id, job in self.jobs.items():
                for op in job.ops:
                    op.sur_id = num_ops
                    self.sur_index_dict[num_ops] = op._id
                    num_ops += 1

    def __call__(self, index):
        return self.jobs[index]

    def __getitem__(self, index):
        return self.jobs[index]

    def observe(self, detach_done=True) -> nx.OrderedDiGraph:
        """
        Args:
            detach_done: True indicates observation will contain no information about completed operations.

        Returns:
            One graph which describes the disjunctive graph and contained other info.
        """

        g = nx.OrderedDiGraph()
        for job_id, job in self.jobs.items():
            for op in job.ops:
                not_start_cond = not (op == job.ops[0])
                not_end_cond = not (op == job.ops[-1])

                done_cond = op.x['type'] == DONE_NODE

                if detach_done:
                    if not done_cond:
                        g.add_node(op.id, **op.x)
                        if not_end_cond:  # Construct forward flow conjunctive edges only
                            g.add_edge(op.id, op.next_op.id,
                                       processing_time=op.processing_time,
                                       type=CONJUNCTIVE_TYPE,
                                       direction=FORWARD)

                            for disj_op in op.disjunctive_ops:
                                if disj_op.x['type'] != DONE_NODE:
                                    g.add_edge(op.id, disj_op.id, type=DISJUNCTIVE_TYPE)

                        if not_start_cond:
                            if op.prev_op.x['type'] != DONE_NODE:
                                g.add_edge(op.id, op.prev_op.id,
                                           processing_time=-1 * op.prev_op.processing_time,
                                           type=CONJUNCTIVE_TYPE,
                                           direction=BACKWARD)
                else:
                    g.add_node(op.id, **op.x)
                    if not_end_cond:  # Construct forward flow conjunctive edges only
                        g.add_edge(op.id, op.next_op.id,
                                   processing_time=op.processing_time,
                                   type=CONJUNCTIVE_TYPE,
                                   direction=FORWARD)

                    if not_start_cond:
                        g.add_edge(op.id, op.prev_op.id,
                                   processing_time=-1 * op.prev_op.processing_time,
                                   type=CONJUNCTIVE_TYPE,
                                   direction=BACKWARD)

                    for disj_op in op.disjunctive_ops:
                        g.add_edge(op.id, disj_op.id, type=DISJUNCTIVE_TYPE)

        return g

    def plot_graph(self, draw: bool = True,
                   node_type_color_dict: dict = None,
                   edge_type_color_dict: dict = None,
                   half_width=None,
                   half_height=None,
                   **kwargs):
        """
        Draw disjunctive graph.

        Args:
            draw: True indicates show the graph.
            
            node_type_color_dict: An dict contains node color.
            
            edge_type_color_dict: An dict contains edge color.
            
            half_width: Half of width.
            
            half_height: Half of height.
        """
        g = self.observe()
        node_colors = get_node_color_map(g, node_type_color_dict)
        edge_colors = get_edge_color_map(g, edge_type_color_dict)
        pos = calc_positions(g, half_width, half_height)

        if not kwargs:
            kwargs['figsize'] = (10, 5)
            kwargs['dpi'] = 300

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1, 1, 1)

        nx.draw(g, pos,
                node_color=node_colors,
                edge_color=edge_colors,
                with_labels=True,
                ax=ax)
        if draw:
            plt.show()
        else:
            return fig, ax

    def draw_gantt_chart(self, path, benchmark_name, max_x):
        """
        Draw the Gantt chart after the episode has terminated.

        Args:
            path: The path that saves the chart. Ends with 'html'.
            
            benchmark_name: The name of instance.
            
            max_x: X maximum(None indicates makespan + 50 )
        """
        random.seed(1)
        gantt_info = []
        for _, job in self.jobs.items():
            for op in job.ops:
                if not isinstance(op, DummyOperation):
                    temp = OrderedDict()
                    temp['Task'] = "Machine" + str(op.machine_id)
                    temp['Start'] = op.start_time
                    temp['Finish'] = op.end_time
                    temp['Resource'] = "Job" + str(op.job_id)
                    gantt_info.append(temp)
        gantt_info = sorted(gantt_info, key=lambda k: k['Task'])
        color = OrderedDict()
        for g in gantt_info:
            _r = random.randrange(0, 255, 1)
            _g = random.randrange(0, 255, 1)
            _b = random.randrange(0, 255, 1)
            rgb = 'rgb({}, {}, {})'.format(_r, _g, _b)
            color[g['Resource']] = rgb
        if gantt_info:
            fig = ff.create_gantt(gantt_info, colors=color, show_colorbar=True, group_tasks=True, index_col='Resource',
                                  title=benchmark_name + ' gantt chart', showgrid_x=True, showgrid_y=True)
            fig['layout']['xaxis'].update({'type': None})
            fig['layout']['xaxis'].update({'range': [0, max_x]})
            fig['layout']['xaxis'].update({'title': 'time'})

            plot(fig, filename=path)


class Job:
    """
    The simulation job.

    Attributes:
        job_id(int): Job ID.
        ops(list): CA list of all the Operations that belong to the Job.
        processing_time(int): The total time required to complete the job.
        num_sequence(int): The number of steps involved in the job.
    """

    def __init__(self, job_id: int, machine_order: np.ndarray, processing_time_order: np.ndarray, embedding_dim):
        """
        Args:
            job_id: Job ID.
            machine_order: A list of machines' ID required for each operation.
            processing_time_order: A list of the processing time required for each operation.
            embedding_dim: Embedding dimension.
        """
        self.job_id = job_id
        self.ops = list()
        self.processing_time = np.sum(processing_time_order)
        self.num_sequence = processing_time_order.size
        # Connecting backward paths (add prev_op to operations)
        cum_pr_t = 0
        for step_id, (m_id, pr_t) in enumerate(zip(machine_order, processing_time_order)):
            cum_pr_t += pr_t
            op = Operation(job_id=job_id, step_id=step_id, machine_id=m_id,
                           prev_op=None,
                           processing_time=pr_t,
                           complete_ratio=cum_pr_t / self.processing_time,
                           job=self)
            self.ops.append(op)
        for i, op in enumerate(self.ops[1:]):
            op.prev_op = self.ops[i]

        # Connecting forward paths (add next_op to operations)
        for i, node in enumerate(self.ops[:-1]):
            node.next_op = self.ops[i + 1]

    def __getitem__(self, index):
        return self.ops[index]

    @property
    def job_done(self):
        if self.ops[-1].node_status == DONE_NODE:
            return True
        else:
            return False

    @property
    def remaining_ops(self):
        c = 0
        for op in self.ops:
            if op.node_status != DONE_NODE:
                c += 1
        return c


class DummyOperation:
    """
    A operation with no real meaning.
    """

    def __init__(self,
                 job_id: int,
                 step_id: int,
                 embedding_dim: int):
        self.job_id = job_id
        self.step_id = step_id
        self._id = (job_id, step_id)
        self.machine_id = 'NA'
        self.processing_time = 0
        self.embedding_dim = embedding_dim
        self.built = False
        self.type = DUMMY_NODE
        self._x = {'type': self.type}
        self.node_status = DUMMY_NODE
        self.remaining_time = 0

    @property
    def id(self):
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id


class Operation:
    """
    The simulation operation.

    Attributes:
        * job_id(int): The job ID to which the operation belongs.
        
        * job: The job to which the operation belongs.
        
        * step_id(int): The step_id step of the job.
        
        * machine_id(int): This operation needs to be processed on machine_id machine.
        
        * processing_time(int): The time required to complete the process.
        
        * delayed_time: Delay time.
        
        * remaining_time(int): The remaining completion time of the operation while the operation is in process.
          In the raw state, it is always 0.
        
        * waiting_time(int): Waiting time.

        * node_status(int): Identifies the node state.
        
        * remaining_ops(int): Number of remaining operations.
        
        * disjunctive_ops(list): The operation of processing all in same machine

        * next_op: Next operation.
    """

    def __init__(self,
                 job_id: int,
                 step_id: int,
                 machine_id: int,
                 complete_ratio: float,
                 prev_op,
                 processing_time: int,
                 job: Job,
                 next_op=None,
                 disjunctive_ops: list = None):
        self.job_id = job_id
        self.step_id = step_id
        self.job = job
        self._id = (job_id, step_id)
        self.machine_id = machine_id
        self.node_status = NOT_START_NODE
        self.complete_ratio = complete_ratio
        self.prev_op = prev_op
        self.delayed_time = 0
        self.processing_time = int(processing_time)
        self.remaining_time = - np.inf
        self.remaining_ops = self.job.num_sequence - (self.step_id + 1)
        self.waiting_time = 0
        self._next_op = next_op
        self._disjunctive_ops = disjunctive_ops

        self.next_op_built = False
        self.disjunctive_built = False
        self.built = False

    def __str__(self):
        return "job {} step {}".format(self.job_id, self.step_id)

    def processible(self):
        """
        Determine whether the operation is processable.

        Returns:
            True indicates that the operation can be processed。
        """
        prev_none = self.prev_op is None
        if self.prev_op is not None:
            prev_done = self.prev_op.node_status is DONE_NODE
        else:
            prev_done = False
        return prev_done or prev_none

    @property
    def id(self):
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id

    @property
    def disjunctive_ops(self):
        return self._disjunctive_ops

    @disjunctive_ops.setter
    def disjunctive_ops(self, disj_ops: list):
        for ops in disj_ops:
            if not isinstance(ops, Operation):
                raise RuntimeError("Given {} is not Operation instance".format(ops))
        self._disjunctive_ops = disj_ops
        self.disjunctive_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def next_op(self):
        return self._next_op

    @next_op.setter
    def next_op(self, next_op):
        self._next_op = next_op
        self.next_op_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def x(self):
        """
        返回该工序（结点）所包含的信息。

        Returns:
            OrderedDict:
                * 'id': tuple。包含(job_id, step_id)。
                * 'type': int。-1表示未完成，0表示加工中，1表示已完成。
                * 'complete_ratio': float。表示完成率。
                * 'processing_time': int。表示完成该工序需要的加工时间。
                * 'remaining_ops': int。后续工序数。
                * 'waiting_time': int。已等待时间。
                * 'remain_time': int。加工剩余时间，未处于加工中时为0。
                * 'doable': bool。true表示该工序当前是可做的。否则为不可做。
        """
        not_start_cond = (self.node_status == NOT_START_NODE)
        delayed_cond = (self.node_status == DELAYED_NODE)
        processing_cond = (self.node_status == PROCESSING_NODE)
        done_cond = (self.node_status == DONE_NODE)

        if not_start_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = self.waiting_time
            _x["remain_time"] = 0
        elif processing_cond or done_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = 0
            _x["remain_time"] = self.remaining_time
        elif delayed_cond:
            raise NotImplementedError("delayed operation")
        else:
            raise RuntimeError("Not supporting node type")
        return _x
