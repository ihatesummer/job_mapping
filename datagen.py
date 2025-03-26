import networkx as nx
import random
import pickle
import os
import matplotlib.pyplot as plt
from collections import deque

def create_subfolder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

class HPCJobGenerator:
    def __init__(self, N, num_jobs_per_queue, min_nodes, max_nodes, max_time):
        self.N = N
        self.num_jobs_per_queue = num_jobs_per_queue
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.max_time = max_time
        self.queues = [deque() for _ in range(N)]
        self.job_list = [[] for _ in range(N)]  # Pre-determined job list with arrival times

    def generate_task(self):
        """Generate an acyclic (DAG) HPC task graph with workload attributes."""
        num_nodes = random.randint(self.min_nodes, self.max_nodes)
        G = nx.DiGraph()

        # Assign workload attributes
        workload_options = [
            (random.randint(10, 50), random.randint(5, 30), random.randint(1, 10))
            for _ in range(num_nodes)
        ]

        for i in range(num_nodes):
            G.add_node(i, cpu=workload_options[i][0], gpu=workload_options[i][1], memory=workload_options[i][2])

        # Ensure connectivity with a spanning tree
        nodes = list(G.nodes)
        random.shuffle(nodes)
        source_node = nodes[0]
        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i + 1])

        # Add extra random edges ensuring acyclic property
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if j != source_node and random.random() < 0.5 and not G.has_edge(j, i):
                    G.add_edge(i, j)
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(i, j)

        return G

    def generate_job_list(self):
        """Pre-generate job list with arrival times for each queue."""
        for queue_id in range(self.N):
            for _ in range(self.num_jobs_per_queue):
                arrival_time = random.randint(0, self.max_time)
                task = self.generate_task()
                self.job_list[queue_id].append((arrival_time, task))

            # Sort jobs by arrival time for each queue
            self.job_list[queue_id].sort(key=lambda x: x[0])

    def fill_queues(self):
        """Load jobs into queues in order of their pre-determined arrival times."""
        for queue_id in range(self.N):
            for arrival_time, task in self.job_list[queue_id]:
                self.queues[queue_id].append((arrival_time, task))

    def print_order(self, task, arrival_time, queue_id, job_id):
        indegree = {node: 0 for node in task.nodes}
        for u, v in task.edges:
            indegree[v] += 1
        
        # Array to store the time each job is completed
        timer = {node: arrival_time for node in task.nodes}
        
        # Initialize queue with nodes having indegree 0
        q = deque()
        for node in task.nodes:
            if indegree[node] == 0:
                q.append(node)
                timer[node] += 1 # TODO make a calculation rule for the job execution time, to replace 1
        
        # Process nodes in topological order
        while q:
            cur = q.popleft()
            for adj in task.successors(cur):
                indegree[adj] -= 1
                if indegree[adj] == 0:
                    timer[adj] = timer[cur] + 1
                    q.append(adj)
        
        sorted_nodes = sorted(timer.items(), key=lambda x: x[1])
        print(f"\nQueue {queue_id} job {job_id}:")
        for node, time in sorted_nodes:
            print(f"Node {node} at time {time} - ", end=" ")

    def print_all_orders(self):
        """Export all job graphs as images."""
        for queue_id, job_list in enumerate(self.job_list):
            for job_id, (arrival_time, task) in enumerate(job_list):
                self.print_order(task, arrival_time, queue_id, job_id)

    def save_task_graph(self, task, queue_id, job_id):
        """Save a visualization of the task graph as a pickle, an image and a text description."""
        save_path = create_subfolder(f"./data/graphs/")

        # --- Save a NetworkX graph as a .pkl file ---
        filename = save_path + f"queue_{queue_id}_job_{job_id}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(task, f)

        # --- Save Graph as an Image ---
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(task, seed=42)

        # Labels with workload values
        labels = {
            node: f"PID:{node}\nCPU:{task.nodes[node]['cpu']}\nGPU:{task.nodes[node]['gpu']}\nMEM:{task.nodes[node]['memory']}"
            for node in task.nodes
        }
        nx.draw(task, pos, with_labels=True, node_color="lightblue", edge_color="gray",
                node_size=5000, font_size=10, font_weight="bold", width=3, arrows=True, arrowsize=20)
        nx.draw_networkx_labels(task, pos, labels=labels, font_size=10, font_color="black",
                                verticalalignment='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        filename = save_path + f"queue_{queue_id}_job_{job_id}.png"
        plt.savefig(filename, format="png", dpi=300)
        plt.close()
    
        # --- Save Graph Description as a Text File ---
        filename = save_path + f"queue_{queue_id}_job_{job_id}.txt"
        with open(filename, "w") as f:
            f.write(f"Task Graph for Queue {queue_id}, Job {job_id}\n")
            f.write("Nodes and Workload Attributes:\n")
            for node in task.nodes:
                cpu = task.nodes[node]['cpu']
                gpu = task.nodes[node]['gpu']
                memory = task.nodes[node]['memory']
                f.write(f"  Node {node}: CPU={cpu}, GPU={gpu}, MEM={memory}\n")

            f.write("\nDirected Edges:\n")
            for src, dst in task.edges:
                f.write(f"  {src} -> {dst}\n")

    def export_all_jobs(self):
        """Export all job graphs as images."""
        for queue_id, job_list in enumerate(self.job_list):
            for job_id, (arrival_time, task) in enumerate(job_list):
                self.save_task_graph(task, queue_id, job_id)

if __name__ == "__main__":
    N = 3  # Number of queues
    generator = HPCJobGenerator(N, num_jobs_per_queue=5, min_nodes=2, max_nodes=5, max_time=50)
    generator.generate_job_list()
    generator.fill_queues()

    # Print job list for each queue
    for i in range(N):
        print(f"Queue {i}:")
        for arrival_time, task in generator.job_list[i]:
            print(f"  Arrival Time {arrival_time}: Task with {task.number_of_nodes()} nodes")
    # Export job graphs
    generator.export_all_jobs()
    generator.print_all_orders()
