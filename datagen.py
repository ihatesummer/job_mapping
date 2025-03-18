import networkx as nx
import random
from collections import deque
import matplotlib.pyplot as plt

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
                arrival_time = random.randint(0, self.max_time)  # Fixed arrival time
                task = self.generate_task()
                self.job_list[queue_id].append((arrival_time, task))

            # Sort jobs by arrival time for each queue
            self.job_list[queue_id].sort(key=lambda x: x[0])

    def fill_queues(self):
        """Load jobs into queues in order of their pre-determined arrival times."""
        for queue_id in range(self.N):
            for arrival_time, task in self.job_list[queue_id]:
                self.queues[queue_id].append((arrival_time, task))

    def save_task_graph(self, task, queue_id, job_id):
        """Save a visualization of the task graph as an image."""
        plt.figure(figsize=(6, 6))  # Slightly larger figure size
        pos = nx.spring_layout(task, seed=42)

        # Labels with workload values
        labels = {node: f"CPU:{task.nodes[node]['cpu']}\nGPU:{task.nodes[node]['gpu']}\nMEM:{task.nodes[node]['memory']}"
                  for node in task.nodes}
        nx.draw(task, pos, with_labels=True, node_color="lightblue", edge_color="gray",
                node_size=5000, font_size=10, font_weight="bold", width=3, arrows=True, arrowsize=20)
        nx.draw_networkx_labels(task, pos, labels=labels, font_size=10, font_color="black",
                                verticalalignment='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        filename = f"queue_{queue_id}_job_{job_id}.png"
        plt.savefig(filename, format="png", dpi=300)
        plt.close()
    
    def export_all_jobs(self):
        """Export all job graphs as images."""
        for queue_id, job_list in enumerate(self.job_list):
            for job_id, (arrival_time, task) in enumerate(job_list):
                self.save_task_graph(task, queue_id, job_id)

# Example usage
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
