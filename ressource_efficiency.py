import time
import psutil
import matplotlib.pyplot as plt


def measure_resource_usage(model, dataset, incremental=True):
    cpu_usage = []
    memory_usage = []

    for i in range(len(dataset)):
        data_point = dataset.iloc[i]

        if incremental:
            # Measure CPU and memory usage before processing the data point
            start_cpu = psutil.cpu_percent()
            start_memory = psutil.virtual_memory().percent

            # Process the data point with your incremental model
            # Replace this with the code to process the data point with your model
            # ...

            # Measure CPU and memory usage after processing the data point
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory().percent
        else:
            # For the batch model, measure resource usage without increments
            # Process the data point with your batch model
            # Replace this with the code to process the data point with your batch model
            # ...

            # Measure CPU and memory usage after processing the data point
            start_cpu = end_cpu = psutil.cpu_percent()
            start_memory = end_memory = psutil.virtual_memory().percent

        cpu_usage.append(end_cpu - start_cpu)
        memory_usage.append(end_memory - start_memory)

    return cpu_usage, memory_usage


# Example usage:
# Replace 'your_incremental_model' and 'your_batch_model' with your actual model instances.
# Replace 'your_incremental_dataset' and 'your_batch_dataset' with your dataset DataFrames.
incremental_cpu, incremental_memory = measure_resource_usage(your_incremental_model, your_incremental_dataset,
                                                             incremental=True)
batch_cpu, batch_memory = measure_resource_usage(your_batch_model, your_batch_dataset, incremental=False)

# Plot the resource usage comparison
plt.figure(figsize=(10, 6))
plt.bar([1, 2], [sum(incremental_cpu), sum(batch_cpu)], tick_label=["Incremental", "Batch"], color=['blue', 'green'])
plt.title('CPU Usage Comparison')
plt.xlabel('Model Type')
plt.ylabel('CPU Usage')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar([1, 2], [sum(incremental_memory), sum(batch_memory)], tick_label=["Incremental", "Batch"],
        color=['blue', 'green'])
plt.title('Memory Usage Comparison')
plt.xlabel('Model Type')
plt.ylabel('Memory Usage')
plt.show()
