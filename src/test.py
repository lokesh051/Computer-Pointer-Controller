import matplotlib.pyplot as plt

precision_list = ['FP16', 'FP32', 'FP32-INT8']
inference_time = []
loading_time = []
fps = []

for precision in precision_list:
    with open('result-' + precision+ '.txt', 'r') as f:
        loading_time.append(float(f.readline().split('\n')[0]))
        inference_time.append(float(f.readline().split('\n')[0]))
        fps.append(float(f.readline().split('\n')[0]))



plt.bar(precision_list, inference_time)
plt.xlabel('Model Precision Value')
plt.ylabel('Inference Time in seconds')
plt.title('Different Precision Comparison Graph')
plt.show()


plt.bar(precision_list, fps)
plt.xlabel('Model Precision Value')
plt.ylabel('Frames per Second')
plt.title('Different Precision Comparison Graph')
plt.show()


plt.bar(precision_list, loading_time)
plt.xlabel('Model Precision Value')
plt.ylabel('Model Load Time')
plt.title('Different Precision Comparison Graph')
plt.show()
