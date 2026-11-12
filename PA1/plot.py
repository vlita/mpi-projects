import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# data
processors = [1, 6, 12, 24]
runtime = [39.8287, 6.64604, 3.32371, 1.70345]

# plot
plt.figure()
plt.scatter(processors, runtime, marker='o')
plt.xlabel("Number of processors")
plt.ylabel("Run time (s)")
plt.xticks(processors)
# plt.title("T(n,p) vs. p")
plt.grid(False)

# save figure for LaTeX
plt.savefig("runtime_vs_processors.pdf", bbox_inches="tight")
plt.show()