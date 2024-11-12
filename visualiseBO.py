
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def readFile(runName):
    inputs = []
    yMean = []
    ystd = []
    yMV = []

    readFolder = 'results/' + str(runName) + '/data.txt'
    with open(readFolder, 'r') as file:
        for line in file:
            # Splitting the line into parts
            parts = line.strip().split('] ')
            if len(parts) >= 2:
                input_part = parts[0] + ']'
                try:
                    input_list = eval(input_part)
                    inputs.append(input_list)
                except:
                    print(f"Error processing input: {input_part}")

                yMean_part = parts[1] + ']'
                try:
                    yMean_list = eval(yMean_part)
                    yMean.append(yMean_list)
                except:
                    print(f"Error processing input: {yMean}")

                ystd_part = parts[2] + ']'
                try:
                    ystd_list = eval(ystd_part)
                    ystd.append(ystd_list)
                except:
                    print(f"Error processing input: {ystd}")

                yMV_part = parts[3] 
                try:
                    yMV_list = eval(yMV_part)
                    yMV.append(yMV_list)
                except:
                    print(f"Error processing input: {yMV}")

    return inputs, yMean, ystd, yMV 

if __name__ == "__main__":
    runName = "test3"
    n_random_samples = 5

    inputs, yMean, ystd, yMV = readFile(runName)
    power = [x[0] for x in inputs]
    radius = [x[1] for x in inputs]

    power = np.array(power)
    radius = np.array(radius)
    yMean = np.array(yMean)
    ystd = np.array(ystd)
    yMV = np.array(yMV)

    print(yMV)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42                               
    matplotlib.rcParams['font.family'] = 'serif'                          
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.size'] = '12'
    titles = ['Colored by Mean', 'Colored by Standard Deviation', 'Colored by Mean-Variance']

    data_list = [yMean, ystd, yMV]
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), dpi=300)

    for i in range(len(data_list)):
        c_values = data_list[i].squeeze()  # Ensure c_values is a 1D array
        sc = axs[i].scatter(power, radius, c=c_values, cmap='cool', s=80, label="Random Samples")
        sc = axs[i].scatter(power[n_random_samples:], radius[n_random_samples:], c=c_values[n_random_samples:], cmap='cool', edgecolors='black', s=80, linewidths=2, label="BO Samples")
        axs[i].set_xlabel('Power')
        axs[i].set_ylabel('Radius')
        axs[i].set_title(titles[i])
        plt.colorbar(sc, ax=axs[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/'+ str(runName) + '/colorPlot.png', dpi=300)
    plt.close()

    yMV_max = [max(yMV[:(i+1)]) for i in range(len(yMV))]
    plt.figure(figsize=(10, 6))
    plt.plot(yMV_max, label='Mean-Variance*')
    plt.ylabel('Maximum Observed Mean-Variance')
    plt.xlabel('Iteration')
    plt.axvline(x=n_random_samples, color='red', linestyle='--', label='Random Samples')
    plt.legend()
    plt.savefig('results/'+ str(runName) + '/MVMax.png', dpi=300)
    plt.close()

