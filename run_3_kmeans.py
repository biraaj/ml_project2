from functions import KMeans,load_data_kmeans

program_data_main = "./data/program_data_2c_2d_2e_hw1.txt"

train_features,train_targets = load_data_kmeans(program_data_main,0)

print("Results of Q3 Kmeans")
k_values = [3,6,9]
for _k in k_values:
    print("**********************************")
    print("for K = ",_k)
    model = KMeans(train_features,_k,10)
    model.train()
    print("Overall accuracy = ",model.accuracy(train_targets))
    print("**********************************")
    print()

