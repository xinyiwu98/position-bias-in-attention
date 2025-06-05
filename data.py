import torch
from torch.utils.data import Dataset
import random
import numpy as np





class SeqDataset(Dataset):
    def __init__(self, N=8, B=1, K=256, L=32, D=128, eps=0.1):

        self.N = N
        self.B = B

        self.K = K # number of clusters
        self.L = L # number of labels

        self.D = D # number of content dimension

       

        self.class_mean = {} # each cluster mean
        self.class_label = {} # each cluster label
        self.label_content = {} # each cluster content

        self.eta = torch.normal(mean=torch.tensor(0), std=torch.tensor(1/self.D), size=(self.D,)) # eta
        self.eps = eps # within class variability


        # assign the label content
        for l in range(self.L):
            content = torch.normal(mean=torch.tensor(0), std=torch.tensor(1/self.D), size=(self.D,))
            self.label_content[l] = content

        # assign the cluster mean and label
        for k in range(self.K):
            # Sample from a D-dimensional Gaussian distribution
            mu = torch.normal(mean=torch.tensor(0), std=torch.tensor(1/self.D), size=(self.D,))
            # Add the sample to the dictionary
            self.class_mean[k] = mu

            # Sample a label for each class
            label = random.randint(0, self.L-1)
            # Add the sample to the dictionary
            self.class_label[k] = label

    # generate content information
    def generate_content(self, classes, class_mean, class_label):
        content = [torch.concat((((class_mean[key] + self.eps * self.eta)/np.sqrt(1 + self.eps**2)).reshape(1,-1),
                                self.label_content[class_label[key]].reshape(1,-1)), axis=0) 
                                for key in classes[:-1]]
        content = torch.concat(content, axis=0)

        # add the query content at the end
        query_class = classes[-1]
        content = torch.concat((content, 
                                ((class_mean[query_class] + self.eps * self.eta)/np.sqrt(1 + self.eps**2)).reshape(1,-1)),
                                axis = 0)
        return content

  
class TrainDataset(SeqDataset):
    def __init__(self, pos_bias=False, index=[0], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_bias = pos_bias
        self.index = index

    def __len__(self):
        return 128*500000

    def generate_train(self):  # correct answer is either at the first or the last position
        
        if self.N < self.B:
            raise ValueError("N cannot be smaller than B.")

        if self.N % self.B != 0:
            raise ValueError("N must be divisible by B.")
       
        classes = np.random.choice(self.K, size=self.N//self.B, replace=False)
        classes = np.tile(classes, self.B)
        classes = np.random.permutation(classes)
        if self.pos_bias:
            answer_position = np.random.choice(self.index, size=1)[0]
        else: 
            answer_position = np.random.choice(self.N, size=1)[0]
        query_class = classes[answer_position]
        classes = np.append(classes, query_class)

        content = self.generate_content(classes, self.class_mean, self.class_label)

        query_label = self.class_label[query_class]
        
        return content, query_label

    def __getitem__(self, idx):
        torch.random.manual_seed(idx)
        np.random.seed(idx)
        return self.generate_train()

    

class TestDataset(SeqDataset):
    def __init__(self, num_seqs, test_type, train_dataset, pos_bias=False, index=[0], *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.contents = []
        self.labels = []
        self.class_mean = train_dataset.class_mean
        self.class_label = train_dataset.class_label
        self.label_content = train_dataset.label_content
        self.pos_bias=pos_bias
        self.index = index

        for i in range(num_seqs):
            torch.random.manual_seed(128*500000+i)
            np.random.seed(128*500000+i)
            
            if test_type == "IC_middle_first": # novel class; both first and middle example are semantically the same, but middle is correct
                self.class_mean_novel = {}
                self.class_label_novel = {}
        
                for k in range(self.K):
                    # Sample from a D-dimensional Gaussian distribution
                    mu = torch.normal(mean=torch.tensor(0), std=torch.tensor(1/self.D), size=(self.D,))
                    # Add the sample to the dictionary
                    self.class_mean_novel[k] = mu

                    # Sample a label for each class
                    label = random.randint(0, self.L-1)
                    # Add the sample to the dictionary
                    self.class_label_novel[k] = label

                # generate one test seq (B copies in the context is the same as the query)
                classes = np.random.choice(self.K, size=self.N//self.B, replace=False)
                classes = np.tile(classes, self.B)
                classes = np.random.permutation(classes)
                answer_position = self.N//2
                query_class = classes[answer_position]
                classes = np.append(classes, query_class)

                content = self.generate_content(classes, self.class_mean_novel, self.class_label_novel)

                content[0, :] = content[self.N, :] # making first and middle example semantically the same 

                query_label = self.class_label_novel[query_class]
                
                self.contents.append(content)
                self.labels.append(query_label) 


            elif test_type == "IC_first_middle": # novel class; both first and middle example are semantically the same, but first is correct
                self.class_mean_novel = {}
                self.class_label_novel = {}
        
                for k in range(self.K):
                    # Sample from a D-dimensional Gaussian distribution
                    mu = torch.normal(mean=torch.tensor(0), std=torch.tensor(1/self.D), size=(self.D,))
                    # Add the sample to the dictionary
                    self.class_mean_novel[k] = mu

                    # Sample a label for each class
                    label = random.randint(0, self.L-1)
                    # Add the sample to the dictionary
                    self.class_label_novel[k] = label

                # generate one test seq (B copies in the context is the same as the query)
                classes = np.random.choice(self.K, size=self.N//self.B, replace=False)
                classes = np.tile(classes, self.B)
                classes = np.random.permutation(classes)
                answer_position = 0
                query_class = classes[answer_position]
                classes = np.append(classes, query_class)

                content = self.generate_content(classes, self.class_mean_novel, self.class_label_novel)

                content[self.N, :] = content[0, :] # making first and middle example semantically the same 

                query_label = self.class_label_novel[query_class]
                
                self.contents.append(content)
                self.labels.append(query_label) 


            elif test_type == "IC_last_first": # novel class; both first and last example are semantically the same, but first is correct
                self.class_mean_novel = {}
                self.class_label_novel = {}
        
                for k in range(self.K):
                    # Sample from a D-dimensional Gaussian distribution
                    mu = torch.normal(mean=torch.tensor(0), std=torch.tensor(1/self.D), size=(self.D,))
                    # Add the sample to the dictionary
                    self.class_mean_novel[k] = mu

                    # Sample a label for each class
                    label = random.randint(0, self.L-1)
                    # Add the sample to the dictionary
                    self.class_label_novel[k] = label

                # generate one test seq (B copies in the context is the same as the query)
                classes = np.random.choice(self.K, size=self.N//self.B, replace=False)
                classes = np.tile(classes, self.B)
                classes = np.random.permutation(classes)
                answer_position = -1
                query_class = classes[answer_position]
                classes = np.append(classes, query_class)

                content = self.generate_content(classes, self.class_mean_novel, self.class_label_novel)

                content[0, :] = content[-3, :] # making first and middle example semantically the same 

                query_label = self.class_label_novel[query_class]
                
                self.contents.append(content)
                self.labels.append(query_label) 

            elif test_type == "IC_first_last": # novel class; both first and last example are semantically the same, but first is correct
                self.class_mean_novel = {}
                self.class_label_novel = {}
        
                for k in range(self.K):
                    # Sample from a D-dimensional Gaussian distribution
                    mu = torch.normal(mean=torch.tensor(0), std=torch.tensor(1/self.D), size=(self.D,))
                    # Add the sample to the dictionary
                    self.class_mean_novel[k] = mu

                    # Sample a label for each class
                    label = random.randint(0, self.L-1)
                    # Add the sample to the dictionary
                    self.class_label_novel[k] = label

                # generate one test seq (B copies in the context is the same as the query)
                classes = np.random.choice(self.K, size=self.N//self.B, replace=False)
                classes = np.tile(classes, self.B)
                classes = np.random.permutation(classes)
                answer_position = 0
                query_class = classes[answer_position]
                classes = np.append(classes, query_class)

                content = self.generate_content(classes, self.class_mean_novel, self.class_label_novel)

                content[-3, :] = content[0, :] # making first and middle example semantically the same 

                query_label = self.class_label_novel[query_class]
                
                self.contents.append(content)
                self.labels.append(query_label) 


            elif test_type == "IC_last_middle": # novel class; both first and last example are semantically the same, but first is correct
                self.class_mean_novel = {}
                self.class_label_novel = {}
        
                for k in range(self.K):
                    # Sample from a D-dimensional Gaussian distribution
                    mu = torch.normal(mean=torch.tensor(0), std=torch.tensor(1/self.D), size=(self.D,))
                    # Add the sample to the dictionary
                    self.class_mean_novel[k] = mu

                    # Sample a label for each class
                    label = random.randint(0, self.L-1)
                    # Add the sample to the dictionary
                    self.class_label_novel[k] = label

                # generate one test seq (B copies in the context is the same as the query)
                classes = np.random.choice(self.K, size=self.N//self.B, replace=False)
                classes = np.tile(classes, self.B)
                classes = np.random.permutation(classes)
                answer_position = -1
                query_class = classes[answer_position]
                classes = np.append(classes, query_class)

                content = self.generate_content(classes, self.class_mean_novel, self.class_label_novel)

                content[self.N, :] = content[-3, :] # making first and middle example semantically the same 

                query_label = self.class_label_novel[query_class]
                
                self.contents.append(content)
                self.labels.append(query_label) 

            elif test_type == "IC_middle_last": # novel class; both first and last example are semantically the same, but first is correct
                self.class_mean_novel = {}
                self.class_label_novel = {}
        
                for k in range(self.K):
                    # Sample from a D-dimensional Gaussian distribution
                    mu = torch.normal(mean=torch.tensor(0), std=torch.tensor(1/self.D), size=(self.D,))
                    # Add the sample to the dictionary
                    self.class_mean_novel[k] = mu

                    # Sample a label for each class
                    label = random.randint(0, self.L-1)
                    # Add the sample to the dictionary
                    self.class_label_novel[k] = label

                # generate one test seq (B copies in the context is the same as the query)
                classes = np.random.choice(self.K, size=self.N//self.B, replace=False)
                classes = np.tile(classes, self.B)
                classes = np.random.permutation(classes)
                answer_position = self.N//2
                query_class = classes[answer_position]
                classes = np.append(classes, query_class)

                content = self.generate_content(classes, self.class_mean_novel, self.class_label_novel)

                content[-3, :] = content[self.N, :] # making first and middle example semantically the same 

                query_label = self.class_label_novel[query_class]
                
                self.contents.append(content)
                self.labels.append(query_label) 


    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        return self.contents[idx], self.labels[idx]