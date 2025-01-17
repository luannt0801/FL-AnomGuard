import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import random
import string
import torch
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split



class Data():
    def __init__(self, name_data, num_data, num_class, label_drops):
        self.num_data = num_data
        self.num_class = num_class
        self.label_drops = label_drops
        self.name_data = name_data

        self.trainset, self.testset = self._dataset_install()
        self.distribution_data = self._generate_array()

    def _generate_array(self):
        """
        Create a 1xN array with the sum of its elements equal to `total` and different distribution values.

        :param total: The sum of the elements in the array
        :param size: The size of the array
        :param label: List of indices to be set to 0
        :return: A 1xN array with the sum of its elements equal to `total`
        """
        total = self.num_data
        size = self.num_class
        label = self.label_drops

        proportions = np.random.dirichlet(np.ones(size), size=1)[0]
        array = np.round(proportions * total).astype(int)

        diff = total - np.sum(array)
        if diff != 0:
            array[np.argmax(array)] += diff

        for idx in label:
            diff = array[idx]
            array[idx] = 0
            indices = [i for i in range(size) if i not in label]
            array[indices] += np.random.multinomial(diff, [1/len(indices)] * len(indices))
        
        return array
    
    @staticmethod
    def pad_sequences(encoded_domains, maxlen):
        domains = []
        for domain in encoded_domains:
            if len(domain) >= maxlen:
                domains.append(domain[:maxlen])
            else:
                domains.append([0]*(maxlen-len(domain))+domain)
        return np.asarray(domains)
    
    def _dataset_install(self):
        if self.name_data == 'cifar10':
            path = "D:\\2025\\Projects\\Federated Learning with the raw meat in the dish\\data\\images"
            transform = transforms.Compose([transforms.RandomGrayscale(0.2),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.RandomVerticalFlip(0.2),
                                          transforms.RandomRotation(30),
                                          transforms.RandomAdjustSharpness(0.4),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                         ])
            transform_test = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                        )
            
            trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
            testset = datasets.CIFAR10(root=path, train=False,
                                        download=True, transform=transform_test)
        elif self.name_data == 'dga':
            data_folder = path = "D:\\2025\\Projects\\Federated Learning with the raw meat in the dish\\data\\dga"
            dga_types = [dga_type for dga_type in os.listdir(data_folder) if os.path.isdir(f"{data_folder}/{dga_type}")]
            # print(f"Detected DGA types: {dga_types}")
            my_df = pd.DataFrame(columns=['domain', 'type', 'label'])

            for dga_type in dga_types:
                files = os.listdir(f"{data_folder}/{dga_type}")
                for file in files:
                    # num_labels += 1
                    with open(f"{data_folder}/{dga_type}/{file}", 'r') as fp:
                        lines = fp.readlines()

                        if self.num_class == 2:
                            domains_with_type = [[line.strip(), dga_type, 1] for line in lines]
                        elif self.num_class == 11:
                            label_index = dga_types.index(dga_type) + 1
                            domains_with_type = [[line.strip(), dga_type, label_index] for line in lines]
                        else:
                            raise ValueError("Please input the correct number of labels for DGA data!")

                        appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
                        my_df = pd.concat([my_df, appending_df], ignore_index=True)

            with open(f'{data_folder}/benign.txt', 'r') as fp:
                benign_lines = fp.readlines()[:]
                domains_with_type = [[line.strip(), 'benign', 0] for line in benign_lines]
                appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
                my_df = pd.concat([my_df, appending_df], ignore_index=True)
            
            # Pre-processing
            domains = my_df['domain'].to_numpy()
            labels = my_df['label'].to_numpy()

            char2ix = {x: idx + 1 for idx, x in enumerate(string.printable)}
            ix2char = {ix: char for char, ix in char2ix.items()}

            encoded_domains = [[char2ix[y] for y in x if y in char2ix] for x in domains]
            encoded_labels = labels  # Giữ nguyên nhãn từ dữ liệu

            encoded_labels = np.asarray([label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1])
            encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]

            assert len(encoded_domains) == len(encoded_labels)

            maxlen = max(len(domain) for domain in encoded_domains)  # Đặt chiều dài tối đa
            padded_domains = self.pad_sequences(encoded_domains, maxlen)

            X_train, X_test, y_train, y_test = train_test_split(padded_domains, encoded_labels, test_size=0.10, shuffle=True)
            trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
            testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))
            # this is all of the data dga
            # dga_dataset = TensorDataset(torch.tensor(padded_domains, dtype=torch.long), torch.tensor(encoded_labels, dtype=torch.long))

            # print(f"DGA dataset: {dga_dataset}")

        return trainset, testset
    
    def split_dataset_by_class(self):
        """
        Split the dataset into a subset according to the number of samples given in `distribution`.

        :param dataset: The original dataset (eg: CIFAR-10)
        :param distribution: An array containing the number of samples to take from each class
        :return: The subset of the dataset split according to `distribution`
        """
        if self.name_data == 'cifar10': # num_class = 10
            class_indices = {i: [] for i in range(self.num_class)}

            for idx, (_, label) in enumerate(self.trainset):
                class_indices[label].append(idx)
            
            selected_indices = []
            # in order
            # for class_id, num_samples in enumerate(distribution):
                # selected_indices.extend(class_indices[class_id][:num_samples])
                
            for class_id, num_samples in enumerate(self.distribution_data):
                random.shuffle(class_indices[class_id])  # Xáo trộn chỉ số
                selected_indices.extend(class_indices[class_id][:num_samples])
            subset = Subset(self.trainset, selected_indices)

        elif self.name_data == 'dga':
            class_indices = {i: [] for i in range(self.num_class)}
            print(f"check class_indices: {class_indices}")

            for idx, (_, label) in enumerate(self.trainset):
                # class_indices[label].append(idx)
                # class_indices[label.item()].append(idx)
                class_indices[int(label.item())].append(idx)

            selected_indices = []
            # in order
            # for class_id, num_samples in enumerate(distribution):
                # selected_indices.extend(class_indices[class_id][:num_samples])
            for class_id, num_samples in enumerate(self.distribution_data):
                random.shuffle(class_indices[class_id])  # Xáo trộn chỉ số
                selected_indices.extend(class_indices[class_id][:num_samples])
            subset = Subset(self.trainset, selected_indices)
        return subset, self.testset

    def count_dataset(self, dataset):
        if self.name_data == 'cifar10':
            print(f"Tổng số mẫu: {len(dataset)}")
            class_count = {i: 0 for i in range(10)}
            for _, label in dataset:
                class_count[label] += 1
            print("Số lượng mẫu mỗi lớp:", class_count) 
        elif self.name_data == 'dga':
            # labels = dataset.tensors[1].numpy()  # Lấy tensor nhãn từ dataset và chuyển sang numpy dành cho cifar10
            # dga
            indices = dataset.indices
            labels = [dataset.dataset[i][1].item() for i in indices]

            label_counts = Counter(labels)  # Đếm số lượng mỗi nhãn
            
            print("Label distribution in the dataset:")
            for label, count in sorted(label_counts.items()):
                print(f"Label {int(label)}: {count} samples")


# if __name__ == '__main__':

#     total_data_in_round = 5000
#     num_classes = 11 # dga: 11 or 1, cifar10: 10
#     labels_drop = []
#     name_data = 'dga'

#     get_data = Data(name_data=name_data, num_data=total_data_in_round, num_class=num_classes, label_drops=labels_drop)

#     trainset_round, testset = get_data.split_dataset_by_class()
#     get_data.count_dataset(trainset_round)

