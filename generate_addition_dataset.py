import random

def generate_addition_dataset(num_samples, max_value=999):
    dataset = []
    for _ in range(num_samples):
        a = random.randint(0, max_value)
        b = random.randint(0, max_value)
        c = a + b

        # Create the input-output pair in the format "a+b=c"
        example = f"{a:03}+{b:03}={int(str(c)[::-1]):04}"
        dataset.append(example)

    return dataset

def save_dataset_to_file(dataset, filename):
    with open(filename, 'w') as file:
        for example in dataset:
            file.write(example + '\n')

# Example: Generate a dataset with 100 samples and save it to a file
num_samples = 40000
dataset = generate_addition_dataset(num_samples)
save_dataset_to_file(dataset, 'addition_dataset.txt')