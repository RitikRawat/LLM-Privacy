from utilities import train_gpt2
import torch





if __name__ == '__main__' :
    train_gpt2("unredacted_dataset_small.txt", "models/model_unredacted", epochs=1, batch_size=2)