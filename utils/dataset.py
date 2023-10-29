import os

from datasets import NCI1, BA3Motif, Mutagenicity, SynGraphDataset, WebDataset, bbbp


def get_datasets(name, root="data/"):
    """
    Get preloaded datasets by name
    :param name: name of the dataset
    :param root: root path of the dataset
    :return: train_dataset, test_dataset, val_dataset
    """
    if name == "mutag":
        folder = os.path.join(root, "MUTAG")
        train_dataset = Mutagenicity(folder, mode="training")
        test_dataset = Mutagenicity(folder, mode="testing")
        val_dataset = Mutagenicity(folder, mode="evaluation")
    elif name == "NCI1":
        folder = os.path.join(root, "NCI1")
        train_dataset = NCI1(folder, mode="training")
        test_dataset = NCI1(folder, mode="testing")
        val_dataset = NCI1(folder, mode="evaluation")
    elif name == "ba3":
        folder = os.path.join(root, "BA3")
        train_dataset = BA3Motif(folder, mode="training")
        test_dataset = BA3Motif(folder, mode="testing")
        val_dataset = BA3Motif(folder, mode="evaluation")
    elif name == "BA_shapes":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode="testing", name="BA_shapes")
        val_dataset = SynGraphDataset(folder, mode="evaluating", name="BA_shapes")
        train_dataset = SynGraphDataset(folder, mode="training", name="BA_shapes")
    elif name == "Tree_Cycle":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode="testing", name="Tree_Cycle")
        val_dataset = SynGraphDataset(folder, mode="evaluating", name="Tree_Cycle")
        train_dataset = SynGraphDataset(folder, mode="training", name="Tree_Cycle")
    elif name == "Tree_Grids":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode="testing", name="Tree_Grids")
        val_dataset = SynGraphDataset(folder, mode="evaluating", name="Tree_Grids")
        train_dataset = SynGraphDataset(folder, mode="training", name="Tree_Grids")
    elif name == "bbbp":
        folder = os.path.join(root, "bbbp")
        dataset = bbbp(folder)
        test_dataset = dataset[:200]
        val_dataset = dataset[200:400]
        train_dataset = dataset[400:]
    elif name == "cornell":
        folder = os.path.join(root)
        test_dataset = WebDataset(folder, mode="testing", name=name)
        val_dataset = WebDataset(folder, mode="evaluating", name=name)
        train_dataset = WebDataset(folder, mode="training", name=name)
    else:
        raise ValueError
    return train_dataset, val_dataset, test_dataset
